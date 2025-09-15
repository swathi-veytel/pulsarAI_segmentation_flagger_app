import streamlit as st
import pandas as pd
import os
import json
from io import BytesIO
from google.cloud import storage
from PIL import Image
import requests
import numpy as np

import time
from google.cloud.exceptions import NotFound

# Set wide layout and sidebar width
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            min-width: 350px;
            width: 350px;
        }
        div[data-testid="stNumberInput"] input {
            height: 2.2em;
            font-size: 1.1em;
        }
        div[data-testid="stForm"] {
            padding: 2rem;
        }
        div[data-testid="stImage"] img {
            max-width: 100%;
            height: auto;
        }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIG ---
GCS_BUCKET = "veytel-cloud-store"
MASTER_CSV_PATH = "pulsarai_masters/master_250723.csv"
SERVICE_ACCOUNT_JSON = st.secrets["gcs_service_account"]
VALID_USERS = sorted(["Ellen", "Cathy", "Robin", "Anrey", "Song", "Kevin", "Swathi", "Nitya", "Manasvi",
                      "Rachel", "Mike", "Paul", "Test_1", "Test_2", "Test_3"])
PASSWORD = "PulsarAIFlagger!"

# --- NEW: GCS paths for per-user logs ---
APP_ROOT   = "pulsarai_flagger_app"
USERS_ROOT = f"{APP_ROOT}/users"
def user_flag_log_path(user: str) -> str:
    return f"{USERS_ROOT}/{user}/{user}_flag_log.json"
def user_page_log_path(user: str) -> str:
    return f"{USERS_ROOT}/{user}/{user}_page_log.json"

# --- GCS CLIENT ---
def get_gcs_client():
    return storage.Client.from_service_account_info(SERVICE_ACCOUNT_JSON)

# --- GCS helpers ---
def gcs_read_json(bucket_name: str, blob_path: str, default):
    client = get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_path)
    try:
        return json.loads(blob.download_as_bytes().decode("utf-8"))
    except NotFound:
        return default
def gcs_write_json(bucket_name: str, blob_path: str, obj):
    client = get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_path)
    blob.upload_from_string(json.dumps(obj, indent=2), content_type="application/json")
def gcs_list_json_paths(bucket_name: str, prefix: str):
    client = get_gcs_client()
    return [b.name for b in client.list_blobs(bucket_name, prefix=prefix) if b.name.endswith(".json")]

@st.cache_data(show_spinner=False)
def load_image_np_from_blob(blob_path, size=(256, 256), mode="RGB"):
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_path)
    content = blob.download_as_bytes()
    img = Image.open(BytesIO(content)).convert(mode).resize(size)
    return np.array(img)

def overlay_mask_on_image_np(base_np, mask_array, color=(0, 255, 0), alpha=128):
    base_img = Image.fromarray(base_np).convert("RGBA")
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    pixels = overlay.load()
    for y in range(base_img.size[1]):
        for x in range(base_img.size[0]):
            if mask_array[y, x] > 0:
                pixels[x, y] = color + (alpha,)
    return np.array(Image.alpha_composite(base_img, overlay).convert("RGB"))

def load_user_page_data(user: str) -> dict:
    # default includes assigned_page = 1
    return gcs_read_json(GCS_BUCKET, user_page_log_path(user), default={"page_number": 1, "assigned_page": 1})

def save_user_page_data(user: str, page_number: int = None, assigned_page: int = None):
    data = load_user_page_data(user)
    if page_number is not None:
        data["page_number"] = int(page_number)
    if assigned_page is not None:
        data["assigned_page"] = int(assigned_page)
    gcs_write_json(GCS_BUCKET, user_page_log_path(user), data)

# --- LOGIN ---
def login():
    st.title("\U0001FA7B PulsarAI Segmentation Flagging Tool")
    st.subheader("Login")
    if "login_attempted" not in st.session_state:
        st.session_state.login_attempted = False
    user = st.selectbox("Username", VALID_USERS, key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login", use_container_width=True):
        if password == PASSWORD:
            st.session_state.user = user
            st.session_state.login_attempted = False
            st.rerun()
        else:
            st.session_state.login_attempted = True
    if st.session_state.login_attempted:
        st.error("Invalid password")

# --- LOADERS ---
@st.cache_data
def load_master():
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(MASTER_CSV_PATH)
    content = blob.download_as_bytes()
    return pd.read_csv(BytesIO(content))

# --- per-user flags helpers ---
def load_user_flags(user: str) -> set:
    return set(gcs_read_json(GCS_BUCKET, user_flag_log_path(user), default=[]))

def save_user_flags(user: str, flags_set: set):
    gcs_write_json(GCS_BUCKET, user_flag_log_path(user), sorted(list(flags_set)))

# --- aggregated flags map ---
def load_flag_log(cache_bust: int):
    @st.cache_data(show_spinner=False)
    def _load_all_flags(_bust: int):
        paths = gcs_list_json_paths(GCS_BUCKET, USERS_ROOT + "/")
        img_to_users = {}
        for path in paths:
            if not path.endswith("_flag_log.json"):
                continue
            u = os.path.basename(path).split("_flag_log.json")[0]
            flagged_list = gcs_read_json(GCS_BUCKET, path, default=[])
            for img in flagged_list:
                img_key = os.path.basename(str(img)).strip()
                img_to_users.setdefault(img_key, set()).add(u)
        return {k: sorted(list(v)) for k, v in img_to_users.items()}
    return _load_all_flags(cache_bust)

# --- apply pending updates ---
def flush_pending_flags_to_gcs(user: str):
    pending = st.session_state.get("pending_flag_updates", {})
    if not pending:
        return False
    user_set = load_user_flags(user)
    changed = False
    for img_name, is_flagged in pending.items():
        img_key = os.path.basename(str(img_name)).strip()
        if is_flagged and img_key not in user_set:
            user_set.add(img_key); changed = True
        elif (not is_flagged) and img_key in user_set:
            user_set.remove(img_key); changed = True
    if changed:
        save_user_flags(user, user_set)
        st.session_state["flags_refresh"] = st.session_state.get("flags_refresh", 0) + 1
    st.session_state["pending_flag_updates"] = {}
    return changed

# --- MAIN ---
def main():
    user = st.session_state.user
    page_data = load_user_page_data(user)
    page_number = page_data.get("page_number", 1)
    assigned_page = page_data.get("assigned_page", 1)

    flags_refresh = st.session_state.get("flags_refresh", 0)
    default_settings = {"page_number": page_number, "page_size": 25}

    master_df = load_master()
    master_df["imgName"] = master_df["imgName"].astype(str).str.strip()

    flag_log = load_flag_log(flags_refresh)
    all_flagged_set = set(flag_log.keys())
    user_flag_set = load_user_flags(user)

    with st.sidebar:
        st.markdown("\U0001FA7B PulsarAI Segmentation Flagging Tool")
        st.markdown(f"**👤 Logged in as:** `{user}`")

        # Show assigned range if available
        assigned_start = page_data.get("assigned_start")
        assigned_end = page_data.get("assigned_end")
        if assigned_start and assigned_end:
            st.markdown(f"**📖 Assigned Pages:** {assigned_start}-{assigned_end}")

        st.markdown("### View Options")

        view_mode = st.selectbox("View Mode", ["All Images", "Flagged Images", "Flagged by Selected Users"], key="view_mode_select")
        if "prev_view_mode" not in st.session_state:
            st.session_state.prev_view_mode = view_mode

        selected_users = st.multiselect("Select Users", VALID_USERS) if view_mode == "Flagged by Selected Users" else []

        if view_mode == "Flagged Images":
            view_df = master_df[master_df["imgName"].isin(all_flagged_set)]
        elif view_mode == "Flagged by Selected Users":
            sel = set(selected_users)
            imgs = {img for img, users in flag_log.items() if sel & set(users)}
            view_df = master_df[master_df["imgName"].isin(imgs)]
        else:
            view_df = master_df

        if "page_number" not in st.session_state:
            st.session_state.page_number = default_settings["page_number"]
        st.session_state.page_size = 25

        if st.session_state.prev_view_mode != view_mode and view_mode == "All Images":
            st.session_state.page_number = page_data.get("page_number", 1)

        total_pages = max(1, (len(view_df) - 1) // st.session_state.page_size + 1)
        st.session_state.page_number = max(1, min(st.session_state.page_number, total_pages))

        st.markdown(f"### page {st.session_state.page_number} / {total_pages}")
        st.markdown(f"#### Images per page: 25")

        # --- Jump to Page dropdown ---
        st.markdown("### Jump To Page")
        page_labels = [f"Page {i} ({(i-1)*st.session_state.page_size+1}-{min(i*st.session_state.page_size, len(view_df))})"
                       for i in range(1, total_pages+1)]
        selected_page_label = st.selectbox("Select Page", options=page_labels, index=st.session_state.page_number-1)
        selected_page_num = page_labels.index(selected_page_label) + 1
        if st.button("Go to Page", use_container_width=True, key="jump_page_btn"):
            st.session_state.page_number = selected_page_num
            st.rerun()

        st.markdown("### ⬇ Export Options")
        export_option = st.selectbox(
            "Select export option",
            ["Select...", "⬇ Full Master file with Flagged column", "⬇ Master file without Flagged Images"]
        )

        if export_option == "⬇ Full Master file with Flagged column":
            export_df_full = master_df.copy()
            export_df_full["flagged_by"] = export_df_full["imgName"].map(lambda x: ", ".join(flag_log.get(x, [])))
            export_data = export_df_full.to_csv(index=False).encode()
            export_filename = "master_with_flagged.csv"
        elif export_option == "⬇ Master file without Flagged Images":
            export_df_unflagged = master_df[~master_df["imgName"].isin(all_flagged_set)].copy()
            export_data = export_df_unflagged.to_csv(index=False).encode()
            export_filename = "master_unflagged.csv"
        else:
            export_data = None
            export_filename = None

        st.download_button(
            "Export", data=export_data if export_data else "Select a valid export option".encode(),
            file_name=export_filename if export_filename else "invalid.csv",
            use_container_width=True, disabled=export_data is None
        )

        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.session_state.prev_view_mode = view_mode

    # --- Page Slice ---
    start_idx = (st.session_state.page_number - 1) * st.session_state.page_size
    end_idx = start_idx + st.session_state.page_size
    page_df = view_df.iloc[start_idx:end_idx]

    # --- FORM for Flags ---
    with st.form("flagging_form", clear_on_submit=False):
        for idx, row in page_df.iterrows():
            img_name = row.get("imgName", "N/A")
            st.markdown("---")
            st.markdown(f"**{idx + 1}. Image Name:** `{img_name}`")

            norm_np = load_image_np_from_blob(row["normalizedPath"])
            old_mask = load_image_np_from_blob(row["maskPath_old"], mode="L")
            new_mask = load_image_np_from_blob(row["maskPath"], mode="L")

            norm_pil = Image.fromarray(norm_np)
            old_overlay = Image.fromarray(overlay_mask_on_image_np(norm_np, old_mask, color=(0, 255, 0)))
            new_overlay = Image.fromarray(overlay_mask_on_image_np(norm_np, new_mask, color=(255, 0, 0)))
            both_overlay = Image.fromarray(
                overlay_mask_on_image_np(
                    overlay_mask_on_image_np(norm_np, old_mask, color=(0, 255, 0), alpha=128),
                    new_mask, color=(255, 0, 0), alpha=128
                )
            )

            cols = st.columns(4, gap="small")
            with cols[0]: st.image(norm_pil, caption="Normalized", use_container_width=True)
            with cols[1]: st.image(old_overlay, caption="Old Mask Overlay", use_container_width=True)
            with cols[2]: st.image(new_overlay, caption="New Mask Overlay", use_container_width=True)
            with cols[3]: st.image(both_overlay, caption="Intersection", use_container_width=True)

            flag_cols = st.columns([2, 2, 2, 2])
            with flag_cols[3]:
                key = f"flag_{idx}_{img_name}"
                initial = (img_name in user_flag_set)
                new_flagged = st.checkbox(" Flag Image ", key=key, value=initial)
                if "pending_flag_updates" not in st.session_state:
                    st.session_state.pending_flag_updates = {}
                if new_flagged != initial:
                    st.session_state.pending_flag_updates[img_name] = new_flagged

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1: prev_and_save = st.form_submit_button("⬅ Prev")
        with c2: save_only = st.form_submit_button("✅ Save Flags")
        with c3: next_and_save = st.form_submit_button("Next ➡")

        if prev_and_save or save_only or next_and_save:
            if flush_pending_flags_to_gcs(user):
                st.success("Flags saved.")
            if prev_and_save:
                if st.session_state.page_number > 1:
                    st.session_state.page_number -= 1
                    save_user_page_data(user, st.session_state.page_number)
                st.rerun()
            if next_and_save:
                if st.session_state.page_number < total_pages:
                    st.session_state.page_number += 1
                    save_user_page_data(user, st.session_state.page_number)
                st.rerun()
            if save_only:
                st.rerun()

# --- START ---
if "user" not in st.session_state:
    login()
else:
    main()
