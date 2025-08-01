import streamlit as st
import pandas as pd
import os
import json
from io import BytesIO
from google.cloud import storage
from PIL import Image
import requests
import numpy as np

# Set wide layout
st.set_page_config(layout="wide")

# --- CONFIG ---
GCS_BUCKET = "veytel-cloud-store"
MASTER_CSV_PATH = "pulsarai_masters/master_250723.csv"
SERVICE_ACCOUNT_JSON = st.secrets["gcs_service_account"]
FLAG_LOG_FILE = "flag_log.json"
PAGE_LOG_FILE = "page_log.json"
VALID_USERS = sorted(["Ellen", "Cathy", "Robin", "Anrey", "Song", "Kevin", "Swathi", "Nitya", "Manasvi",
                      "Rachel", "Mike", "Paul", "Test_1", "Test_2", "Test_3"])
PASSWORD = "Veytel2025"

# --- GCS CLIENT ---
def get_gcs_client():
    return storage.Client.from_service_account_info(SERVICE_ACCOUNT_JSON)

def gcs_get_blob_url(bucket_name, blob_path):
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.generate_signed_url(version="v4", expiration=3600)

def get_image(url, size=(256, 256), mode="RGB"):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert(mode).resize(size)

def overlay_mask_on_image(base_img, mask_array, color=(0, 255, 0), alpha=128):
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    pixels = overlay.load()
    for y in range(base_img.size[1]):
        for x in range(base_img.size[0]):
            if mask_array[y, x] > 0:
                pixels[x, y] = color + (alpha,)
    return Image.alpha_composite(base_img.convert("RGBA"), overlay).convert("RGB")

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

def load_flag_log():
    if os.path.exists(FLAG_LOG_FILE):
        with open(FLAG_LOG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_flag_log(flag_log):
    with open(FLAG_LOG_FILE, "w") as f:
        json.dump(flag_log, f, indent=2)

# --- MAIN ---
def main():
    user = st.session_state.user

    if os.path.exists(PAGE_LOG_FILE):
        with open(PAGE_LOG_FILE, "r") as f:
            page_log = json.load(f)
    else:
        page_log = {}

    default_settings = page_log.get(user, {"page_number": 1, "page_size": 50})
    master_df = load_master()
    flag_log = load_flag_log()
    master_df["flagged_by"] = master_df["imgName"].apply(lambda x: ", ".join(flag_log.get(x, [])))

    with st.sidebar:
        st.markdown("\U0001FA7B PulsarAI Segmentation Flagging Tool")
        st.markdown(f"**👤 Logged in as:** `{user}`")
        st.markdown("### View Options")

        view_mode = st.selectbox("View Mode", ["All Images", "Flagged Images", "Flagged by Selected Users"])
        selected_users = st.multiselect("Select Users", VALID_USERS) if view_mode == "Flagged by Selected Users" else []

        if view_mode == "Flagged Images":
            view_df = master_df[master_df["flagged_by"] != ""]
        elif view_mode == "Flagged by Selected Users":
            view_df = master_df[master_df["flagged_by"].apply(lambda x: any(user in x for user in selected_users))]
        else:
            view_df = master_df

        total_pages = max(1, (len(view_df) - 1) // 50 + 1)
        if "page_number" not in st.session_state:
            st.session_state.page_number = default_settings["page_number"]
        if "page_size" not in st.session_state:
            st.session_state.page_size = default_settings["page_size"]

        # Clamp page number within valid range
        st.session_state.page_number = max(1, min(st.session_state.page_number, total_pages))

        # Now render the input safely
        st.session_state.page_number = st.number_input(
            "Page", min_value=1, max_value=total_pages,
            value=st.session_state.page_number, step=1, key="page_input"
        )

        st.session_state.page_size = st.selectbox("Images per page", [10, 50, 100, 500],
                                                  index=[10, 50, 100, 500].index(default_settings["page_size"]))

        st.markdown(f"**📄 Viewing page {st.session_state.page_number} / {total_pages}**")

        # Save to log
        page_log[user] = {"page_number": st.session_state.page_number, "page_size": st.session_state.page_size}
        with open(PAGE_LOG_FILE, "w") as f:
            json.dump(page_log, f, indent=2)

        st.markdown("### ⬇ Export Options")
        export_option = st.selectbox(
            "Select export option",
            ["Select...", "⬇ Full Master file with Flagged column", "⬇ Master file without Flagged Images"]
        )

        export_df_full = master_df
        export_df_unflagged = master_df[master_df["flagged_by"] == ""].drop(columns=["flagged_by"])

        if export_option == "⬇ Full Master file with Flagged column":
            export_data = export_df_full.to_csv(index=False).encode()
            export_filename = "master_with_flagged.csv"
        elif export_option == "⬇ Master file without Flagged Images":
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

    # --- Pagination ---
    start_idx = (st.session_state.page_number - 1) * st.session_state.page_size
    end_idx = start_idx + st.session_state.page_size
    page_df = view_df.iloc[start_idx:end_idx]

    for idx, row in page_df.iterrows():
        with st.container():
            cols = st.columns([1, 1, 1, 1])
            norm_img = get_image(gcs_get_blob_url(GCS_BUCKET, row["normalizedPath"]))
            old_mask = np.array(get_image(gcs_get_blob_url(GCS_BUCKET, row["maskPath_old"]), mode="L"))
            new_mask = np.array(get_image(gcs_get_blob_url(GCS_BUCKET, row["maskPath"]), mode="L"))

            with cols[0]:
                st.image(norm_img, caption="Normalized", width=256)
            with cols[1]:
                st.image(overlay_mask_on_image(norm_img, old_mask, color=(0, 255, 0)), caption="Old Mask Overlay", width=256)
            with cols[2]:
                st.image(overlay_mask_on_image(norm_img, new_mask, color=(255, 0, 0)), caption="New Mask Overlay", width=256)
            with cols[3]:
                both_overlay = overlay_mask_on_image(overlay_mask_on_image(norm_img, old_mask, color=(0, 255, 0), alpha=128),
                                                     new_mask, color=(255, 0, 0), alpha=128)
                st.image(both_overlay, caption="Intersection", width=256)

        flag_cols = st.columns(4)
        with flag_cols[0]:
            with st.expander(" More details"):
                st.markdown(f"- **Image Name**: `{row.get('imgName', 'N/A')}`")
                st.markdown(f"- **Image Quality**: `{row.get('imgQuality', 'N/A')}`")
                st.markdown(f"- **Left Atelectasis**: `{row.get('left_atelectasis', 'N/A')}`")
                st.markdown(f"- **Right Atelectasis**: `{row.get('right_atelectasis', 'N/A')}`")
                st.markdown(f"- **RALE Score**: `{row.get('RALE', 'N/A')}`")
                st.markdown(f"- **mRALE**: `{row.get('mRALE', 'N/A')}`")
        with flag_cols[3]:
            key = f"flag_{idx}_{row['imgName']}"
            initial = user in flag_log.get(row["imgName"], [])
            new_flagged = st.checkbox(" Flag Image ", key=key, value=initial)
            if new_flagged != initial:
                if new_flagged:
                    flag_log.setdefault(row["imgName"], []).append(user)
                else:
                    flag_log[row["imgName"]].remove(user)
                    if not flag_log[row["imgName"]]:
                        del flag_log[row["imgName"]]
                save_flag_log(flag_log)

# --- START ---
if "user" not in st.session_state:
    login()
else:
    main()
