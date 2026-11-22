
import streamlit as st
import sqlite3
import uuid
import time
from pathlib import Path
from io import BytesIO

import pandas as pd
import psutil
from PIL import Image
from ultralytics import YOLO

from src.auth import DB_PATH, create_users_table, login_user, register_user
from src.complaints import (
    create_complaints_table,
    save_complaint,
    get_all_complaints,
    get_user_complaints,
    update_complaint_status
)

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "runs" / "road_damage" /
    "yolov8n_cpu" / "weights" / "best.pt"
)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="RoadCare AI | India",
    page_icon="🛣️",
    layout="wide"
)


# ---------------- ROADCARE AI PREMIUM UI ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
  --rc-bg: #0b1020;
  --rc-panel: #131b2e;
  --rc-panel2: #18243a;
  --rc-blue: #3b82f6;
  --rc-cyan: #22d3ee;
  --rc-text: #edf4ff;
  --rc-muted: #95a4bd;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 12% 0%, #142341 0, #0b1020 42%, #080d18 100%); color: var(--rc-text); }
[data-testid="stHeader"] { background: rgba(8,13,24,.65); }
.block-container { max-width: 1380px; padding-top: 2rem; padding-bottom: 2.5rem; }
h1,h2,h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
h1 { font-size: 2.4rem !important; font-weight: 800 !important; }
h2 { font-size: 1.65rem !important; font-weight: 750 !important; }
h3 { font-size: 1.15rem !important; font-weight: 700 !important; }
p, label, .stMarkdown { color: var(--rc-text); }
small, [data-testid="stCaptionContainer"] { color: var(--rc-muted) !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#111b30 0%,#0b1222 100%); border-right: 1px solid rgba(148,163,184,.12); }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
[data-testid="stMetric"] { background: linear-gradient(145deg,#19253b,#111a2c); border: 1px solid rgba(148,163,184,.15); padding: 18px 20px; border-radius: 18px; box-shadow: 0 8px 26px rgba(0,0,0,.16); }
[data-testid="stMetricLabel"] p { color: #9db0ca !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: #f4f8ff !important; font-family: 'Manrope',sans-serif; }
div[data-testid="stTabs"] button { border-radius: 12px 12px 0 0; font-weight: 650; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #60a5fa !important; border-bottom-color: #3b82f6 !important; }
.stButton > button, .stFormSubmitButton > button, [data-testid="stDownloadButton"] button {
  border-radius: 12px !important; border: 1px solid rgba(96,165,250,.25) !important;
  font-weight: 700 !important; min-height: 2.7rem; transition: .2s ease;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(135deg,#2563eb,#0891b2) !important; color: white !important; border: 0 !important;
  box-shadow: 0 6px 20px rgba(37,99,235,.25);
}
.stButton > button:hover, .stFormSubmitButton > button:hover { transform: translateY(-1px); border-color: #60a5fa !important; }
[data-testid="stFileUploader"] { background: rgba(20,31,51,.65); border: 1px dashed rgba(96,165,250,.5); border-radius: 18px; padding: 12px; }
[data-testid="stFileUploader"] section { background: transparent; }
[data-testid="stImage"] img { border-radius: 16px; border: 1px solid rgba(148,163,184,.15); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 14px; overflow: hidden; }
input, textarea, [data-baseweb="select"] > div { border-radius: 10px !important; }
div[data-testid="stForm"] { background: rgba(19,27,46,.75); padding: 1.2rem; border: 1px solid rgba(148,163,184,.12); border-radius: 18px; }
div[data-testid="stExpander"] { background: rgba(19,27,46,.65); border: 1px solid rgba(148,163,184,.12); border-radius: 14px; }
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: rgba(148,163,184,.15) !important; }
.rc-hero { background: linear-gradient(120deg,rgba(37,99,235,.22),rgba(8,145,178,.10)), #121c30; border: 1px solid rgba(96,165,250,.22); border-radius: 24px; padding: 26px 30px; margin: 0 0 24px 0; }
.rc-eyebrow { color: #67e8f9; text-transform: uppercase; font-size: .76rem; letter-spacing: .15em; font-weight: 800; }
.rc-title { font-family: Manrope,sans-serif; font-size: 2.15rem; font-weight: 800; line-height: 1.15; color: #f8fafc; margin: 8px 0; }
.rc-subtitle { color: #a8bad3; font-size: 1rem; }
.rc-pill { display: inline-block; background: rgba(59,130,246,.16); border: 1px solid rgba(96,165,250,.25); color: #93c5fd; padding: 5px 11px; border-radius: 999px; font-size: .78rem; font-weight: 700; }
.rc-section { color: #dce8fb; font-weight: 700; margin: 5px 0 12px; }
@media (max-width: 768px) { .block-container { padding-left: 1rem; padding-right: 1rem; } .rc-title { font-size: 1.6rem; } }
</style>
""", unsafe_allow_html=True)


create_users_table()
create_complaints_table()


# ---------------- LOAD AI MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# ---------------- SESSION ----------------

if "user" not in st.session_state:
    st.session_state.user = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None


def logout():
    st.session_state.user = None
    st.session_state.analysis = None
    st.rerun()


# ---------------- HEADER ----------------

st.markdown("""
<div class="rc-hero">
  <div class="rc-eyebrow">SMART CIVIC TECHNOLOGY • INDIA</div>
  <div class="rc-title">🛣️ RoadCare AI</div>
  <div class="rc-subtitle">AI-powered road damage detection, reporting & monitoring</div>
  <div style="margin-top:16px"><span class="rc-pill">YOLO AI Vision</span>&nbsp; <span class="rc-pill">Citizen Reporting</span>&nbsp; <span class="rc-pill">Road Intelligence</span></div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# LOGGED-IN USER
# ==================================================

if st.session_state.user:

    user = st.session_state.user

    st.sidebar.markdown("""
<div style="padding:8px 4px 18px">
  <div style="font-size:1.7rem;font-weight:800;color:#f8fafc;font-family:Manrope">🛣️ RoadCare</div>
  <div style="font-size:.8rem;color:#91a4c1;letter-spacing:.08em">AI ROAD INTELLIGENCE</div>
</div>
""", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Name:** {user['name']}")
    st.sidebar.write(f"**Role:** {user['role'].upper()}")

    if st.sidebar.button("Logout"):
        logout()

    # ==============================================
    # CITIZEN DASHBOARD
    # ==============================================

    if user["role"] == "citizen":

        st.markdown(f"""
        <div class="rc-hero">
          <div class="rc-eyebrow">CITIZEN PORTAL</div>
          <div class="rc-title" style="font-size:1.8rem">Welcome back, {user['name']} 👋</div>
          <div class="rc-subtitle">Help make our roads safer. Upload a road image, detect damage and submit a location-based report.</div>
        </div>
        """, unsafe_allow_html=True)

        report_tab, history_tab = st.tabs(
            ["📷 Report Road Damage", "📋 My Complaints"]
        )

        with report_tab:

            st.subheader("Submit a New Road Damage Report")

            uploaded_file = st.file_uploader(
                "Upload a photograph of the damaged road",
                type=["jpeg", "jpg", "png", "gif", "webp", "tiff", "tif", "bmp", "heic", "raw"],
                key="road_image"
            )

            if uploaded_file:

                image_bytes = uploaded_file.getvalue()

                if len(image_bytes) > 50 * 1024 * 1024:
                    st.error("Image size must be less than 50 MB.")
                    st.stop()

                image = Image.open(BytesIO(image_bytes)).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded Road Photograph",
                    use_container_width=True
                )

                if st.button("🔍 Detect Road Damage", type="primary"):

                    if not MODEL_PATH.exists():
                        st.error(f"Trained model not found: {MODEL_PATH}")
                        st.stop()

                    with st.spinner("AI is analyzing your road image..."):

                        model = load_model()
                        start_time = time.perf_counter()

                        results = model.predict(
                            source=image,
                            conf=0.15,
                            imgsz=640,
                            iou=0.50,
                            device="cpu",
                            verbose=False
                        )

                        elapsed = time.perf_counter() - start_time
                        result = results[0]

                        detections = []

                        for box in result.boxes:
                            class_id = int(box.cls.item())
                            confidence = float(box.conf.item())

                            detections.append({
                                "Damage Type": model.names[class_id],
                                "Confidence (%)": round(
                                    confidence * 100, 2
                                )
                            })

                        annotated = result.plot()[:, :, ::-1]
                        annotated_image = Image.fromarray(annotated)

                        st.session_state.analysis = {
                            "image_bytes": image_bytes,
                            "annotated_bytes": None,
                            "detections": detections,
                            "processing_time": elapsed,
                            "annotated_image": annotated_image
                        }

                        buffer = BytesIO()
                        annotated_image.save(buffer, format="JPEG")

                        st.session_state.analysis["annotated_bytes"] = (
                            buffer.getvalue()
                        )

                # Display previous analysis after rerun
                analysis = st.session_state.analysis

                if (
                    analysis is not None
                    and analysis["image_bytes"] == image_bytes
                ):

                    st.divider()
                    st.markdown("### ✨ AI Analysis Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            image,
                            caption="Original Image",
                            use_container_width=True
                        )

                    with col2:
                        st.image(
                            analysis["annotated_image"],
                            caption="AI Detection Image",
                            use_container_width=True
                        )

                    detections = analysis["detections"]

                    a, b, c = st.columns(3)

                    a.metric("Damage Detections", len(detections))

                    b.metric(
                        "Damage Types",
                        len(set(d["Damage Type"] for d in detections))
                    )

                    c.metric(
                        "Processing Time",
                        f"{analysis['processing_time']:.2f} sec"
                    )

                    if detections:
                        st.dataframe(
                            pd.DataFrame(detections),
                            use_container_width=True
                        )
                    else:
                        st.warning(
                            "No damage was detected at this confidence "
                            "threshold. You can still submit the image "
                            "for manual inspection."
                        )

                    st.divider()
                    st.markdown("### 📍 Report Location")

                    st.write(
                        "Enter the location where this photograph was taken. "
                        "Coordinates must be for the damaged road, not "
                        "your current location unless they are the same."
                    )

                    with st.form("submit_complaint_form"):

                        location_text = st.text_input(
                            "Road name / Area / Landmark",
                            placeholder="Example: Road name, nearby landmark, Barabanki"
                        )

                        col_lat, col_lon = st.columns(2)

                        with col_lat:
                            latitude = st.number_input(
                                "Latitude",
                                min_value=-90.0,
                                max_value=90.0,
                                value=26.9400,
                                format="%.6f"
                            )

                        with col_lon:
                            longitude = st.number_input(
                                "Longitude",
                                min_value=-180.0,
                                max_value=180.0,
                                value=81.1900,
                                format="%.6f"
                            )

                        confirm = st.checkbox(
                            "I confirm that this is the location of the damaged road."
                        )

                        submit = st.form_submit_button(
                            "📨 Submit Road Damage Complaint",
                            type="primary"
                        )

                    if submit:

                        if not location_text.strip():
                            st.error("Please enter the road location.")

                        elif not confirm:
                            st.error("Please confirm the road location.")

                        else:

                            complaint_id = (
                                "RCA-" + uuid.uuid4().hex[:10].upper()
                            )

                            original_path = (
                                UPLOAD_DIR / f"{complaint_id}_original.jpg"
                            )

                            annotated_path = (
                                UPLOAD_DIR / f"{complaint_id}_detected.jpg"
                            )

                            original_path.write_bytes(
                                analysis["image_bytes"]
                            )

                            annotated_path.write_bytes(
                                analysis["annotated_bytes"]
                            )

                            if detections:
                                summary = "; ".join(
                                    f"{item['Damage Type']} "
                                    f"({item['Confidence (%)']}%)"
                                    for item in detections
                                )
                            else:
                                summary = "No AI detections; manual inspection requested"

                            save_complaint(
                                complaint_id=complaint_id,
                                user_id=user["id"],
                                image_path=str(original_path),
                                annotated_path=str(annotated_path),
                                location_text=location_text.strip(),
                                latitude=latitude,
                                longitude=longitude,
                                damage_summary=summary
                            )

                            st.session_state.analysis = None

                            st.success(
                                f"Complaint submitted successfully! "
                                f"Your complaint ID is {complaint_id}"
                            )

                            st.info(
                                "Your complaint is saved in RoadCare AI "
                                "for administrator review. It has NOT yet "
                                "been sent to Nagar Palika."
                            )

            else:
                st.info("Upload a road photograph to begin.")

        # CITIZEN COMPLAINT HISTORY

        with history_tab:

            st.subheader("My Submitted Complaints")

            complaints = get_user_complaints(user["id"])

            if complaints:

                df = pd.DataFrame(
                    complaints,
                    columns=[
                        "Complaint ID",
                        "Location",
                        "AI Detection",
                        "Status",
                        "Submitted At"
                    ]
                )

                st.dataframe(df, use_container_width=True)

            else:
                st.info("You have not submitted any complaints yet.")

    # ==============================================
    # ADMIN DASHBOARD
    # ==============================================

    elif user["role"] == "admin":

        st.markdown(f"""
        <div class="rc-hero">
          <div class="rc-eyebrow">CONTROL CENTER</div>
          <div class="rc-title" style="font-size:1.8rem">Administrator Dashboard</div>
          <div class="rc-subtitle">Welcome, {user['name']} — review reports, monitor system health and manage road issues.</div>
        </div>
        """, unsafe_allow_html=True)

        complaints = get_all_complaints()

        with sqlite3.connect(DB_PATH) as conn:
            users = conn.execute(
                "SELECT id, name, email, role FROM users ORDER BY id DESC"
            ).fetchall()

        total_complaints = len(complaints)
        pending = sum(1 for c in complaints if c[10] == "Pending")
        progress = sum(1 for c in complaints if c[10] == "In Progress")
        resolved = sum(1 for c in complaints if c[10] == "Resolved")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Reports", total_complaints)
        c2.metric("Pending", pending)
        c3.metric("In Progress", progress)
        c4.metric("Resolved", resolved)

        report_tab, users_tab, system_tab = st.tabs(
            ["📋 All Complaints", "👥 User Management", "📊 System Monitoring"]
        )

        with report_tab:

            st.subheader("Citizen Road Damage Reports")

            if complaints:

                for complaint in complaints:

                    (
                        db_id, complaint_id, citizen_name, citizen_email,
                        image_path, annotated_path, location_text,
                        latitude, longitude, damage_summary,
                        status, created_at
                    ) = complaint

                    with st.expander(
                        f"{complaint_id} | {status} | {location_text}"
                    ):

                        st.write("**Citizen:**", citizen_name)
                        st.write("**Email:**", citizen_email)
                        st.write("**Location:**", location_text)
                        st.write(
                            f"**Coordinates:** {latitude}, {longitude}"
                        )
                        st.write("**AI Results:**", damage_summary)
                        st.write("**Submitted:**", created_at)

                        col1, col2 = st.columns(2)

                        if Path(image_path).exists():
                            col1.image(
                                image_path,
                                caption="Original Photograph",
                                use_container_width=True
                            )

                        if Path(annotated_path).exists():
                            col2.image(
                                annotated_path,
                                caption="AI Detection",
                                use_container_width=True
                            )

                        map_url = (
                            "https://www.google.com/maps/search/?api=1&query="
                            f"{latitude},{longitude}"
                        )

                        st.link_button(
                            "📍 Open Complaint Location in Google Maps",
                            map_url
                        )

                        with st.form(f"status_{complaint_id}"):

                            new_status = st.selectbox(
                                "Update Complaint Status",
                                ["Pending", "In Progress", "Resolved"],
                                index=[
                                    "Pending",
                                    "In Progress",
                                    "Resolved"
                                ].index(status)
                            )

                            update = st.form_submit_button(
                                "Update Status"
                            )

                        if update:

                            update_complaint_status(
                                complaint_id,
                                new_status
                            )

                            st.success("Complaint status updated.")
                            st.rerun()

            else:
                st.info(
                    "No complaints received yet. "
                    "Reports submitted by citizens will appear here."
                )

        with users_tab:

            st.subheader("Registered Accounts")

            if users:

                user_df = pd.DataFrame(
                    users,
                    columns=["User ID", "Name", "Email", "Role"]
                )

                st.dataframe(
                    user_df,
                    use_container_width=True
                )

            st.caption(
                "Official account provisioning will be added in the "
                "next development step."
            )

        with system_tab:

            st.subheader("System Resource Monitoring")

            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent

            col1, col2 = st.columns(2)

            col1.metric("CPU Usage", f"{cpu}%")
            col2.metric("RAM Usage", f"{ram}%")

            st.caption(
                "These are current resource readings, not historical metrics."
            )

    # ==============================================
    # OFFICIAL DASHBOARD
    # ==============================================

    elif user["role"] == "official":

        st.header("🏛️ Nagar Palika Official Dashboard")

        st.info(
            "The official complaint assignment dashboard will be "
            "connected after we configure authorized Nagar Palika accounts."
        )


# ==================================================
# LOGIN AND REGISTRATION
# ==================================================

else:

    login_tab, register_tab = st.tabs(
        ["🔐 Login", "📝 Citizen Registration"]
    )

    with login_tab:

        st.markdown("### Welcome to RoadCare AI")
        st.caption("Sign in to access road intelligence and reporting tools.")

        with st.form("login_form"):

            email = st.text_input("Email address")

            password = st.text_input(
                "Password",
                type="password"
            )

            login_button = st.form_submit_button(
                "Login",
                use_container_width=True
            )

        if login_button:

            if not email.strip() or not password:
                st.error("Enter your email and password.")

            else:

                user = login_user(email, password)

                if user:
                    st.session_state.user = user
                    st.session_state.analysis = None
                    st.rerun()

                else:
                    st.error("Invalid email or password.")

    with register_tab:

        st.markdown("### Create your citizen account")
        st.caption("Register to report road damage and track your submissions.")

        with st.form("register_form"):

            name = st.text_input("Full name")
            email = st.text_input("Email address")

            password = st.text_input(
                "Create password",
                type="password"
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password"
            )

            register_button = st.form_submit_button(
                "Register",
                use_container_width=True
            )

        if register_button:

            if password != confirm_password:
                st.error("Passwords do not match.")

            else:

                success, message = register_user(
                    name,
                    email,
                    password,
                    role="citizen"
                )

                if success:
                    st.success(message)
                    st.info("You can now log in.")

                else:
                    st.error(message)


# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div style="text-align:center;padding:18px 0;color:#8193ad;font-size:.82rem">
  <strong style="color:#a9bdd8">ROADCARE AI</strong> &nbsp;•&nbsp; India, Uttar Pradesh<br>
  AI Road Damage Reporting & Monitoring Platform
</div>
""", unsafe_allow_html=True)
