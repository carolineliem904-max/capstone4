# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO

from ppe_logic import evaluate_compliance

st.set_page_config(page_title="Construction Safety PPE Detector", layout="wide")
st.title("🦺⛑️ Construction Safety - PPE Compliance Detector")

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")  # taruh best.pt di folder models/

model = load_model()

st.sidebar.header("Settings")
conf_min = st.sidebar.slider("Confidence threshold", 0.05, 0.90, 0.35, 0.05)
iou_nms  = st.sidebar.slider("NMS IoU", 0.10, 0.90, 0.60, 0.05)
imgsz_inf = st.sidebar.selectbox("Inference image size", [640, 768, 896], index=1)  # default 768
min_person_area = st.sidebar.slider("Min person area (px^2)", 0, 20000, 2500, 250)


tab1, tab2 = st.tabs(["📤 Upload Images", "🔗 Image URL"])

def run_one_image(img: Image.Image, caption: str):
    res = model.predict(img, conf=conf_min, iou=iou_nms, imgsz=imgsz_inf, verbose=False)[0]
    annotated = Image.fromarray(res.plot()[:, :, ::-1])  # BGR->RGB

    summary, df = evaluate_compliance(res, conf_min=conf_min, min_person_area=min_person_area)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.image(annotated, caption=caption, use_container_width=True)

    with right:
        st.subheader("Summary")
        st.write(summary)
        st.subheader("Per Person Result")
        st.dataframe(df, use_container_width=True)

    st.divider()

with tab1:
    files = st.file_uploader(
        "Upload one or multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if files:
        for f in files:
            img = Image.open(f).convert("RGB")
            run_one_image(img, caption=f"Result: {f.name}")

with tab2:
    url = st.text_input("Paste image URL (direct image link)")
    if st.button("Analyze URL") and url:
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            run_one_image(img, caption=f"Result: {url}")
        except Exception as e:
            st.error(f"Failed to load image from URL. Error: {e}")

