# app.py
# Streamlit: Webcam/URL Image Processing Playground (Modern UI)
# -----------------------------------------------------------
# Features
# - Input sources: Webcam (via st.camera_input), Upload, or Internet URL
# - Simple image-processing pipeline with tunable parameters (sidebar)
# - Live preview: Original vs Processed
# - One chart from image features (intensity/color histogram)
# - Download processed image
# - Modern layout with a clean theme & subtle CSS polish

import io
from io import BytesIO
import base64
import numpy as np
import requests
from PIL import Image
import cv2
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# Page & basic theme
# -------------------------------
st.set_page_config(
    page_title="VisionLab: Webcam/URL Image Processing",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle custom CSS for modern look
st.markdown(
    """
    <style>
        .app-title {
            font-size: 2rem;
            font-weight: 800;
            padding: 0.5rem 0;
            background: linear-gradient(90deg, #6EE7B7, #3B82F6, #A78BFA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .soft-card {
            border-radius: 1rem;
            padding: 1rem;
            background: rgba(255,255,255,0.6);
            box-shadow: 0 10px 30px rgba(0,0,0,0.06);
            border: 1px solid rgba(0,0,0,0.05);
        }
        .metric-pill {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            border: 1px solid rgba(0,0,0,0.1);
            background: rgba(0,0,0,0.03);
            font-size: 0.85rem;
            margin-right: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='app-title'>VisionLab ▸ Webcam/URL Image Processing</div>", unsafe_allow_html=True)
st.caption("ลองแหล่งภาพหลายแบบ + ปรับพารามิเตอร์ประมวลผลภาพได้แบบเรียลไทม์ พร้อมกราฟคุณสมบัติของภาพ")

# -------------------------------
# Helpers
# -------------------------------

def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img_cv: np.ndarray) -> Image.Image:
    if len(img_cv.shape) == 2:
        return Image.fromarray(img_cv)
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def fetch_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# -------------------------------
# Sidebar: Input source & controls
# -------------------------------
with st.sidebar:
    st.header("แหล่งข้อมูลภาพ")
    source = st.radio(
        "เลือกแหล่งภาพ",
        ["Webcam", "Upload", "Image URL"],
        index=0,
        help="ใช้กล้องเว็บแคม (ผ่านเบราว์เซอร์), อัปโหลดไฟล์, หรือดึงจากลิงก์อินเทอร์เน็ต",
    )

    img_pil = None

    if source == "Webcam":
        cam = st.camera_input("ถ่ายภาพจากเว็บแคม", help="คลิกเพื่อถ่ายภาพจากกล้องของคุณ")
        if cam is not None:
            img_pil = Image.open(cam).convert("RGB")

    elif source == "Upload":
        up = st.file_uploader(
            "อัปโหลดรูปภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False
        )
        if up is not None:
            img_pil = Image.open(up).convert("RGB")

    else:  # Image URL
        st.caption("ตัวอย่าง URL ภาพ: https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d")
        url = st.text_input("วางลิงก์รูปภาพจากอินเทอร์เน็ต")
        if url:
            try:
                img_pil = fetch_image_from_url(url)
            except Exception as e:
                st.error(f"ไม่สามารถดึงรูปจาก URL นี้ได้: {e}")

    st.divider()

    st.header("การประมวลผลภาพ")
    use_grayscale = st.checkbox("Grayscale", value=False)

    blur_on = st.checkbox("Gaussian Blur", value=False)
    blur_ksize = st.slider("Blur kernel (odd)", min_value=1, max_value=31, value=5, step=2)

    canny_on = st.checkbox("Canny Edge", value=False)
    canny_t1 = st.slider("Canny threshold1", 0, 255, 100)
    canny_t2 = st.slider("Canny threshold2", 0, 255, 200)

    bright_on = st.checkbox("Brightness/Contrast", value=False)
    contrast = st.slider("Contrast (α)", 0.1, 3.0, 1.0)
    brightness = st.slider("Brightness (β)", -100, 100, 0)

    resize_on = st.checkbox("Resize (scale %)", value=False)
    scale_percent = st.slider("Scale %", 10, 200, 100, step=5)

    rotate_on = st.checkbox("Rotate", value=False)
    angle = st.slider("Angle (°)", -180, 180, 0)

# -------------------------------
# Processing pipeline
# -------------------------------
proc_cv = None
orig_cv = None

if img_pil is not None:
    orig_cv = pil_to_cv2(img_pil)
    img = orig_cv.copy()

    # Grayscale first (optional)
    if use_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    if blur_on:
        k = max(1, blur_ksize)
        if k % 2 == 0:
            k += 1
        if len(img.shape) == 2:
            img = cv2.GaussianBlur(img, (k, k), 0)
        else:
            img = cv2.GaussianBlur(img, (k, k), 0)

    # Canny Edge
    if canny_on:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(gray, canny_t1, canny_t2)

    # Brightness / Contrast
    if bright_on:
        # alpha = contrast, beta = brightness
        if len(img.shape) == 2:
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        else:
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    # Resize
    if resize_on and scale_percent != 100:
        h, w = img.shape[:2]
        new_w = max(1, int(w * scale_percent / 100.0))
        new_h = max(1, int(h * scale_percent / 100.0))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale_percent < 100 else cv2.INTER_LINEAR)

    # Rotate (around center)
    if rotate_on and angle != 0:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        border_mode = cv2.BORDER_REFLECT if len(img.shape) == 2 else cv2.BORDER_REFLECT
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)

    proc_cv = ensure_uint8(img)

# -------------------------------
# Layout & display
# -------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("ภาพต้นฉบับ")
    if orig_cv is not None:
        st.image(cv2_to_pil(orig_cv), caption="Original", use_column_width=True)
    else:
        st.info("⬅ เลือกแหล่งภาพจากแถบด้านซ้าย (Webcam / Upload / URL)")

with col2:
    st.subheader("ผลลัพธ์หลังประมวลผล")
    if proc_cv is not None:
        st.image(cv2_to_pil(proc_cv), caption="Processed", use_column_width=True)

        # Download button
        out_pil = cv2_to_pil(proc_cv)
        buf = BytesIO()
        out_pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.download_button(
            label="ดาวน์โหลดภาพผลลัพธ์ (PNG)",
            data=buf.getvalue(),
            file_name="processed.png",
            mime="image/png",
        )
    else:
        st.warning("ยังไม่มีภาพผลลัพธ์ — โปรดใส่รูปและเลือกพารามิเตอร์การประมวลผล")

st.divider()

# -------------------------------
# Chart from image features (one chart)
# -------------------------------
st.subheader("กราฟคุณสมบัติของภาพ (Histogram)")

if proc_cv is not None:
    fig = plt.figure()
    if len(proc_cv.shape) == 2:
        # Grayscale histogram
        plt.title("Histogram (Grayscale)")
        plt.xlabel("Pixel value")
        plt.ylabel("Frequency")
        hist = cv2.calcHist([proc_cv], [0], None, [256], [0,256])
        plt.plot(hist)
        plt.xlim([0, 256])
    else:
        # RGB histogram
        plt.title("Histogram (RGB)")
        plt.xlabel("Pixel value")
        plt.ylabel("Frequency")
        for ch in range(3):
            hist = cv2.calcHist([proc_cv], [ch], None, [256], [0,256])
            plt.plot(hist)
        plt.xlim([0, 256])

    st.pyplot(fig, use_container_width=True)

    # Simple metrics
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    mean_val = float(np.mean(proc_cv))
    std_val = float(np.std(proc_cv))
    if len(proc_cv.shape) == 2:
        edge_ratio = float(np.mean(proc_cv > 0)) if canny_on else 0.0
    else:
        gray_for_edges = cv2.cvtColor(proc_cv, cv2.COLOR_BGR2GRAY)
        canny_auto = cv2.Canny(gray_for_edges, 100, 200)
        edge_ratio = float(np.mean(canny_auto > 0))

    st.markdown(
        f"<span class='metric-pill'>Mean intensity: {mean_val:.2f}</span>"
        f"<span class='metric-pill'>Std: {std_val:.2f}</span>"
        f"<span class='metric-pill'>Edge density: {edge_ratio*100:.1f}%</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.caption("กราฟจะแสดงเมื่อมีภาพผลลัพธ์")

# -------------------------------
# Footer / Tips
# -------------------------------
with st.expander("เคล็ดลับการใช้งาน"):
    st.markdown(
        """
        - ใช้ **Webcam** เพื่อถ่ายภาพทันที (ทำงานผ่านเบราว์เซอร์ของ Streamlit)
        - ถ้าใช้ **Image URL**: ลองภาพจาก Unsplash หรือเว็บไซต์ที่เปิดให้ดึงรูปได้
        - ปรับ **Gaussian Blur**, **Canny**, และ **Brightness/Contrast** เพื่อดูผลที่แตกต่าง
        - กราฟ Histogram ด้านล่างแสดงการกระจายของค่า Pixel ของภาพผลลัพธ์
        - ปรับ **Rotate** และ **Resize** เพื่อเตรียมภาพก่อนนำไปใช้งานอื่น
        """
    )

