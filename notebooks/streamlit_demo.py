# notebooks/streamlit_demo.py
import streamlit as st
from pathlib import Path
import random
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # ← корень проекта
sys.path.append(str(ROOT))
ROOT = Path("data/interim/products_224")    # папка с картинками
from src.utils.gradcam import run_gradcam
st.set_page_config(page_title="Fashion Grad-CAM demo", page_icon="🩳")
st.title("👗 Grad-CAM для MobileNetV3")

# ── выбираем изображение ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 Случайное изображение из датасета"):
        pool = list(ROOT.rglob("*.jpg"))
        if pool:
            st.session_state["img_path"] = random.choice(pool)
        else:
            st.error("В каталоге нет картинок 😥")

with col2:
    up = st.file_uploader("или загрузите свой .jpg", type=["jpg", "jpeg"])
    if up:
        tmp = Path("tmp_upload.jpg")
        tmp.write_bytes(up.read())
        st.session_state["img_path"] = tmp

# ── показываем выбранное изображение ─────────────────────────────────
img_path = st.session_state.get("img_path")

if img_path:
    st.image(str(img_path), caption=Path(img_path).name, use_container_width=True)
else:
    st.warning("Сначала выберите или загрузите изображение")

# ── Grad-CAM ─────────────────────────────────────────────────────────
if st.button("🔍 Показать Grad-CAM", disabled=img_path is None):
    st.info("Считаем Grad-CAM, подождите…")
    fig = run_gradcam(Path(img_path))
    st.pyplot(fig)
