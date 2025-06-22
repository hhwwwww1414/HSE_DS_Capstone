# src/utils/gradcam.py
"""
Grad-CAM для обученной MobileNetV3-small.

Запуск из терминала
-------------------
    python -m src.utils.gradcam <jpg-путь>   # явная картинка
    python -m src.utils.gradcam              # случайная из датасета

Использование в Streamlit
-------------------------
    from src.utils.gradcam import run_gradcam
    fig = run_gradcam(Path("data/interim/products_224/Apparel/00002.jpg"))
    st.pyplot(fig)
"""
from __future__ import annotations

# ── stdlib
from pathlib import Path
import random, sys, logging

# ── third-party
import torch
import torchvision.models as models
import torchvision.transforms as T
from torchcam.methods import SmoothGradCAMpp
from PIL import Image
import matplotlib.pyplot as plt


# ──────────────────────────── CONFIG ──────────────────────────────────
logging.getLogger().setLevel(logging.ERROR)          # глушим root-warnings

CKPT = Path("data/models/mobilenet_v3_small.ckpt")   # Lightning-чекпойнт
ROOT = Path("data/interim/products_224")             # подпапки = классы

_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ──────────────────── helpers ─────────────────────────────────────────
def _load_model() -> tuple[torch.nn.Module, list[str]]:
    """Создаём MobileNetV3-small под n классов и вливаем веса ckpt."""
    classes = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"Папки-классы не найдены в {ROOT}")

    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = torch.nn.Linear(1024, len(classes))

    ckpt = torch.load(CKPT, map_location="cpu")           # Lightning ckpt
    state = {k.replace("net.", ""): v
             for k, v in ckpt["state_dict"].items()
             if k.startswith("net.")}
    net.load_state_dict(state, strict=False)
    net.eval()
    return net, classes


def _pick_image() -> Path:
    """CLI: берём argv[1] или рандомный .jpg из датасета."""
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        pool = list(ROOT.rglob("*.jpg"))
        img_path = random.choice(pool)
        print(f"[i] Using random image: {img_path.relative_to(Path.cwd())}")
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    return img_path


# ──────────────────── public API ──────────────────────────────────────
def run_gradcam(img_path: Path) -> plt.Figure:
    """
    Строит Figure с тепловой картой Grad-CAM поверх исходного изображения.
    plt.show() не вызывается — это делает внешний код (Streamlit или CLI).
    """
    net, classes = _load_model()
    cam_extractor = SmoothGradCAMpp(net, target_layer="features")  # явный слой

    img = Image.open(img_path).convert("RGB")
    inp = _TF(img).unsqueeze(0)           # (1, 3, 224, 224)

    net.zero_grad()
    out = net(inp)
    idx = out.argmax(1).item()

    cam = cam_extractor(idx, out)[0][0].cpu()   # (H, W)

    # ── визуализация ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.imshow(cam, alpha=0.35, cmap="jet")
    ax.set_title(f"Predicted: {classes[idx]}", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ──────────────────── CLI entry-point ────────────────────────────────
def main() -> None:
    fig = run_gradcam(_pick_image())
    plt.show()


if __name__ == "__main__":
    main()

__all__ = ["run_gradcam"]
