# src/utils/gradcam.py
"""
Grad-CAM для обученного MobileNetV3-small.

Запуск:

    # явный путь к файлу
    python -m src.utils.gradcam data/interim/products_224/Apparel/00002.jpg

    # без аргументов — возьмётся случайная картинка из датасета
    python -m src.utils.gradcam
"""
from pathlib import Path
import sys, random
import torch
import torchvision.models as models
import torchvision.transforms as T
from torchcam.methods import SmoothGradCAMpp
from PIL import Image
import matplotlib.pyplot as plt


CKPT = Path("data/models/mobilenet_v3_small.ckpt")      # Lightning-чекпойнт
ROOT = Path("data/interim/products_224")                # папка с классами


# ──────────────────────────────────────────────────────────────────────
def load_model() -> tuple[torch.nn.Module, list[str]]:
    """Создаём MobileNetV3-small под n классов и вливаем веса из Lightning ckpt."""
    classes = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"Нет подпапок-классов внутри {ROOT}")

    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = torch.nn.Linear(1024, len(classes))

    ckpt = torch.load(CKPT, map_location="cpu")
    state = {k.replace("net.", ""): v
             for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
    net.load_state_dict(state, strict=False)
    net.eval()
    return net, classes


# трансформация → tensor 3×224×224
tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ──────────────────────────────────────────────────────────────────────
def pick_image() -> Path:
    """CLI-аргумент или случайный .jpg из датасета."""
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        pool = list(ROOT.rglob("*.jpg"))
        if not pool:
            raise RuntimeError(f"*.jpg не найдено в {ROOT}")
        img_path = random.choice(pool)
        print(f"[i] Using random image: {img_path}")
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    return img_path


# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    net, classes = load_model()
    cam_extractor = SmoothGradCAMpp(net)          # target_layer берётся дефолтно
    img_path = pick_image()

    img = Image.open(img_path).convert("RGB")
    inp = tf(img).unsqueeze(0)                    # (1, 3, 224, 224)

    net.zero_grad() 
    out = net(inp)

    pred_idx = out.argmax(1).item()
    cam = cam_extractor(pred_idx, out)[0][0].cpu()   # (H, W)

    # ── отрисовка ─────────────────────────────────────────────────────
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.imshow(cam, alpha=0.5, cmap="jet")
    plt.title(f"Predicted: {classes[pred_idx]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
