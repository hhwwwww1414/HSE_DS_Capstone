# ─── Makefile ─────────────────────────────────────────────────────────
# Все цели «фантомные», чтобы make всегда их выполнял при вызове
.PHONY: train_churn train_dl demo clean

## Обучить классические ML-модели (churn)
train_churn:
	python -m src.models.train_churn

## Обучить DL-модель MobileNetV3 (product images)
train_dl:
	python -m src.models.train_products

## Запустить Streamlit-демо c Grad-CAM
demo:
	python -m streamlit run notebooks/streamlit_demo.py

## Очистить временные файлы / логи (при желании расширите список)
clean:
	rm -rf lightning_logs .pytest_cache __pycache__
# ──────────────────────────────────────────────────────────────────────
