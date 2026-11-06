# Crop Prediction Project

This repository contains the collaborative 2021 B.Sc. crop suitability thesis and a later, separate 2026 evaluation reassessment. The dataset has seven soil and environmental features and 22 crop labels; it has no crop-yield measurements.

## Legacy Streamlit app

The app in `Predict_Crop.py` uses the historical `model.pkl`, serialized with scikit-learn 0.24.1. Use Python 3.9 with the pinned legacy dependencies; newer scikit-learn versions cannot load this model.

From the repository root in PowerShell:

```powershell
uv python install 3.9
uv venv --python 3.9 .venv-legacy
uv pip install --python .venv-legacy\Scripts\python.exe -r requirements.txt
& .\.venv-legacy\Scripts\python.exe -m streamlit run Predict_Crop.py
```

Open `http://localhost:8501` (or the URL printed by Streamlit). Stop it with `Ctrl+C`.

## 2026 reassessment

The frozen Random Forest baseline, its reproducibility workflow, commands, saved results, and limitations are documented in [EXTENSION.md](EXTENSION.md). It has its own pinned environment in `requirements-extension.txt` and does not replace the historical app model.

## Original thesis team

The 2021 B.Sc. thesis was a four-member collaboration. The public repository was prepared and maintained by Md. Mehedi Hasan.

- Md. Mehedi Hasan — repository maintenance, dataset review, implementation, and coding.
- Md. Zahid Hasan — coding support, dataset review, prediction-result testing, thesis documentation, and presentation preparation.
- Md. Ismail Hossen — thesis documentation and presentation preparation.
- Md. Ashraful Islam — thesis documentation and presentation preparation.
