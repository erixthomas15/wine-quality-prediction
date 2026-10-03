# VinoSense — Wine Quality Predictor

A Flask website for the Wine Quality Random Forest data-science project.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start:

```bash
python app.py
```

5. Open `http://127.0.0.1:5000`

## Files

- `app.py` — Flask backend and prediction pipeline
- `templates/index.html` — interactive website
- `static/style.css` — styling
- `wine_quality_random_forest.pkl` — supplied Random Forest model
- `WineQT_batch_id_dataset.csv` — supplied original dataset

## Important

The model was saved with scikit-learn 1.6.1. If you see a version warning, install the same scikit-learn version used to create the model when possible:

```bash
pip install scikit-learn==1.6.1
```

The supplied preprocessed dataset confirms IQR outlier capping followed by standardization for the 11 original numerical features. The app applies the same transformation and creates the four engineered features expected by the saved model.
