# 🎬 Movie Review Sentiment Analysis

A beginner-friendly NLP project that classifies movie reviews as **positive**
or **negative** using a TF-IDF vectorizer and a Logistic Regression model,
served through a Streamlit web app.

# Live Demo
Live Link: https://sentimentanalysis-vdrms4acrgrp4vvsiiqnml.streamlit.app/
## Project structure

```
sentiment_project/
├── app.py               # Streamlit app
├── train_model.py       # Trains and saves the TF-IDF + LogReg pipeline
├── data/
│   └── reviews.csv       # ~230 labeled movie review sentences
├── models/               # Created automatically after training (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Quickstart (local)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (optional — app.py will auto-train on first run too)
python train_model.py

# 5. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: sentiment analysis app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select your repo/branch, and set the main file to `app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically and
   `app.py` will train the model on first launch since `models/` isn't
   committed to git.

## Code review — notebook vs. this repo

I reviewed `sentiment_analysis.ipynb` while turning it into this repo. Here's
what I found and how it was addressed:

### Bugs
- **Save step crashes as-is:** `joblib.dump(vectrizer, ...)` referenced an
  undefined variable (`vectrizer` vs. `vectorizer`) — this cell would raise
  a `NameError`. Fixed by wrapping the vectorizer and model in a single
  `sklearn.pipeline.Pipeline` and saving that one object (see `train_model.py`).
- **`display(sentiment_df.head())`** only works inside a Jupyter kernel; it
  fails in a plain `.py` script. Replaced with `st.dataframe(...)` in the app.
- **`while True: input(...)`** loop for interactive testing works in a
  notebook/terminal but isn't usable in a deployed app — replaced with the
  Streamlit text box in `app.py`.

### Design/structure issues worth fixing
- **Unused heavy imports:** the first cell imports `tensorflow` and `cv2`
  (OpenCV), neither of which is used anywhere in this notebook. They likely
  belong to a different assignment/project template. Leaving them in
  massively bloats `requirements.txt` and install time/size, and can break
  `pip install` on constrained deploy environments (e.g. Streamlit Cloud's
  free tier) for zero benefit. They were dropped entirely here.
- **Vectorizer and model saved separately** — easy to accidentally load a
  vectorizer that doesn't match the model it was trained with. A single
  `Pipeline` object removes that failure mode and is the standard
  scikit-learn pattern for shipping a text classifier.
- **No reusable training function** — the notebook trains inline. Wrapping
  it in `train_and_save()` makes it testable and re-runnable, and lets the
  Streamlit app call it directly if no saved model exists yet.

### Modeling / data concerns (good to know, not "bugs")
- **The dataset is small and synthetic** (~230 sentences generated from a
  handful of templates like *"The X was Y and kept me hooked"*). A
  train/test split on this data will report very high accuracy, but that's
  partly because the model can key in on individual adjectives (e.g.
  "brilliant", "dreadful") rather than genuine context or negation
  handling. It will **not generalize well** to real-world reviews with
  sarcasm, mixed sentiment, or negation ("not bad at all").
- **No cross-validation** — a single fixed `train_test_split` is fine for a
  learning demo but a real project should use k-fold CV to get a more
  reliable accuracy estimate.
- **No handling of negation/context** — TF-IDF + Logistic Regression is a
  solid, fast baseline, but it treats words independently (a bag-of-words
  model), so phrases like "not great" can still register positive words.
  For better real-world performance, consider n-grams
  (`TfidfVectorizer(ngram_range=(1,2))`), or a small transformer model
  (e.g. `distilbert-base-uncased-finetuned-sst-2-english` via
  🤗 `transformers`) fine-tuned or used directly.

### Overall verdict
The core idea (TF-IDF → Logistic Regression → sentiment) is a completely
reasonable and appropriately simple baseline for a first ML/NLP project, and
the notebook's structure (import → EDA → split → vectorize → train →
evaluate → inference function) follows good practice. The main things
holding it back from "production-ready" were the save-step bug, the
unrelated heavy dependencies, and the toy dataset — all addressed or called
out above. If you want to extend this, the highest-value next step is
training on a real dataset (e.g. the
[IMDB 50K reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/))
instead of the synthetic one.

## Next steps / ideas
- Swap in the IMDB dataset (or your own labeled data) for real generalization.
- Add `ngram_range=(1, 2)` to the TF-IDF vectorizer to capture short phrases.
- Add a confusion matrix / metrics tab to the Streamlit app.
- Add basic unit tests for `train_and_save()` and `predict()`.
