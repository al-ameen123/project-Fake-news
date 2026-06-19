# 🔍 Fake News Detector

A polished, high-performance machine learning application designed to classify news articles and headlines as **Real** or **Fake**. The system uses a **PassiveAggressiveClassifier** paired with **TF-IDF (Term Frequency-Inverse Document Frequency)** feature extraction to achieve state-of-the-art text classification.

This repository has been structured to separate the **Frontend UI interface** and the **Backend machine learning logic & Streamlit dashboard**.

---

## 🗺️ File Map & Project Structure

Here is the complete layout of the project:

```text
project-Fake-news/
├── backend/                  # Machine learning pipeline and Streamlit dashboard
│   ├── __init__.py           # Marks backend as a Python package
│   ├── app.py                # Main Streamlit web application dashboard
│   ├── train_model.py        # Model training pipeline script
│   ├── utils.py              # Text preprocessing & NLP cleaning utilities
│   ├── data/                 # Dataset directory
│   │   ├── Fake.csv          # Dataset of fake news articles (Kaggle)
│   │   ├── True.csv          # Dataset of real news articles (Kaggle)
│   │   └── .gitkeep
│   └── model/                # Serialized model artifacts
│       ├── metadata.json     # Model metrics (accuracy, confusion matrix, etc.)
│       ├── pac_model.joblib  # Pre-trained PassiveAggressiveClassifier model
│       └── tfidf_vectorizer.joblib # Fit TF-IDF Vectorizer
├── frontend/                 # Static web interface (Premium UI mockup)
│   ├── css/
│   │   └── style.css         # Modern dark-mode styling with glassmorphism
│   ├── js/
│   │   └── main.js           # Client-side validation & simulated heuristics
│   └── index.html            # Static mockup homepage for the detector
├── .gitignore                # Files excluded from git tracking
├── requirements.txt          # Python library dependencies
└── README.md                 # Project documentation and file map (this file)
```

---

## 📊 Machine Learning Model Details

The classifier is trained on the Kaggle *Fake and Real News Dataset* (44,889 articles).

### Performance Metrics:
* **Accuracy:** **99.54%**
* **Total Samples:** 44,889 articles
  * **Training Set:** 35,911 articles
  * **Testing Set:** 8,978 articles
* **Feature Extraction:** TF-IDF Vectorizer (max 5,000 features, 1-2 n-grams)
* **Classifier:** PassiveAggressiveClassifier (max 50 iterations)

### Confusion Matrix:
| | Predicted FAKE | Predicted REAL |
|---|---|---|
| **Actual FAKE** | **4,666** *(True Neg)* | **29** *(False Pos)* |
| **Actual REAL** | **12** *(False Neg)* | **4,271** *(True Pos)* |

---

## ⚡ Getting Started

### 1. Installation & Environment Setup
Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/al-ameen123/project-Fake-news.git
cd project-Fake-news

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Model
If you want to re-train the model or update it with new dataset files, run the training pipeline:

```bash
python backend/train_model.py
```
This script will clean the dataset, extract TF-IDF features, train the PassiveAggressiveClassifier, print performance metrics, and save the updated model to `backend/model/`.

### 3. Running the Streamlit App
Launch the interactive Streamlit web dashboard:

```bash
streamlit run backend/app.py
```

### 4. Opening the Static Web UI Mockup
You can also open the premium mockup frontend directly in your browser:
* Double-click `frontend/index.html` or open it with any web browser.
* This UI displays a premium, glassmorphism design and client-side simulation.

---

## 🛠️ NLP Preprocessing Pipeline
To clean the raw text articles, the system performs the following transformations defined in `backend/utils.py`:
1. Converts all text to lowercase.
2. Removes URLs (e.g., `http://...`, `https://...`).
3. Removes punctuation, special symbols, and digits.
4. Removes standard English stopwords using **NLTK**.
5. (Optional) Applies Porter stemming to collapse tokens to their root form.
