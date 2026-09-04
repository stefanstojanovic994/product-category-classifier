"""
Treniranje modela za predikciju kategorije proizvoda.

Skript učitava podatke, čisti ih, kreira karakteristike,
trenira finalni Pipeline i čuva ga u .pkl formatu.
"""

from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# Putanje se određuju u odnosu na lokaciju ovog skripta
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "products.csv"
MODELS_DIRECTORY = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIRECTORY / "product_category_model.pkl"

NUMERIC_FEATURES = [
    "title_length",
    "word_count",
    "digit_count",
    "contains_number",
    "special_char_count",
    "longest_word_length"
]


def load_and_clean_data(data_path):
    """
    Učitava i čisti skup proizvoda.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset nije pronađen: {data_path}"
        )

    df = pd.read_csv(data_path)

    # Standardizacija naziva kolona
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    required_columns = {
        "product_title",
        "category_label"
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Nedostaju obavezne kolone: {sorted(missing_columns)}"
        )

    # Uklanjanje redova bez naslova ili kategorije
    df = df.dropna(
        subset=["product_title", "category_label"]
    ).copy()

    # Standardizacija tekstualnih vrednosti
    df["product_title"] = (
        df["product_title"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["category_label"] = (
        df["category_label"]
        .astype(str)
        .str.strip()
    )

    # Objedinjavanje nedosledno napisanih kategorija
    category_mapping = {
        "CPU": "CPUs",
        "Mobile Phone": "Mobile Phones",
        "fridge": "Fridges"
    }

    df["category_label"] = (
        df["category_label"]
        .replace(category_mapping)
    )

    # Uklanjanje eventualnih praznih tekstualnih vrednosti
    df = df[
        (df["product_title"].str.len() > 0)
        & (df["category_label"].str.len() > 0)
    ].copy()

    # Normalizovani naslov za proveru ponavljanja
    df["normalized_title"] = (
        df["product_title"]
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Uklanjanje naslova povezanih sa više kategorija
    categories_per_title = (
        df.groupby("normalized_title")["category_label"]
        .nunique()
    )

    conflicting_titles = categories_per_title[
        categories_per_title > 1
    ].index

    df = df[
        ~df["normalized_title"].isin(conflicting_titles)
    ].copy()

    # Zadržavanje jednog reda po normalizovanom naslovu
    df = (
        df.drop_duplicates(
            subset="normalized_title",
            keep="first"
        )
        .reset_index(drop=True)
    )

    return df


def create_title_features(product_titles):
    """
    Kreira ulaznu tabelu sa tekstualnim i numeričkim
    karakteristikama izvedenim iz naslova.
    """
    titles = pd.Series(product_titles).astype(str)

    features = pd.DataFrame({
        "product_title": titles
    })

    features["title_length"] = (
        features["product_title"].str.len()
    )

    features["word_count"] = (
        features["product_title"].str.split().str.len()
    )

    features["digit_count"] = (
        features["product_title"].str.count(r"\d")
    )

    features["contains_number"] = (
        features["product_title"]
        .str.contains(r"\d", regex=True)
        .astype(int)
    )

    features["special_char_count"] = (
        features["product_title"]
        .str.count(r"[^A-Za-z0-9\s]")
    )

    features["longest_word_length"] = (
        features["product_title"]
        .str.split()
        .apply(
            lambda words: max(
                (len(word) for word in words),
                default=0
            )
        )
    )

    return features


def build_model():
    """
    Kreira finalni Pipeline za obradu karakteristika
    i višeklasnu klasifikaciju.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True
                ),
                "product_title"
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=60000,
                    sublinear_tf=True
                ),
                "product_title"
            ),
            (
                "numeric_features",
                StandardScaler(with_mean=False),
                NUMERIC_FEATURES
            )
        ],
        transformer_weights={
            "word_tfidf": 1.0,
            "char_tfidf": 3.0,
            "numeric_features": 0.1
        }
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                max_iter=10000,
                random_state=42
            )
        )
    ])

    return model


def train_and_save_model():
    """
    Pokreće kompletan proces treniranja i čuvanja modela.
    """
    print("Učitavanje i čišćenje podataka...")

    df = load_and_clean_data(DATA_PATH)

    X = create_title_features(df["product_title"])
    y = df["category_label"]

    print(f"Broj proizvoda za treniranje: {len(df)}")
    print(f"Broj kategorija: {y.nunique()}")
    print("Treniranje modela...")

    model = build_model()

    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH,
        compress=3
    )

    model_size_mb = (
        MODEL_PATH.stat().st_size / (1024 ** 2)
    )

    print("\nTreniranje je uspešno završeno.")
    print(f"Vreme treniranja: {training_time:.2f} sekundi")
    print(f"Model je sačuvan: {MODEL_PATH}")
    print(f"Veličina modela: {model_size_mb:.2f} MB")


if __name__ == "__main__":
    train_and_save_model()