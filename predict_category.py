"""
Interaktivno predviđanje kategorije proizvoda.

Korisnik unosi naziv proizvoda, a sačuvani model
predlaže odgovarajuću kategoriju.
"""

from pathlib import Path

import joblib

from train_model import create_title_features


# Putanja do sačuvanog modela
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "product_category_model.pkl"
)


def load_model(model_path):
    """
    Učitava prethodno trenirani model sa diska.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            "Model nije pronađen. "
            "Prvo pokrenite train_model.py."
        )

    return joblib.load(model_path)


def predict_category(model, product_title):
    """
    Kreira karakteristike i predviđa kategoriju
    za jedan naziv proizvoda.
    """
    features = create_title_features([product_title])
    prediction = model.predict(features)[0]

    return prediction


def run_interactive_prediction():
    """
    Pokreće interaktivni unos naziva proizvoda.
    """
    try:
        model = load_model(MODEL_PATH)
    except Exception as error:
        print(f"Greška pri učitavanju modela: {error}")
        return

    print("=" * 60)
    print("PREDIKCIJA KATEGORIJE PROIZVODA")
    print("=" * 60)
    print("Unesite naziv proizvoda.")
    print("Za završetak unesite: exit")
    print()

    while True:
        product_title = input("Naziv proizvoda: ").strip()

        if product_title.lower() in {
            "exit",
            "quit",
            "izlaz"
        }:
            print("Program je završen.")
            break

        if not product_title:
            print("Naziv ne može biti prazan.\n")
            continue

        try:
            predicted_category = predict_category(
                model,
                product_title
            )

            print(
                f"Predviđena kategorija: "
                f"{predicted_category}\n"
            )

        except Exception as error:
            print(
                f"Greška tokom predikcije: {error}\n"
            )


if __name__ == "__main__":
    run_interactive_prediction()