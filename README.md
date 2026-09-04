# Predikcija kategorije proizvoda na osnovu naslova

Projekat mašinskog učenja za automatsko predlaganje kategorije proizvoda na osnovu njegovog naziva.

Model je namenjen online trgovinama koje svakodnevno unose veliki broj novih proizvoda. Automatska kategorizacija može ubrzati unos artikala, smanjiti broj ručnih grešaka i unaprediti pretragu proizvoda.

## Autor

Stefan Stojanović

## Cilj projekta

Cilj je razvoj kompletnog i ponovljivog ML rešenja koje:

- učitava i čisti podatke o proizvodima;
- analizira raspodelu kategorija;
- kreira tekstualne i numeričke karakteristike;
- poredi više algoritama;
- bira i evaluira najbolje rešenje;
- čuva trenirani model u `.pkl` formatu;
- omogućava interaktivno testiranje novih naslova.

## Skup podataka

Projekat koristi skup `products.csv` sa početnih 35.311 proizvoda i sledećim kolonama:

- `Product ID` – jedinstveni identifikator proizvoda;
- `Product Title` – naziv proizvoda;
- `Merchant ID` – identifikator prodavca;
- `Category Label` – ciljna kategorija;
- `Product Code` – interni kod proizvoda;
- `Number of Views` – broj pregleda;
- `Merchant Rating` – ocena prodavca;
- `Listing Date` – datum postavljanja.

Dataset je dostavljen u okviru materijala kursa.

## Čišćenje podataka

Tokom pripreme podataka izvršeni su sledeći koraci:

- standardizovani su nazivi kolona;
- uklonjeni su redovi bez naslova ili ciljne kategorije;
- objedinjene su nedosledno napisane oznake:
  - `CPU` → `CPUs`;
  - `Mobile Phone` → `Mobile Phones`;
  - `fridge` → `Fridges`;
- uklonjena su četiri reda sa konfliktnim oznakama;
- uklonjeni su ponovljeni normalizovani naslovi;
- zadržano je 30.822 jedinstvena proizvoda iz 10 kategorija.

Uklanjanje ponovljenih naslova sprečava da se isti tekst pojavi i u trening i u test skupu i tako veštački poveća rezultat evaluacije.

## Inženjering karakteristika

Model kombinuje tri grupe karakteristika:

1. TF-IDF karakteristike reči i parova reči;
2. karakterne n-grame dužine od tri do pet karaktera;
3. numeričke karakteristike izvedene iz naslova:
   - broj karaktera;
   - broj reči;
   - broj cifara;
   - prisustvo broja;
   - broj specijalnih znakova;
   - dužina najduže reči.

Karakterne n-grame koristimo jer dobro prepoznaju delove reči, spojene izraze i šifre modela kao što su `kgv39vl31g` i `sbs8004po`.

Analizirana je i karakteristika broja reči napisanih velikim slovima, ali je odbačena jer su svi naslovi u skupu zapisani malim slovima.

## Testirani modeli

Tokom razvoja upoređeni su sledeći pristupi:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression – word TF-IDF | 0,9526 | 0,9549 | 0,9527 |
| Linear SVC – word TF-IDF | 0,9601 | 0,9620 | 0,9601 |
| Multinomial NB – word TF-IDF | 0,9619 | 0,9634 | 0,9619 |
| Linear SVC – word + char TF-IDF | 0,9877 | 0,9876 | 0,9877 |
| Linear SVC – word + char + numeric | 0,9880 | 0,9880 | 0,9880 |
| Poboljšani Linear SVC | **0,9908** | **0,9907** | **0,9908** |

Za finalni model povećan je relativni značaj karakternih n-grama i smanjena težina pomoćnih numeričkih karakteristika.

## Finalni rezultat

Finalni model je **Linear SVC** sa kombinovanim tekstualnim i numeričkim karakteristikama.

Na test skupu od 6.165 proizvoda ostvario je:

- accuracy: **99,08%**;
- macro F1: **99,07%**;
- weighted F1: **99,08%**;
- 57 pogrešnih predikcija;
- stopu greške od približno 0,92%.

Najviše preostalih grešaka javlja se između sličnih kategorija `Fridges`, `Freezers` i `Fridge Freezers`.

## Struktura projekta

```text
product-category-classifier/
├── data/
│   └── products.csv
├── models/
│   └── product_category_model.pkl
├── notebooks/
│   └── product_category_classification.ipynb
├── .gitignore
├── predict_category.py
├── README.md
├── requirements.txt
└── train_model.py
```

## Instalacija

### 1. Kloniranje repozitorijuma

```bash
git clone https://github.com/stefanstojanovic994/product-category-classifier.git
cd product-category-classifier
```

### 2. Kreiranje virtuelnog okruženja

Windows:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instaliranje biblioteka

```bash
pip install -r requirements.txt
```

## Treniranje modela

Za ponovno čišćenje podataka, treniranje i čuvanje modela pokrenuti:

```bash
python train_model.py
```

Skript:

1. učitava `data/products.csv`;
2. čisti i standardizuje podatke;
3. kreira karakteristike;
4. trenira finalni Pipeline;
5. čuva model kao `models/product_category_model.pkl`.

## Interaktivno predviđanje

Za interaktivno testiranje pokrenuti:

```bash
python predict_category.py
```

Primer:

```text
Naziv proizvoda: smeg sbs8004po
Predviđena kategorija: Fridge Freezers
```

Za završetak programa uneti:

```text
exit
```

## Ručni testovi

Finalni model pravilno je klasifikovao svih šest proizvoda iz instrukcija:

| Naziv proizvoda | Predviđena kategorija |
|---|---|
| iphone 7 32gb gold,4,3,Apple iPhone 7 32GB | Mobile Phones |
| olympus e m10 mark iii geh use silber | Digital Cameras |
| kenwood k20mss15 solo | Microwaves |
| bosch wap28390gb 8kg 1400 spin | Washing Machines |
| bosch serie 4 kgv39vl31g | Fridge Freezers |
| smeg sbs8004po | Fridge Freezers |

## Jupyter analiza

Kompletna analiza dostupna je u:

```text
notebooks/product_category_classification.ipynb
```

Sveska sadrži:

- početnu analizu i čišćenje;
- vizualizacije;
- inženjering karakteristika;
- stratifikovanu podelu podataka;
- poređenje više modela;
- klasifikacione izveštaje;
- matrice zabune;
- analizu pogrešnih predikcija;
- ručno testiranje;
- čuvanje i proveru modela.

## Ograničenja i moguća poboljšanja

Model koristi samo informacije dostupne u nazivu proizvoda. Kratki ili nejasni naslovi koji sadrže samo brend i šifru modela mogu biti teži za klasifikaciju.

Moguća buduća poboljšanja:

- dodatna provera nedoslednih ciljnih oznaka;
- korišćenje opisa proizvoda ako postane dostupan;
- podešavanje hiperparametara pomoću unakrsne validacije;
- vraćanje pouzdanosti predikcije;
- praćenje performansi na novim proizvodima;
- periodično ponovno treniranje modela.

## Napomena

Sačuvani `.pkl` fajl treba učitavati samo iz pouzdanog izvora. Za reprodukciju modela uvek je moguće ponovo pokrenuti `train_model.py`.