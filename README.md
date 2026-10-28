# Learning Topology — optimizare topologică accelerată cu CNN

Proiect pentru disciplina *Artificial Intelligence and Machine Learning Applied in CAE and CAM*
(master ACE Craiova). O rețea **U-Net** învață să prezică direct layout-ul optim de material al
unei console 2D — rezultatul pe care îl dă în mod normal **optimizarea topologică SIMP** — într-o
singură trecere, fără iterații. Totul este software (Python).

## Pornire rapidă (Windows)

1. Instalează **Python 3.10+** de pe python.org (bifează *Add Python to PATH*).
2. Dublu-click pe **`setup.bat`** — creează mediul `.venv` și instalează dependențele (o singură dată).
3. Dublu-click pe **`run_gui.bat`** — se deschide interfața în browser (`http://localhost:8501`).

Proiectul vine cu un **dataset generat** (`data/dataset.npz`, 2000 de cazuri SIMP) și un **model
pre-antrenat** (`models/unet.pt`), deci interfața funcționează imediat.

## Interfața (5 tab-uri)

| Tab | Ce face |
| --- | --- |
| Predicție interactivă | Alegi poziția, direcția forței și fracția de volum → CNN-ul dă layout-ul instant; opțional rulează SIMP alături și compară complianța (Δc) și timpul. |
| Generare date | Eșantionare Latin Hypercube + rulări SIMP în paralel → `data/dataset.npz`. |
| Antrenare | Antrenează U-Net cu grafic live al loss-ului; salvează în `models/`. |
| Evaluare | Metrici pe setul de test: MAE, IoU, Δc (complianță FE), eroare de volum, timp; figuri și cele mai slabe cazuri. |
| Despre proiect | Rezumat și cele 6 întrebări din tutorial. |

## Dataset extern: Sosnovik & Oseledets (TOP)

Tab-ul **Dataset Sosnovik** folosește datasetul public [TOP](https://github.com/ISosnovik/top)
(10.000 de probleme, grilă 40×40, generate cu ToPy). Sarcina: CNN-ul primește starea SIMP la
iterația k și prezice layout-ul final; comparație cu pragul naiv pe x_k.

1. Descarcă arhiva (~3 GB, Yandex Disk) de pe pagina GitHub și dezarhiveaz-o în `data\sosnovik_raw`.
2. În tab: **Importă** → **Antrenează** → **Evaluare**.

Din linia de comandă: `python -m lt.sosnovik import`, `python -m lt.sosnovik train`,
`python -m lt.sosnovik evaluate`.

## Linie de comandă

```
.venv\Scripts\activate
python -m lt.dataset --n 2000          # generare date
python -m lt.train --epochs 60         # antrenare  -> models/unet.pt
python -m lt.evaluate                  # raport     -> reports/unet/
```
sau `run_pipeline.bat` pentru toți pașii (suprascrie datasetul și modelul).

## Problema

- **Domeniu:** consolă 2D, grilă 64×32 elemente, latura stângă încastrată.
- **Variabile de intrare:** poziția forței (x ∈ [0,5; 1], y ∈ [0; 1]), direcția (−180°…180°),
  fracția de volum (0,25…0,55).
- **Parametri impuși:** E = 1, ν = 0,3, penalizare SIMP p = 3, rază filtru 1,5.
- **Model de referință:** SIMP, min c(ρ) = UᵀK(ρ)U s.t. V(ρ)/V₀ = f.
- **Model ML:** U-Net, 6 canale de intrare (Fx, Fy, încastrare, fracție volum, x, y) → densitate.
- **Cost:** MSE + λ·(volum prezis − f)² (termen „physics-informed").
- **Verificare:** complianța FE a layout-ului prezis (cu volumul corectat) față de SIMP: Δc %.

## Structura

```
proiect/
├─ app.py              interfața grafică (Streamlit)
├─ lt/
│  ├─ simp.py          solver SIMP (88-line vectorizat, Cholesky bandă) + complianță FE
│  ├─ encoding.py      condiții -> tensor de intrare (6 canale)
│  ├─ dataset.py       LHS + generare paralelă
│  ├─ unet.py          arhitectura U-Net + Predictor
│  ├─ train.py         antrenare
│  ├─ evaluate.py      metrici + figuri
│  └─ plotting.py      figuri
├─ data/dataset.npz    date generate
├─ models/unet.pt      model pre-antrenat
├─ reports/unet/       raportul de evaluare (summary.json, metrics_test.csv, figuri)
├─ setup.bat  run_gui.bat  run_pipeline.bat
└─ requirements.txt
```

## Note

- Dacă ai placă NVIDIA, poți instala varianta CUDA de PyTorch; codul folosește GPU automat.
- Generarea a 2000 de cazuri durează ~5–15 min, în funcție de numărul de nuclee.
