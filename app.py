"""Learning Topology - interfata grafica (Streamlit).

Pornire:  python -m streamlit run app.py     (sau dublu-click pe run_gui.bat)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lt import dataset as ds
from lt import plotting
from lt.simp import (ANGLE_RANGE, NELX, NELY, PX_RANGE, PY_RANGE, VOLFRAC_RANGE, Conditions,
                     compliance, enforce_volume, run_simp)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "dataset.npz"

st.set_page_config(page_title="Learning Topology", page_icon="🧩", layout="wide")


# ----------------------------------------------------------------------------
# Resurse cache-uite
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_predictor(path: str, mtime: float):
    from lt.unet import Predictor
    p = Predictor(path)
    if p.config.get("task") != "sosnovik":
        p.predict_one([1.0, 0.5, -90.0, 0.4])   # incalzire (prima inferenta e mai lenta)
    return p


@st.cache_data(show_spinner=False, max_entries=64)
def cached_simp(px: float, py: float, angle: float, volfrac: float):
    r = run_simp(Conditions(px, py, angle, volfrac))
    return r["density"], r["compliance"], r["iterations"], r["time_s"], r["history"]


@st.cache_data(show_spinner=False)
def cached_dataset(path: str, mtime: float):
    return ds.load(path)


def list_models() -> list[Path]:
    return sorted(MODELS.glob("*.pt"), key=lambda p: (p.name != "unet.pt", p.name))


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plotting.plt.close(fig)


# ----------------------------------------------------------------------------
# Antet
# ----------------------------------------------------------------------------
st.title("Learning Topology")
st.caption("Optimizare topologica accelerata cu CNN (U-Net) - consola 2D "
           f"{NELX}x{NELY}, referinta SIMP (minimizarea compliantei la volum impus)")

tab_pred, tab_data, tab_train, tab_eval, tab_sos, tab_about = st.tabs(
    ["Predictie interactiva", "Generare date", "Antrenare", "Evaluare", "Dataset Sosnovik",
     "Despre proiect"])

# ============================================================================
# 1. PREDICTIE INTERACTIVA
# ============================================================================
with tab_pred:
    models = list_models()
    if not models:
        st.warning("Nu exista niciun model in `models/`. Genereaza date si antreneaza un model "
                   "din tab-urile urmatoare.")
    else:
        left, right = st.columns([1, 3], gap="large")
        with left:
            st.subheader("Conditii de intrare")
            model_path = st.selectbox("Model", models, format_func=lambda p: p.name)

            if "cond" not in st.session_state:
                st.session_state.cond = (1.0, 0.5, -90.0, 0.40)
            if st.button("Caz aleator din setul de test", width="stretch") and DATA.exists():
                d = cached_dataset(str(DATA), DATA.stat().st_mtime)
                _, _, te = ds.split_indices(len(d["conds"]))
                c = d["conds"][np.random.choice(te)]
                st.session_state.cond = tuple(float(v) for v in c)
            px0, py0, a0, v0 = st.session_state.cond

            px = st.slider("Pozitia fortei pe x (fractie din lungime)", *PX_RANGE, float(px0), 0.01)
            py = st.slider("Pozitia fortei pe y (0 = jos, 1 = sus)", *PY_RANGE, float(py0), 0.01)
            angle = st.slider("Directia fortei (grade, 0 = dreapta, 90 = sus)",
                              *ANGLE_RANGE, float(a0), 1.0)
            volfrac = st.slider("Fractia de volum impusa", *VOLFRAC_RANGE, float(v0), 0.01)
            st.session_state.cond = (px, py, angle, volfrac)

            compare = st.toggle("Compara cu SIMP (referinta)", value=True)
            fix_vol = st.toggle("Corecteaza volumul predictiei", value=True,
                                help="Post-procesare: deplaseaza densitatile ca media sa fie exact "
                                     "fractia de volum impusa (impune constrangerea).")

        with right:
            cond = Conditions(px, py, angle, volfrac)
            pred = get_predictor(str(model_path), model_path.stat().st_mtime)
            d_cnn, t_cnn = pred.predict_one(cond.as_array())
            if fix_vol:
                d_cnn = enforce_volume(d_cnn, volfrac)

            if compare:
                with st.spinner("Rulez SIMP..."):
                    d_simp, c_simp, iters, t_simp, hist = cached_simp(px, py, angle, volfrac)
                c_cnn = compliance(d_cnn, cond)
                dc = (c_cnn - c_simp) / c_simp * 100

                m = st.columns(5)
                m[0].metric("Timp CNN", f"{t_cnn * 1000:.1f} ms")
                m[1].metric("Timp SIMP", f"{t_simp:.2f} s", f"{iters} iteratii", delta_color="off")
                m[2].metric("Accelerare", f"{t_simp / t_cnn:,.0f}x")
                m[3].metric("Complianta SIMP / CNN", f"{c_simp:.2f} / {c_cnn:.2f}")
                m[4].metric("Δc (CNN fata de SIMP)", f"{dc:+.1f} %",
                            "mai bun" if dc <= 0 else "mai flexibil", delta_color="inverse"
                            if dc > 0 else "normal")
                show(plotting.compare_figure(cond, d_simp, d_cnn))
                with st.expander("Convergenta SIMP (complianta pe iteratii)"):
                    st.line_chart(pd.DataFrame({"complianta": hist}), x_label="iteratie")
            else:
                m = st.columns(3)
                m[0].metric("Timp CNN", f"{t_cnn * 1000:.1f} ms")
                m[1].metric("Volum prezis", f"{d_cnn.mean():.3f}", f"impus {volfrac:.2f}",
                            delta_color="off")
                m[2].metric("Complianta (FE)", f"{compliance(d_cnn, cond):.2f}")
                show(plotting.compare_figure(cond, None, d_cnn))

            st.caption("Negru = material, sageata rosie = forta, albastru = incastrare. "
                       "Complianta mai mica = structura mai rigida.")

# ============================================================================
# 2. GENERARE DATE
# ============================================================================
with tab_data:
    st.subheader("Generarea setului de date (DoE + SIMP)")
    st.write("Conditiile sunt esantionate prin **Latin Hypercube** in domeniul: "
             f"x ∈ {PX_RANGE}, y ∈ {PY_RANGE}, unghi ∈ {ANGLE_RANGE}°, "
             f"fractie volum ∈ {VOLFRAC_RANGE}. Pentru fiecare caz se ruleaza SIMP pana la "
             "convergenta, iar layout-ul final devine tinta pentru CNN.")

    if DATA.exists():
        d = cached_dataset(str(DATA), DATA.stat().st_mtime)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cazuri in dataset", len(d["conds"]))
        c2.metric("Timp mediu SIMP", f"{d['time_s'].mean():.2f} s")
        c3.metric("Iteratii medii", f"{d['iterations'].mean():.0f}")
        c4.metric("Fisier", f"{DATA.stat().st_size / 1e6:.1f} MB")
        with st.expander("Exemple din dataset", expanded=False):
            show(plotting.dataset_grid(d["conds"], d["density"], 12))
    else:
        st.info("Nu exista inca un dataset in `data/dataset.npz`.")

    with st.form("gen"):
        g1, g2, g3 = st.columns(3)
        n = g1.number_input("Numar de cazuri", 20, 20000, 2000, 100)
        seed = g2.number_input("Seed", 0, 10**6, 0)
        import os
        workers = g3.number_input("Procese paralele", 1, 64, max(1, (os.cpu_count() or 2) - 1))
        go = st.form_submit_button("Genereaza (suprascrie data/dataset.npz)", type="primary")
    if go:
        bar = st.progress(0.0, "Pornire...")

        def prog(done, total, eta):
            bar.progress(done / total, f"{done}/{total} cazuri - ETA {eta / 60:.1f} min")

        t0 = time.perf_counter()
        try:
            ds.generate(int(n), int(seed), int(workers), DATA, prog)
        except Exception as e:   # ex. probleme cu procesele paralele pe unele sisteme
            st.warning(f"Rularea paralela a esuat ({e}); reiau pe un singur proces.")
            ds.generate(int(n), int(seed), 1, DATA, prog)
        st.success(f"Dataset generat in {(time.perf_counter() - t0) / 60:.1f} min.")
        st.cache_data.clear()
        st.rerun()

# ============================================================================
# 3. ANTRENARE
# ============================================================================
with tab_train:
    st.subheader("Antrenarea U-Net")
    st.write("Cost = MSE pe pixel + λ·(media densitatii prezise − fractia de volum)². "
             "Datele de antrenare sunt dublate prin simetria sus-jos a problemei. "
             "Se salveaza checkpoint-ul cu cel mai mic loss de validare.")
    if not DATA.exists():
        st.info("Genereaza mai intai un dataset.")
    else:
        with st.form("train"):
            a1, a2, a3, a4, a5 = st.columns(5)
            epochs = a1.number_input("Epoci", 1, 500, 60)
            bs = a2.number_input("Batch size", 4, 512, 32)
            lr = a3.number_input("Learning rate", 1e-5, 1e-1, 2e-3, format="%.4f")
            lam = a4.number_input("λ volum", 0.0, 100.0, 1.0)
            base = a5.selectbox("Latime retea (canale)", [16, 32, 48, 64], index=1)
            out_name = st.text_input("Salveaza ca", "unet_nou.pt")
            go_t = st.form_submit_button("Porneste antrenarea", type="primary")
        st.caption("Orice alta actiune in interfata opreste antrenarea; cel mai bun checkpoint "
                   "ramane salvat.")
        if go_t:
            from lt.train import train
            bar = st.progress(0.0, "Pregatesc datele...")
            chart = st.empty()
            info = st.empty()

            def prog(ep, total, h):
                bar.progress(ep / total, f"epoca {ep}/{total}")
                chart.line_chart(pd.DataFrame({"train": h["train_loss"], "validare": h["val_loss"]},
                                              index=h["epoch"]), x_label="epoca", y_label="loss")
                info.write(f"MAE validare: **{h['val_mae'][-1]:.4f}** - "
                           f"timp scurs {h['time_s'][-1] / 60:.1f} min")

            out = MODELS / Path(out_name).name
            hist = train(DATA, out, int(epochs), int(bs), float(lr), float(lam), int(base),
                         progress=prog)
            st.success(f"Model salvat in `models/{out.name}` (cel mai bun loss validare "
                       f"{min(hist['val_loss']):.5f}). Il gasesti in tab-ul de predictie.")
            st.cache_resource.clear()

# ============================================================================
# 4. EVALUARE
# ============================================================================
with tab_eval:
    st.subheader("Evaluare pe setul de test (cazuri nevazute)")
    models = list_models()
    if not models or not DATA.exists():
        st.info("Ai nevoie de un dataset si de un model antrenat.")
    else:
        e1, e2 = st.columns([3, 1])
        m_eval = e1.selectbox("Model evaluat", models, format_func=lambda p: p.name, key="m_eval")
        out_dir = REPORTS / m_eval.stem
        if e2.button("Ruleaza evaluarea", type="primary", width="stretch"):
            from lt.evaluate import evaluate
            bar = st.progress(0.0, "Evaluez...")
            evaluate(m_eval, DATA, out_dir,
                     progress=lambda d, n: bar.progress(d / n, f"{d}/{n} cazuri test (FE)"))
            bar.empty()

        summ = out_dir / "summary.json"
        if summ.exists():
            s = json.loads(summ.read_text(encoding="utf-8"))
            st.markdown("**1. Asemanare structurala**")
            k = st.columns(3)
            k[0].metric("MAE densitate", f"{s['mae_mean']:.3f}")
            k[1].metric("IoU (prag 0.5)", f"{s['iou_mean']:.3f}")
            k[2].metric("Cazuri test", s["n_test"])
            st.markdown("**2. Metrica fizica - complianta CNN fata de SIMP (Δc)**")
            k = st.columns(4)
            k[0].metric("Δc median", f"{s['dc_median_pct']:+.1f} %")
            k[1].metric("Δc mediu", f"{s['dc_mean_pct']:+.1f} %")
            k[2].metric("Cazuri cu Δc < 5%", f"{s['frac_dc_below_5pct'] * 100:.0f} %")
            k[3].metric("Cazuri cu Δc < 10%", f"{s['frac_dc_below_10pct'] * 100:.0f} %")
            st.markdown("**3. Volum si timp**")
            k = st.columns(4)
            k[0].metric("|eroare volum| medie (brut)", f"{s['vol_err_abs_mean']:.3f}")
            k[1].metric("Timp CNN / caz", f"{s['cnn_ms_single']:.1f} ms")
            k[2].metric("Timp SIMP / caz", f"{s['simp_s_mean']:.2f} s")
            k[3].metric("Accelerare", f"{s['speedup_single']:,.0f}x")

            f1, f2 = st.columns(2)
            for col, name, cap in [(f1, "hist_dc.png", "Distributia Δc"),
                                   (f2, "learning_curve.png", "Curba de invatare"),
                                   (f1, "dc_vs_volfrac.png", "Δc in functie de fractia de volum")]:
                if (out_dir / name).exists():
                    col.image(str(out_dir / name), caption=cap, width="stretch")
            if (out_dir / "examples.png").exists():
                f2.image(str(out_dir / "examples.png"), caption="Cel mai bun / median / cel mai slab",
                         width="stretch")
            csvp = out_dir / "metrics_test.csv"
            if csvp.exists():
                df = pd.read_csv(csvp)
                st.markdown("**Cele mai slabe 10 cazuri** (unde cedeaza modelul)")
                st.dataframe(df.sort_values("dc_pct", ascending=False).head(10),
                             width="stretch", hide_index=True)
                st.download_button("Descarca metricile (CSV)", csvp.read_bytes(),
                                   file_name=f"metrics_{m_eval.stem}.csv")
        else:
            st.info("Apasa **Ruleaza evaluarea** pentru a genera raportul.")

# ============================================================================
# 5. DATASET SOSNOVIK & OSELEDETS (date externe, generate independent cu ToPy)
# ============================================================================
with tab_sos:
    from lt import sosnovik as sos

    st.subheader("Dataset public TOP - Sosnovik & Oseledets (2017)")
    st.markdown(
        "10.000 de probleme de optimizare topologica generate cu **ToPy** (alt solver SIMP decat "
        "cel din proiect), grila 40×40, cu istoria a 100 de iteratii pentru fiecare problema. "
        "Datasetul nu contine fortele, deci sarcina este cea din articol: **CNN-ul primeste starea "
        "SIMP la iteratia k si prezice direct layout-ul final** - SIMP \"terminat\" dupa k iteratii. "
        "Baseline-ul este pragul naiv (binarizarea lui x_k la 0.5).")

    with st.expander("Cum obtii datele", expanded=not sos.PACKED.exists()):
        st.markdown(
            "1. Deschide pagina [github.com/ISosnovik/top](https://github.com/ISosnovik/top) si "
            "urmeaza linkul de descarcare (Yandex Disk, ~3 GB).\n"
            "2. Dezarhiveaza in folderul `data\\sosnovik_raw` din proiect "
            "(sau in orice folder si scrie calea mai jos).\n"
            "3. Apasa **Importa** - se pastreaza doar cadrele necesare (k = "
            f"{', '.join(map(str, sos.K_VALUES))} si starea finala), intr-un fisier compact.")

    with st.form("sos_import"):
        i1, i2 = st.columns([3, 1])
        src = i1.text_input("Folderul cu fisierele .npz originale", str(sos.RAW_DIR))
        max_n = i2.number_input("Maxim cazuri (0 = toate)", 0, 100000, 0, 500)
        go_i = st.form_submit_button("Importa", type="primary")
    if go_i:
        files = sos.find_files(src) if Path(src).exists() else []
        if not files:
            st.error(f"Nu am gasit fisiere .npz in `{src}`.")
        else:
            bar = st.progress(0.0, f"Citesc {len(files)} fisiere...")
            try:
                r = sos.import_dataset(src, sos.PACKED, int(max_n) or None,
                                       lambda d, n: bar.progress(d / n, f"{d}/{n} fisiere"))
                st.success(f"Importat: {r['n_samples']} cazuri, grila {r['shape'][0]}×"
                           f"{r['shape'][1]} (sarite: {r['skipped']}).")
            except Exception as e:
                st.error(f"Importul a esuat: {e}")

    if sos.PACKED.exists():
        sd = sos.load_packed(sos.PACKED)
        frames, kv = sd["frames"], sd["k_values"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Cazuri importate", len(frames))
        s2.metric("Grila", f"{frames.shape[2]}×{frames.shape[3]}")
        s3.metric("Iteratii k pastrate", ", ".join(map(str, kv)))
        with st.expander("Exemple: evolutia SIMP pe iteratii"):
            show(sos.history_figure(frames, kv))

        st.markdown("#### Antrenare")
        with st.form("sos_train"):
            t1, t2, t3, t4 = st.columns(4)
            s_ep = t1.number_input("Epoci", 1, 300, 30, key="s_ep")
            s_bs = t2.number_input("Batch size", 8, 512, 64, key="s_bs")
            s_lr = t3.number_input("Learning rate", 1e-5, 1e-1, 2e-3, format="%.4f", key="s_lr")
            s_lam = t4.number_input("λ volum", 0.0, 100.0, 1.0, key="s_lam")
            go_st = st.form_submit_button("Antreneaza pe datele Sosnovik", type="primary")
        st.caption("k este ales aleator la fiecare pas (model valabil pentru orice k), cu "
                   "augmentare prin cele 8 simetrii ale patratului. Cost = BCE + termen de volum.")
        if go_st:
            bar = st.progress(0.0, "Pornire...")
            chart = st.empty()

            def sprog(ep, total, h):
                bar.progress(ep / total, f"epoca {ep}/{total} - acuratete validare "
                                         f"{h['val_acc'][-1]:.4f}")
                chart.line_chart(pd.DataFrame({"train": h["train_loss"], "validare": h["val_loss"]},
                                              index=h["epoch"]), x_label="epoca", y_label="loss")

            sos.train(sos.PACKED, sos.MODEL, int(s_ep), int(s_bs), float(s_lr), float(s_lam),
                      progress=sprog)
            st.success("Model salvat in `models/sosnovik/unet_sosnovik.pt`.")
            st.cache_resource.clear()

        if sos.MODEL.exists():
            st.markdown("#### Evaluare pe setul de test")
            if st.button("Ruleaza evaluarea Sosnovik", type="primary"):
                bar = st.progress(0.0, "Evaluez...")
                sos.evaluate(progress=lambda d, n: bar.progress(d / n, f"k {d}/{n}"))
                bar.empty()
            ssum = sos.REPORTS / "summary.json"
            if ssum.exists():
                s = json.loads(ssum.read_text(encoding="utf-8"))
                df = pd.DataFrame(s["per_k"]).rename(columns={
                    "acc_cnn": "acuratete CNN", "acc_prag": "acuratete prag naiv",
                    "iou_cnn": "IoU CNN", "iou_prag": "IoU prag naiv",
                    "vol_err_cnn": "|eroare volum| CNN", "ms_per_caz": "ms / caz"})
                st.dataframe(df, width="stretch", hide_index=True)
                st.caption(f"{s['n_test']} cazuri de test, nevazute la antrenare. "
                           "Fara forte in dataset nu se poate calcula complianta, deci "
                           "comparatia este geometrica (fata de layout-ul SIMP final).")
                f1, f2 = st.columns(2)
                for col, name, cap in [(f1, "acc_vs_k.png", "Acuratete in functie de k"),
                                       (f2, "learning_curve.png", "Curba de invatare")]:
                    if (sos.REPORTS / name).exists():
                        col.image(str(sos.REPORTS / name), caption=cap, width="stretch")
                if (sos.REPORTS / "examples.png").exists():
                    st.image(str(sos.REPORTS / "examples.png"),
                             caption=f"Exemple la k = {kv[0]}", width="stretch")

            st.markdown("#### Incearca pe un caz de test")
            _, _, te_s = ds.split_indices(len(frames))
            c1, c2 = st.columns([1, 3])
            case = c1.number_input("Caz de test (index)", 0, len(te_s) - 1, 0)
            k_sel = c1.select_slider("Iteratia k", options=list(kv), value=kv[0])
            spred = get_predictor(str(sos.MODEL), sos.MODEL.stat().st_mtime)
            x, y = sos.make_inputs(frames[[te_s[int(case)]]], list(kv).index(k_sel))
            p = spred.predict_raw(x)
            acc, iou_v = sos._binary_metrics(p, y[:, 0])
            acc_t, iou_t = sos._binary_metrics(x[:, 0], y[:, 0])
            c1.metric("IoU CNN / prag naiv", f"{iou_v[0]:.3f} / {iou_t[0]:.3f}")
            with c2:
                show(sos.examples_figure([("Caz", x[0, 0], p[0], y[0, 0], iou_v[0])], k_sel))

# ============================================================================
# 6. DESPRE
# ============================================================================
with tab_about:
    st.markdown("""
### Ce face proiectul
O retea neuronala convolutionala (**U-Net**) invata sa prezica direct layout-ul optim de material
al unei console 2D, adica rezultatul pe care in mod normal il da **optimizarea topologica SIMP**,
dar intr-o singura trecere, fara iteratii. SIMP este modelul de referinta: genereaza datele de
antrenare si serveste la verificare.

### Cele 6 intrebari (traseul CAE-ML)
1. **Ce problema rezolvam?** Layout de rigiditate maxima (complianta minima) la fractie de volum impusa.
2. **Ce stim despre sistem?** Domeniu 64x32, latura stanga incastrata, forta punctuala (pozitie + directie), material liniar elastic (E = 1, ν = 0.3).
3. **Cum obtinem datele?** Esantionare Latin Hypercube + rulari SIMP automate.
4. **Ce rol are ML?** Mapare image-to-image: conditii (codificate pe 6 canale) → camp de densitati.
5. **Cum verificam?** Pe cazuri nevazute: MAE/IoU, dar mai ales **complianta FE** a layout-ului prezis fata de SIMP (Δc) si respectarea volumului.
6. **Ce castigam?** Timp: milisecunde in loc de sute de iteratii SIMP.

### Rulare din linia de comanda
```
python -m lt.dataset --n 2000          # generare date
python -m lt.train --epochs 60         # antrenare
python -m lt.evaluate                  # raport in reports/
```
""")
