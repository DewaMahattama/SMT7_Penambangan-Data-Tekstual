import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="ABSA", layout="wide")

# ================================================================ KONFIGURASI
BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "artifacts"
LDA_PATH = ART_DIR / "lda_artifacts.joblib"
DF_PATH = ART_DIR / "df_lda.pkl"
BERT_DIR = ART_DIR / "bert_final"
MAX_LEN = 128

SENT_ORDER = ["negatif", "netral", "positif"]
SENT_COLORS = {"negatif": "#EF4444", "netral": "#F59E0B", "positif": "#10B981"}
INDIGO = "#4F46E5"

# Hasil evaluasi pada data uji (perbarui bila model dilatih ulang)
METRICS = {
    "accuracy": 0.7029,
    "macro_f1": 0.6799,
    "n_test": 1013,
    "per_class": {  # precision, recall, f1, support
        "negatif": (0.6797, 0.7982, 0.7342, 327),
        "netral": (0.5401, 0.4414, 0.4858, 290),
        "positif": (0.8240, 0.8157, 0.8198, 396),
    },
    "confusion": [[261, 54, 12], [105, 128, 57], [18, 55, 323]],  # baris = asli, kolom = prediksi
}

EXAMPLES = {
    "Bug setelah update": "The app keeps crashing every time I open it after the latest update. Please fix this problem.",
    "Harga dan langganan": "Most features are locked behind the premium subscription. You have to pay every year, which is a cash grab.",
    "Pengingat": "The reminders work perfectly. I get a notification every day on time and it helps me build a daily habit.",
    "Kalender": "Google Calendar sync works great and all my events show up in the calendar view.",
    "Campuran": "The reminders are great, but the premium subscription is way too expensive for what it offers.",
    "Netral": "It's okay. Does what I need but nothing special.",
}

# ================================================================ STYLE
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th,
.stApp input, .stApp textarea, .stApp button { font-family: 'Inter', system-ui, 'Segoe UI', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.5rem; max-width: 1180px; }

.app-title { font-size: 26px; font-weight: 700; color: #111827; letter-spacing: -0.02em; margin-bottom: 2px; }
.app-sub { font-size: 14px; color: #6B7280; margin-bottom: 20px; }

div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
.card-title { font-size: 16px; font-weight: 600; color: #111827; margin-bottom: 2px; }
.card-sub { font-size: 13px; color: #6B7280; margin-bottom: 8px; }
.lbl { font-size: 12px; color: #6B7280; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
.big { font-size: 24px; font-weight: 700; color: #111827; letter-spacing: -0.01em; }
.badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 14px; font-weight: 600; }
.empty { color: #9CA3AF; text-align: center; padding: 70px 10px; font-size: 14px; }

.step { border: 1px solid #E5E7EB; border-radius: 12px; padding: 16px; height: 100%; background: #fff; }
.step .n { color: #4F46E5; font-size: 12px; font-weight: 700; }
.step b { display: block; color: #111827; font-size: 14px; margin: 4px 0; }
.step span { color: #6B7280; font-size: 12.5px; line-height: 1.5; }
</style>
""",
    unsafe_allow_html=True,
)

# ================================================================ PREPROCESSING (sama dengan Cell 6 notebook)
sentiment_words = {
    "good", "great", "nice", "best", "better", "awesome", "amazing", "excellent", "perfect",
    "love", "like", "liked", "loved", "enjoy", "fantastic", "wonderful", "bad", "worst",
    "terrible", "horrible", "awful", "hate", "super", "cool", "fine", "happy", "thank", "thanks",
}
generic_words = {
    "use", "used", "using", "make", "made", "want", "need", "try", "tried", "give", "gave",
    "go", "going", "know", "think", "say", "see", "look", "thing", "things", "lot", "much",
    "many", "really", "very", "just", "still", "also", "even", "ever", "always", "never",
    "would", "could", "should", "might", "get", "got", "getting", "let", "put", "take", "took",
    "come", "came", "way", "well", "back", "bit", "pretty", "quite", "actually", "definitely",
    "everything", "something", "anything", "nothing", "everyone", "someone", "please", "pls",
    "plz", "ok", "okay", "yes", "yeah", "hey", "hi", "hello", "etc", "since", "ive", "im",
    "dont", "cant", "didnt", "doesnt", "isnt", "wasnt", "wont", "youre", "thats", "theres",
    "one", "two", "first", "last", "new", "old", "every", "another", "may", "else", "though",
}
app_words = {"app", "apps", "application", "android", "ios", "google play", "play", "store",
             "developer", "dev", "devs", "star", "stars", "review", "rating", "rate", "rated"}
KEEP = {"work", "working", "fix", "update", "problem", "time", "day", "phone"}


@st.cache_resource
def get_nlp():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    stop = (set(stopwords.words("english")) | sentiment_words | generic_words | app_words) - KEEP
    return stop, WordNetLemmatizer()


STOP, LEMMA = get_nlp()


def noise_norm(text):
    t = re.sub(r"http\S+|www\.\S+", " ", str(text).lower())
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"(.)\1{2,}", r"\1\1", t)


def preprocess_for_bert(text):
    return noise_norm(text)


def preprocess_for_lda(text):
    tokens = [w for w in noise_norm(text).split() if w not in STOP and len(w) > 2]
    tokens = [LEMMA.lemmatize(w) for w in tokens]
    return [w for w in tokens if w not in STOP and len(w) > 2]


# ================================================================ MODEL
@st.cache_resource(show_spinner="Memuat model (hanya sekali)...")
def load_models():
    if not LDA_PATH.exists():
        st.error(f"File LDA tidak ditemukan: {LDA_PATH}")
        st.stop()
    if not (BERT_DIR / "config.json").exists():
        st.error(f"Model BERT tidak ditemukan di: {BERT_DIR}\n\n"
                 "Folder harus berisi config.json, model.safetensors, dan tokenizer.json.")
        st.stop()
    art = joblib.load(LDA_PATH)
    tok = AutoTokenizer.from_pretrained(str(BERT_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return art, tok, model, device


@st.cache_data
def load_df():
    return pd.read_pickle(DF_PATH)


def analyze(reviews, batch_size=32):
    art, tok, model, device = load_models()
    vec, lda, names = art["vectorizer"], art["lda"], art["ASPECT_NAMES"]
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    cleaned = [preprocess_for_lda(r) for r in reviews]
    bert_txt = [preprocess_for_bert(r) for r in reviews]

    X = vec.transform([" ".join(t) for t in cleaned])
    empty = np.asarray(X.sum(axis=1)).ravel() == 0
    topic = lda.transform(X)
    aspects = ["Tidak teridentifikasi" if e else names[int(i)] for e, i in zip(empty, topic.argmax(1))]

    probs = []
    for s in range(0, len(reviews), batch_size):
        enc = tok(aspects[s:s + batch_size], bert_txt[s:s + batch_size], truncation=True,
                  max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    probs = np.vstack(probs)

    out = pd.DataFrame({
        "ulasan": reviews,
        "aspek": aspects,
        "sentimen": [id2label[i] for i in probs.argmax(1)],
        "confidence": probs.max(1).round(3),
    })
    for i, lab in id2label.items():
        out[f"p_{lab}"] = probs[:, i].round(3)
    return out, topic, cleaned, names


# ================================================================ KOMPONEN UI
def badge(text, color):
    return f'<span class="badge" style="background:{color}1A;color:{color}">{text}</span>'


def card_head(title, sub=""):
    st.markdown(f'<div class="card-title">{title}</div><div class="card-sub">{sub}</div>', unsafe_allow_html=True)


def style_fig(fig, height=320):
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#F3F4F6", zeroline=False)
    fig.update_yaxes(gridcolor="#F3F4F6", zeroline=False)
    return fig


def sentiment_bar(probs):
    fig = go.Figure(go.Bar(
        x=[probs[s] for s in SENT_ORDER], y=[s.capitalize() for s in SENT_ORDER], orientation="h",
        marker_color=[SENT_COLORS[s] for s in SENT_ORDER],
        text=[f"{probs[s]:.0%}" for s in SENT_ORDER], textposition="outside", cliponaxis=False,
    ))
    fig.update_xaxes(range=[0, 1.15], showticklabels=False, showgrid=False)
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 170)


# ================================================================ HEADER
st.markdown('<div class="app-title">ABSA</div>'
            '<div class="app-sub">Analisis sentimen berbasis aspek pada Google Play App Reviews aplikasi produktivitas · LDA + BERT</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Analisis", "Unggah CSV", "Dashboard", "Tentang"])

# ================================================================ TAB 1: ANALISIS
with tab1:
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            card_head("Masukkan ulasan", "Tulis atau tempel ulasan dalam bahasa Inggris.")
            text = st.text_area("Ulasan", height=200, key="review_text",
                                placeholder="Contoh: The reminders work perfectly, but the premium price is too high.",
                                label_visibility="collapsed")
            run = st.button("Analisis ulasan", type="primary", use_container_width=True)

    if run:
        if not text.strip():
            st.session_state["single"] = None
            st.warning("Ulasan tidak boleh kosong.")
        else:
            with st.spinner("Menganalisis..."):
                st.session_state["single"] = analyze([text]) + (text,)

    with right:
        with st.container(border=True):
            card_head("Hasil", "Aspek ditentukan otomatis oleh LDA, sentimen oleh BERT.")
            state = st.session_state.get("single")
            if not state:
                st.markdown('<div class="empty">Hasil akan tampil di sini setelah ulasan dianalisis.</div>',
                            unsafe_allow_html=True)
            else:
                res, topic, cleaned, names, shown_text = state
                r = res.iloc[0]
                color = SENT_COLORS[r["sentimen"]]
                if not cleaned[0]:
                    st.warning("Ulasan tidak memuat kata yang dapat dianalisis, sehingga aspek tidak dapat ditentukan.")

                c1, c2 = st.columns(2)
                c1.markdown(f'<div class="lbl">Aspek</div><div class="big">{r["aspek"]}</div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="lbl">Sentimen</div>{badge(r["sentimen"].capitalize(), color)}',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="lbl" style="margin-top:16px">Keyakinan model: {r["confidence"]:.0%}</div>',
                            unsafe_allow_html=True)
                probs = {s: float(r.get(f"p_{s}", 0)) for s in SENT_ORDER}
                st.plotly_chart(sentiment_bar(probs), use_container_width=True, config={"displayModeBar": False})

    state = st.session_state.get("single")
    if state:
        res, topic, cleaned, names, shown_text = state
        with st.expander("Lihat proses di balik hasil"):
            st.markdown("**Teks untuk BERT:** " + (preprocess_for_bert(shown_text) or "(kosong)"))
            st.markdown("**Token untuk LDA:** " + (", ".join(cleaned[0]) or "(kosong)"))
            adf = pd.DataFrame({"aspek": [names[i] for i in range(len(names))], "probabilitas": topic[0]})
            fig = px.bar(adf.sort_values("probabilitas"), x="probabilitas", y="aspek", orientation="h",
                         color_discrete_sequence=[INDIGO])
            fig.update_xaxes(range=[0, 1], title="")
            fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, 260), use_container_width=True, config={"displayModeBar": False})

# ================================================================ TAB 2: UNGGAH CSV
with tab2:
    with st.container(border=True):
        card_head("Analisis banyak ulasan", "Unggah berkas CSV, pilih kolom teks, lalu proses.")
        up = st.file_uploader("CSV", type="csv", label_visibility="collapsed")
        dfu = None
        if up is not None:
            try:
                dfu = pd.read_csv(up)
            except Exception:
                st.error("Berkas tidak dapat dibaca sebagai CSV.")
        if dfu is not None and len(dfu) > 0:
            guess = next((c for c in dfu.columns if str(c).lower() in ["content", "review", "text", "review_text"]),
                         dfu.columns[0])
            c1, c2 = st.columns(2)
            col = c1.selectbox("Kolom teks ulasan", dfu.columns, index=list(dfu.columns).index(guess))
            if len(dfu) <= 10:
                n = len(dfu)
                c2.markdown(f'<div class="lbl">Jumlah ulasan</div><div class="big">{n}</div>', unsafe_allow_html=True)
            else:
                n = c2.slider("Jumlah ulasan yang diproses", 10, min(1000, len(dfu)), min(200, len(dfu)))
            if st.button("Proses CSV", type="primary"):
                texts = dfu[col].dropna().astype(str).head(n).tolist()
                with st.spinner(f"Menganalisis {len(texts)} ulasan..."):
                    st.session_state["batch"] = analyze(texts)[0]

    if "batch" in st.session_state:
        res = st.session_state["batch"]
        k = st.columns(4)
        k[0].metric("Total ulasan", f"{len(res):,}")
        for i, s in enumerate(SENT_ORDER):
            k[i + 1].metric(s.capitalize(), f"{(res['sentimen'] == s).mean():.0%}")

        g1, g2 = st.columns(2, gap="large")
        with g1:
            with st.container(border=True):
                card_head("Sentimen per aspek")
                ct = pd.crosstab(res["aspek"], res["sentimen"]).reindex(columns=SENT_ORDER, fill_value=0)
                fig = px.bar(ct, barmode="stack", orientation="h", color_discrete_map=SENT_COLORS)
                fig.update_xaxes(title="")
                fig.update_yaxes(title="")
                st.plotly_chart(style_fig(fig, 300), use_container_width=True, config={"displayModeBar": False})
        with g2:
            with st.container(border=True):
                card_head("Hasil analisis")
                st.dataframe(res[["ulasan", "aspek", "sentimen", "confidence"]], use_container_width=True, height=300,
                             hide_index=True)
                st.download_button("Unduh hasil (CSV)", res.to_csv(index=False).encode("utf-8"),
                                   "hasil_absa.csv", "text/csv")

# ================================================================ TAB 3: DASHBOARD
with tab3:
    try:
        df = load_df()
        art = load_models()[0]
    except Exception as e:
        st.error(f"Dashboard tidak dapat dimuat: {e}")
        st.stop()

    top_sent = df["sentiment"].value_counts().idxmax()
    k = st.columns(3)
    k[0].metric("Jumlah ulasan", f"{len(df):,}")
    k[1].metric("Jumlah aspek", df["aspect"].nunique())
    k[2].metric("Sentimen terbanyak", top_sent.capitalize())

    g1, g2 = st.columns(2, gap="large")
    with g1:
        with st.container(border=True):
            card_head("Aspek yang paling banyak dibahas")
            ac = df["aspect"].value_counts().sort_values().reset_index()
            ac.columns = ["aspek", "jumlah"]
            fig = px.bar(ac, x="jumlah", y="aspek", orientation="h", color_discrete_sequence=[INDIGO])
            fig.update_xaxes(title="")
            fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, 320), use_container_width=True, config={"displayModeBar": False})
    with g2:
        with st.container(border=True):
            card_head("Proporsi sentimen")
            sc = df["sentiment"].value_counts().reindex(SENT_ORDER).reset_index()
            sc.columns = ["sentimen", "jumlah"]
            fig = px.pie(sc, names="sentimen", values="jumlah", hole=0.6, color="sentimen",
                         color_discrete_map=SENT_COLORS)
            fig.update_traces(textinfo="percent", textfont_size=13)
            st.plotly_chart(style_fig(fig, 320), use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        card_head("Sentimen pada tiap aspek", "Persentase ulasan per aspek. Aspek dengan batang merah paling banyak dikeluhkan.")
        ct = pd.crosstab(df["aspect"], df["sentiment"]).reindex(columns=SENT_ORDER, fill_value=0)
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        pct = pct.sort_values("negatif")
        fig = px.bar(pct, barmode="stack", orientation="h", color_discrete_map=SENT_COLORS, text_auto=".0f")
        fig.update_xaxes(title="", range=[0, 100])
        fig.update_yaxes(title="")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True, config={"displayModeBar": False})
        st.caption("Sentimen pada dashboard ini adalah label hasil konversi rating (1–2 negatif, 3 netral, 4–5 positif).")

    with st.expander("Kata kunci tiap aspek"):
        fn = art["vectorizer"].get_feature_names_out()
        for i, comp in enumerate(art["lda"].components_):
            words = [fn[j] for j in comp.argsort()[::-1][:10]]
            st.markdown(f"**{art['ASPECT_NAMES'][i]}**: {', '.join(words)}")

# ================================================================ TAB 4: TENTANG
with tab4:
    with st.container(border=True):
        card_head("Cara kerja sistem")
        s = st.columns(4)
        steps = [
            ("1", "Preprocessing", "Huruf kecil, buang simbol dan URL, stopword, lemmatisasi."),
            ("2", "Ekstraksi aspek", "Bag-of-Words dan LDA menentukan aspek dominan ulasan."),
            ("3", "Klasifikasi sentimen", "BERT membaca pasangan aspek dan ulasan."),
            ("4", "Hasil", "Aspek, sentimen, dan tingkat keyakinan model."),
        ]
        for col, (n, title, desc) in zip(s, steps):
            col.markdown(f'<div class="step"><div class="n">LANGKAH {n}</div><b>{title}</b><span>{desc}</span></div>',
                         unsafe_allow_html=True)

    g1, g2 = st.columns(2, gap="large")
    with g1:
        with st.container(border=True):
            card_head("Konfigurasi")
            params = pd.DataFrame({
                "Komponen": ["Model", "Kelas sentimen", "Panjang maksimum", "Batch size", "Epoch",
                             "Learning rate", "Pembagian data", "Ekstraksi aspek"],
                "Nilai": ["bert-base-uncased", "3 (rating 1–2, 3, 4–5)", "128 token", "16", "5",
                          "2e-5", "80 / 10 / 10 (berstrata)", "BoW (1–2 gram) + LDA"],
            })
            st.dataframe(params, hide_index=True, use_container_width=True)
    with g2:
        with st.container(border=True):
            card_head("Kinerja pada data uji", f"{METRICS['n_test']:,} ulasan")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{METRICS['accuracy']:.1%}")
            m2.metric("Macro F1", f"{METRICS['macro_f1']:.3f}")
            rows = [{"Kelas": c.capitalize(), "Precision": p, "Recall": r, "F1": f, "Jumlah": n}
                    for c, (p, r, f, n) in METRICS["per_class"].items()]
            st.dataframe(pd.DataFrame(rows).style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                         hide_index=True, use_container_width=True)

    with st.container(border=True):
        card_head("Confusion matrix", "Baris = label sebenarnya, kolom = prediksi, nilai = persentase per baris.")
        cm = np.array(METRICS["confusion"], dtype=float)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        labels = [s.capitalize() for s in SENT_ORDER]
        fig = go.Figure(go.Heatmap(
            z=cm_pct, x=labels, y=labels, colorscale=[[0, "#EEF2FF"], [1, INDIGO]], showscale=False,
            text=[[f"{cm_pct[i][j]:.0f}%<br>({int(cm[i][j])})" for j in range(3)] for i in range(3)],
            texttemplate="%{text}", textfont=dict(size=13),
        ))
        fig.update_yaxes(autorange="reversed", title="Sebenarnya")
        fig.update_xaxes(title="Prediksi")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True, config={"displayModeBar": False})
        st.caption("Kelas netral paling sulit karena label berasal dari rating 3, yang sering memuat pujian atau keluhan jelas.")