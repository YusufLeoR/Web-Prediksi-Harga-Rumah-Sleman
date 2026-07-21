import warnings
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Prediksi Harga Properti Sleman",
    page_icon="🏠",
    layout="wide"
)

def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #fce4ec 50%, #fff9c4 100%);
        background-attachment: fixed;
    }
    .stApp, .stMarkdown, p, span, label, div { color: #1a237e; }
    h1, h2, h3, h4 { color: #00695c !important; font-weight: 700 !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #e1f5fe 100%);
        border-right: 2px solid #b2ebf2;
    }
    section[data-testid="stSidebar"] * { color: #01579b; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f1f8e9 100%);
        border: 2px solid #aed581;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetric"] label { color: #2e7d32 !important; font-weight: 600 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #1b5e20 !important; }
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        border-radius: 12px;
        border: none;
        font-weight: 600;
        color: #ffffff;
        background: linear-gradient(90deg, #ff8a65 0%, #ffb74d 100%);
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
        background: linear-gradient(90deg, #26c6da 0%, #66bb6a 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.85);
        border: 2px solid #b3e5fc;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
    }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1a237e !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #b2dfdb;
        border-radius: 12px;
    }
    div[data-testid="stAlert"] { border-radius: 12px; }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #26c6da 0%, #66bb6a 50%, #ffca28 100%);
    }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

def format_rupiah(x: float) -> str:
    """Format angka menjadi string Rupiah."""
    return f"Rp {x:,.0f}".replace(",", ".")


def kategori_harga(harga: float) -> str:
    """Menentukan kategori harga properti."""
    if harga < 500_000_000:
        return "🟢 Ekonomis"
    elif harga < 1_500_000_000:
        return "🟡 Menengah"
    else:
        return "🔴 Premium"


def get_kecamatan_defaults(kecamatan: str, df_defaults: pd.DataFrame) -> dict:
    row = df_defaults[df_defaults["kecamatan"] == kecamatan]
    if row.empty:
        raise ValueError(f"Data default untuk kecamatan '{kecamatan}' tidak ditemukan.")
    return row.iloc[0].to_dict()


def engineer_features(luas_tanah: float, luas_bangunan: float,
                      kamar_tidur: int, kamar_mandi: int,
                      garasi: int, carport: int) -> dict:
    lb = luas_bangunan if luas_bangunan > 0 else 1
    return {
        "area_rooms_interaction":   luas_bangunan * (kamar_tidur + kamar_mandi),
        "area_parking_interaction": luas_tanah * (garasi + carport),
        "land_to_building_ratio":   luas_tanah / lb,
        "bathroom_density":         kamar_mandi / lb,
        "bedroom_density":          kamar_tidur / lb,
    }


@st.cache_resource
def load_model():
    import xgboost as xgb
    pipeline = joblib.load("model_prediksi_harga_sleman_audit_v3.pkl")
    new_xgb = xgb.XGBRegressor()
    new_xgb.load_model("booster_sleman.json")
    pipeline.steps[-1] = ("model", new_xgb)
    return pipeline


@st.cache_data
def load_kecamatan_defaults():
    return pd.read_csv("sample_per_kecamatan.csv")

def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "sample_clicked" not in st.session_state:
        st.session_state.sample_clicked = False

def render_sidebar():
    with st.sidebar:
        st.title("ℹ️ Informasi Model")
        st.markdown("---")
        st.markdown("**🤖 Algoritma**")
        st.info("XGBoost")
        st.markdown("**⚙️ Framework**")
        st.info("scikit-learn Pipeline")
        st.markdown("**📊 Jumlah Fitur Input Model**")
        st.info("16 Fitur (setelah seleksi & rekayasa)")
        st.markdown("**🗺️ Dataset**")
        st.info("Properti Kabupaten Sleman")
        st.markdown("---")
        st.markdown("**✅ Fitur Pipeline:**")
        st.markdown("""
        - Auto-fill spasial berdasarkan kecamatan
        - Rekayasa fitur otomatis di backend
        - Preprocessing terintegrasi (Imputer, Scaler, Encoder)
        - Input mentah langsung diprediksi
        """)
        st.markdown("---")
        st.markdown("**📋 Input dari Pengguna:**")
        for f in ["Luas Tanah", "Luas Bangunan", "Kamar Tidur",
                  "Kamar Mandi", "Garasi", "Carport", "Kecamatan"]:
            st.markdown(f"- {f}")
        st.markdown("---")
        st.caption("© 2025 Prediksi Harga Properti Sleman")

def render_input_form(df_defaults: pd.DataFrame):
    """Menampilkan form input prediksi."""
    kecamatan_list = sorted(df_defaults["kecamatan"].dropna().unique().tolist())

    if st.button("📋 Gunakan Contoh Data", type="secondary"):
        st.session_state.sample_clicked = True

    if st.session_state.sample_clicked:
        d_lt, d_lb, d_kt, d_km, d_g, d_c = 150, 120, 3, 2, 1, 1
        d_kec = "Depok"
    else:
        d_lt, d_lb, d_kt, d_km, d_g, d_c = 100, 80, 3, 2, 0, 1
        d_kec = kecamatan_list[0]

    kec_index = kecamatan_list.index(d_kec) if d_kec in kecamatan_list else 0

    with st.form("form_prediksi"):
        st.subheader("📝 Masukkan Data Properti")
        col1, col2, col3 = st.columns(3)
        with col1:
            luas_tanah = st.number_input(
                "Luas Tanah (m²)", min_value=10, max_value=10000,
                value=d_lt, step=1,
                help="Luas tanah properti dalam meter persegi"
            )
            luas_bangunan = st.number_input(
                "Luas Bangunan (m²)", min_value=10, max_value=10000,
                value=d_lb, step=1,
                help="Luas bangunan properti dalam meter persegi"
            )
        with col2:
            kamar_tidur = st.number_input(
                "Jumlah Kamar Tidur", min_value=1, max_value=20,
                value=d_kt, step=1
            )
            kamar_mandi = st.number_input(
                "Jumlah Kamar Mandi", min_value=1, max_value=10,
                value=d_km, step=1
            )
        with col3:
            garasi = st.number_input(
                "Jumlah Garasi", min_value=0, max_value=10,
                value=d_g, step=1
            )
            carport = st.number_input(
                "Jumlah Carport", min_value=0, max_value=10,
                value=d_c, step=1
            )

        kecamatan = st.selectbox(
            "🗺️ Kecamatan",
            options=kecamatan_list,
            index=kec_index,
            help="Pilih kecamatan. Data spasial akan diisi otomatis."
        )
        st.caption("💡 Data spasial dan fitur rekayasa diisi otomatis di backend.")
        submitted = st.form_submit_button(
            "🔍 Prediksi Harga", type="primary", use_container_width=True
        )

    return submitted, luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, garasi, carport, kecamatan


def run_prediction(model, df_defaults,
                   luas_tanah, luas_bangunan,
                   kamar_tidur, kamar_mandi,
                   garasi, carport, kecamatan):
    defaults = get_kecamatan_defaults(kecamatan, df_defaults)
    engineered = engineer_features(
        luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi, garasi, carport
    )
    input_data = {
        "luas_tanah":    float(luas_tanah),
        "luas_bangunan": float(luas_bangunan),
        "kamar_tidur":   float(kamar_tidur),
        "kamar_mandi":   float(kamar_mandi),
        "garasi":        float(garasi),
        "carport":       float(carport),
        "kecamatan":     kecamatan,
        **{k: float(v) for k, v in engineered.items()},
        "dummy_edu_hub":   float(defaults["dummy_edu_hub"]),
        "count_univ_10km": float(defaults["count_univ_10km"]),
        "dist_UGM":        float(defaults["dist_UGM"]),
        "dist_UII":        float(defaults["dist_UII"]),
        "dist_UMY":        float(defaults["dist_UMY"]),
    }
    df_input = pd.DataFrame([input_data])
    raw = float(model.predict(df_input)[0])
    prediksi = max(0.0, raw)
    return prediksi

def run_prediction_monotonic(model, df_defaults,
                             luas_tanah, luas_bangunan,
                             kamar_tidur, kamar_mandi,
                             garasi, carport, kecamatan):
    lt_grid = sorted({*range(50,  int(luas_tanah)   + 1, 25), int(luas_tanah)})
    lb_grid = sorted({*range(30,  int(luas_bangunan)+ 1, 25), int(luas_bangunan)})
    kt_grid = range(1, int(kamar_tidur) + 1)
    km_grid = range(1, int(kamar_mandi) + 1)

    best = 0.0
    for lt_i in lt_grid:
        for lb_i in lb_grid:
            if lb_i > lt_i:          
                continue
            for kt_i in kt_grid:
                for km_i in km_grid:
                    p = run_prediction(model, df_defaults,
                                       lt_i, lb_i, kt_i, km_i,
                                       garasi, carport, kecamatan)
                    if p > best:
                        best = p
    return best

def render_results(prediksi: float, luas_tanah: int, kecamatan: str,
                   luas_bangunan: int, kamar_tidur: int, kamar_mandi: int,
                   garasi: int, carport: int):
    harga_per_m2 = prediksi / luas_tanah if luas_tanah > 0 else 0
    kategori = kategori_harga(prediksi)

    st.success("✅ Prediksi berhasil!")
    st.markdown("---")
    st.subheader("Hasil Prediksi")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimasi Harga Properti", format_rupiah(prediksi))
    with col2:
        st.metric("Harga per m²", format_rupiah(harga_per_m2))
    with col3:
        st.metric("Kategori Harga", kategori)

    st.markdown("**Skala Harga (0 - Rp 5 Miliar)**")
    progress_value = min(prediksi / 5_000_000_000, 1.0)
    st.progress(progress_value)

    with st.expander("📋 Ringkasan Input"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Kecamatan:** {kecamatan}")
            st.markdown(f"**Luas Tanah:** {luas_tanah} m²")
            st.markdown(f"**Luas Bangunan:** {luas_bangunan} m²")
        with col_b:
            st.markdown(f"**Kamar Tidur:** {kamar_tidur}")
            st.markdown(f"**Kamar Mandi:** {kamar_mandi}")
            st.markdown(f"**Garasi:** {garasi} | **Carport:** {carport}")

    st.session_state.history.append({
        "Waktu":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Kecamatan":          kecamatan,
        "Luas Tanah (m²)":    luas_tanah,
        "Luas Bangunan (m²)": luas_bangunan,
        "Harga Prediksi":     format_rupiah(prediksi),
        "Kategori":           kategori,
    })


def render_history():
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Riwayat Prediksi")
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)

        csv_data = df_history.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Unduh Riwayat sebagai CSV",
            data=csv_data,
            file_name=f"riwayat_prediksi_sleman_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="secondary"
        )
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()


def render_info_expanders():
    st.markdown("---")
    with st.expander("🤖 Tentang Model"):
        st.markdown("""
        **Model:** XGBoost  
        **Framework:** scikit-learn Pipeline  
        **Fitur yang digunakan model:** 16 (setelah seleksi fitur)

        Pipeline mencakup preprocessing lengkap:
        - **SimpleImputer** – mengisi nilai yang hilang
        - **RobustScaler** – normalisasi fitur numerik
        - **OneHotEncoder** – encoding fitur kategorikal (kecamatan)
        - **ColumnTransformer** – menggabungkan semua transformer

        **Fitur rekayasa yang dihitung otomatis:**
        - `area_rooms_interaction` = luas_bangunan × (kamar_tidur + kamar_mandi)
        - `area_parking_interaction` = luas_tanah × (garasi + carport)
        - `land_to_building_ratio` = luas_tanah ÷ luas_bangunan
        - `bathroom_density` = kamar_mandi ÷ luas_bangunan
        - `bedroom_density` = kamar_tidur ÷ luas_bangunan
        """)

    with st.expander("📖 Cara Menggunakan"):
        st.markdown("""
        1. **Isi form** dengan data properti yang ingin diprediksi.
        2. **Pilih kecamatan** — data spasial dan fitur rekayasa diisi otomatis di backend.
        3. Klik **\"Prediksi Harga\"** untuk mendapatkan estimasi.
        4. Lihat hasil estimasi harga, harga per m², dan kategori properti.
        5. Gunakan tombol **\"Gunakan Contoh Data\"** untuk mencoba dengan data sampel.
        6. Riwayat prediksi tersimpan selama sesi aktif dan bisa **diunduh sebagai CSV**.
        """)

def main():
    inject_custom_css()
    init_session_state()
    render_sidebar()

    st.title("🏠 Prediksi Harga Properti Kabupaten Sleman")
    st.markdown(
        "Aplikasi ini menggunakan model **XGBoost** untuk memprediksi harga properti "
        "berdasarkan karakteristik fisik, lokasi, dan kedekatan terhadap perguruan tinggi."
    )
    st.markdown("---")

    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

    try:
        df_defaults = load_kecamatan_defaults()
    except Exception as e:
        st.error(f"❌ Gagal membaca data kecamatan: {e}")
        st.stop()

    (submitted, luas_tanah, luas_bangunan,
     kamar_tidur, kamar_mandi, garasi, carport, kecamatan) = render_input_form(df_defaults)

    if submitted:
        try:
            with st.spinner("⏳ Memproses prediksi..."):
                prediksi = run_prediction_monotonic(
                    model, df_defaults,
                    luas_tanah, luas_bangunan,
                    kamar_tidur, kamar_mandi,
                    garasi, carport, kecamatan
                )
            render_results(prediksi, luas_tanah, kecamatan,
                           luas_bangunan, kamar_tidur, kamar_mandi, garasi, carport)
        except ValueError as ve:
            st.error(f"❌ {ve}")
        except Exception:
            st.error("❌ Prediksi gagal. Detail error:")
            st.exception(traceback.format_exc())

    render_history()
    render_info_expanders()


if __name__ == "__main__":
    main()