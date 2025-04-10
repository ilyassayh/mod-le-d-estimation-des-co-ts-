import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
from PIL import Image
import json
import re
from datetime import datetime
import base64
import os
from dotenv import load_dotenv

# Try to import plotly, but don't fail if it's not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# === SET PAGE CONFIG FIRST ===
st.set_page_config(
    page_title="Estimateur de Coût de Réparation Auto Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CONFIG ===
# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDzVhetJLHiv357wc1X-0XgbbfYktqHO-c")  # Fallback for development
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# === LOAD MODEL ===
@st.cache_resource
def load_models():
    try:
        best_model = joblib.load("gradient_boosting_model.pkl")
        preprocessor = joblib.load("preprocessor.pkl")
        return best_model, preprocessor
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        st.error("Veuillez vérifier que les fichiers 'gradient_boosting_model.pkl' et 'preprocessor.pkl' existent dans le répertoire.")
        return None, None

best_model, preprocessor = load_models()

# Check if models are loaded successfully
if best_model is None or preprocessor is None:
    st.error("L'application ne peut pas fonctionner sans les modèles. Veuillez contacter l'administrateur.")
    st.stop()

# === OPTIONS ===
brands = ['Dacia', 'Renault', 'Peugeot', 'Hyundai', 'Volkswagen', 'Toyota', 'Fiat', 'Ford', 'Mercedes', 'BMW']
models = {
    'Dacia': ['Logan', 'Sandero', 'Duster'],
    'Renault': ['Clio', 'Megane', 'Kangoo'],
    'Peugeot': ['208', '301', '308'],
    'Hyundai': ['i10', 'i20', 'Tucson'],
    'Volkswagen': ['Polo', 'Golf', 'Passat'],
    'Toyota': ['Yaris', 'Corolla', 'RAV4'],
    'Fiat': ['Panda', 'Tipo', '500'],
    'Ford': ['Fiesta', 'Focus', 'EcoSport'],
    'Mercedes': ['C-Class', 'E-Class', 'A-Class'],
    'BMW': ['Series 1', 'Series 3', 'X1']
}
car_parts = [
    'Porte arriere gauche', 'Aile arriere droit', 'Aile arriere gauche', 'Pare-brise arriere',
    'Malle', 'Feu arriere droit', 'Feu arriere gauche', 'Pare-choc arriere', 'Plaque immatriculation arriere',
    'Pare-choc avant', 'Capot', 'Grille', 'Phare avant gauche', 'Phare avant droit'
]
car_parts_en = [
    'Left rear door', 'Right rear wing', 'Left rear wing', 'Rear windshield',
    'Trunk', 'Right rear light', 'Left rear light', 'Rear bumper', 'Rear license plate',
    'Front bumper', 'Hood', 'Grille', 'Left headlight', 'Right headlight'
]
car_parts_map = dict(zip(car_parts, car_parts_en))

severities = {1: 'Minor (Scratches)', 2: 'Moderate (Dents)', 3: 'Severe (Cracks, Breaks)'}
severity_descriptions = {
    1: "Surface scratches, minor scuffs, small paint chips",
    2: "Visible dents, partial component damage, deeper scratches",
    3: "Structural damage, broken components, deep cracks requiring complete replacement"
}

# === UTILITY FUNCTIONS ===
def get_image_download_link(img_data, filename, text):
    b64 = base64.b64encode(img_data).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def format_currency(amount):
    return f"{amount:,.2f} MAD"

def get_cost_category(cost):
    if cost < 1000:
        return "Low", "🟢"
    elif cost < 3000:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

# === GEMINI ANALYSIS ===
def analyze_image_with_gemini(image):
    prompt = """
    Please analyze this car image and extract the following:
    - Car brand and model (e.g., Dacia Sandero)
    - Estimated year of manufacture
    - Damaged part (choose one from: Left rear door, Right rear wing, Left rear wing, Rear windshield,
    Trunk, Right rear light, Left rear light, Rear bumper, Rear license plate,
    Front bumper, Hood, Grille, Left headlight, Right headlight)
    - Severity of the damage (minor, moderate, severe)
    - Brief description of the damage (e.g., "Deep scratches along the front bumper with visible paint damage")
    - Estimated repair cost in MAD (Moroccan Dirham)

    Respond with JSON like:
    {
        "brand": "...",
        "model": "...",
        "year": 2020,
        "damaged_part": "...",
        "severity": "...",
        "damage_description": "...",
        "estimated_cost": 2500
    }
    """
    response = model.generate_content([prompt, image])
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            for fr_part, en_part in car_parts_map.items():
                if result.get("damaged_part", "").lower() in en_part.lower():
                    result["damaged_part"] = fr_part
                    break
            return result
        except Exception as e:
            st.error(f"Error parsing Gemini response: {e}")
            return None
    return None

# === CUSTOM CSS ===
st.markdown("""
<style>
    /* Modern UI Theme */
    .stApp {
        background-color: #f8f9fa;
        color: #2c3e50;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
        text-align: center;
        padding: 20px 0;
    }
    
    .card {
        background-color: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border: 1px solid #e5e7eb;
        transition: transform 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    .highlight {
        background-color: #f8fafc;
        padding: 20px;
        border-radius: 15px;
        border-left: 4px solid #3B82F6;
        color: #2c3e50;
        margin: 15px 0;
    }
    
    .cost-high {
        color: #DC2626;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .cost-medium {
        color: #F59E0B;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .cost-low {
        color: #10B981;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #1E3A8A;
        padding-bottom: 10px;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .info-box {
        background-color: #f8fafc;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px;
        color: #6B7280;
        font-size: 1.1rem;
        font-weight: 500;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
    
    .stButton>button {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 500;
        background: linear-gradient(120deg, #1E3A8A, #3B82F6);
        border: none;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stSelectbox>div>div {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
    
    .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
    
    .upload-box {
        border: 2px dashed #3B82F6;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        background-color: #f8fafc;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        background-color: #f0f7ff;
        border-color: #1E3A8A;
    }
    
    .cost-breakdown {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin: 20px 0;
    }
    
    .cost-item {
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        color: white;
        transition: transform 0.2s ease;
    }
    
    .cost-item:hover {
        transform: translateY(-3px);
    }
    
    .cost-item.parts {
        background: linear-gradient(135deg, #3B82F6, #1E3A8A);
    }
    
    .cost-item.labor {
        background: linear-gradient(135deg, #10B981, #059669);
    }
    
    .cost-item.painting {
        background: linear-gradient(135deg, #F59E0B, #D97706);
    }
    
    .timeline {
        position: relative;
        max-width: 1200px;
        margin: 40px auto;
        padding: 20px 0;
    }
    
    .timeline::after {
        content: '';
        position: absolute;
        width: 4px;
        background: linear-gradient(to bottom, #3B82F6, #1E3A8A);
        top: 0;
        bottom: 0;
        left: 50%;
        margin-left: -2px;
    }
    
    .timeline-item {
        padding: 10px 40px;
        position: relative;
        width: 50%;
        box-sizing: border-box;
    }
    
    .timeline-item::after {
        content: '';
        position: absolute;
        width: 25px;
        height: 25px;
        right: -17px;
        background-color: white;
        border: 4px solid #3B82F6;
        top: 15px;
        border-radius: 50%;
        z-index: 1;
        box-shadow: 0 0 0 4px #f8fafc;
    }
    
    .timeline-content {
        padding: 25px 30px;
        background-color: white;
        position: relative;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .timeline-content:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
with st.sidebar:
    st.title("⚙️ Paramètres")
    theme = st.radio("Thème", ["Clair", "Sombre"], key="theme")
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    if 'history' in st.session_state and st.session_state.history:
        total_estimates = len(st.session_state.history)
        avg_cost = sum(entry['cost'] for entry in st.session_state.history) / total_estimates
        st.metric("Total des Estimations", total_estimates)
        st.metric("Coût Moyen", format_currency(avg_cost))
        
        # Cost distribution
        st.markdown("#### Distribution des Coûts")
        low_cost = sum(1 for entry in st.session_state.history if entry['cost'] < 1000)
        medium_cost = sum(1 for entry in st.session_state.history if entry['cost'] < 3000 and entry['cost'] >= 1000)
        high_cost = sum(1 for entry in st.session_state.history if entry['cost'] >= 3000)
        
        st.markdown(f"🟢 Coût Faible: {low_cost}")
        st.markdown(f"🟡 Coût Moyen: {medium_cost}")
        st.markdown(f"🔴 Coût Élevé: {high_cost}")
    else:
        st.info("Aucune estimation pour le moment")
    
    st.markdown("---")
    st.markdown("### 📱 À Propos")
    st.markdown("""
    Cette application utilise l'IA pour analyser les dommages de voiture et estimer les coûts de réparation.
    
    **Fonctionnalités:**
    - 🖼️ Analyse d'Image
    - 💰 Estimation des Coûts
    - 📊 Détail des Coûts
    - 📈 Comparaison Historique
    """)

# === MAIN UI ===
st.markdown("<div class='main-header'>🚗 Analyseur de Dommages Auto Pro</div>", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize default values
default_brand = brands[0]
default_model = models[default_brand][0]  # This will be the first model of the first brand
default_year = 2020
default_damaged_part = car_parts[0]
default_severity = 1

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📝 Saisie Manuelle", "📸 Analyse par Photo"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🚘 Détails du Véhicule</div>", unsafe_allow_html=True)
    
    # Vehicle Selection
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Marque de Voiture", brands, index=brands.index(default_brand))
        # Get the first model of the selected brand as default
        current_default_model = models[brand][0]
        model_name = st.selectbox("Modèle de Voiture", models[brand], index=0)
        year = st.number_input("Année de la Voiture", min_value=2005, max_value=2024, value=default_year)

    with col2:
        damaged_part = st.selectbox(
            "Partie Endommagée",
            car_parts,
            index=car_parts.index(default_damaged_part),
            format_func=lambda x: f"{x} ({car_parts_map.get(x, x)})"
        )
        severity = st.selectbox(
            "Sévérité des Dommages",
            options=list(severities.keys()),
            format_func=lambda x: severities[x],
            index=default_severity - 1
        )

    st.markdown(f"<div class='info-box'>{severity_descriptions[severity]}</div>", unsafe_allow_html=True)

    if st.button("Calculer le Coût de Réparation", key="calculate_btn", use_container_width=True):
        with st.spinner("Calcul du coût de réparation..."):
            input_data = pd.DataFrame([{
                'Brand': brand,
                'Model': model_name,
                'Year': year,
                'Damaged_Part': damaged_part,
                'Damage_Severity': severity
            }])

            try:
                input_encoded = preprocessor.transform(input_data)
                predicted_cost = best_model.predict(input_encoded)[0]

                # Cost breakdown
                parts_cost = predicted_cost * (0.4 if severity == 1 else 0.5 if severity == 2 else 0.6)
                labor_hours = 2 if severity == 1 else 4 if severity == 2 else 7
                labor_cost = labor_hours * 150
                painting_cost = predicted_cost - parts_cost - labor_cost

                # Display predictions
                st.markdown("### 💰 Estimation des Coûts")
                cost_category, emoji = get_cost_category(predicted_cost)
                cost_class = "cost-low" if predicted_cost < 1000 else "cost-medium" if predicted_cost < 3000 else "cost-high"
                st.markdown(f"<div class='highlight'><h2 class='{cost_class}'>{emoji} {format_currency(predicted_cost)}</h2></div>", unsafe_allow_html=True)

                # Cost breakdown visualization
                st.markdown("""
                <div class='cost-breakdown'>
                    <div class='cost-item parts'>
                        <h3>Pièces</h3>
                        <p>{}</p>
                    </div>
                    <div class='cost-item labor'>
                        <h3>Main d'œuvre</h3>
                        <p>{}</p>
                    </div>
                    <div class='cost-item painting'>
                        <h3>Peinture</h3>
                        <p>{}</p>
                    </div>
                </div>
                """.format(
                    format_currency(parts_cost),
                    format_currency(labor_cost),
                    format_currency(painting_cost)
                ), unsafe_allow_html=True)

                # Repair Timeline
                st.markdown("<div class='section-header'>⏱️ Calendrier de Réparation</div>", unsafe_allow_html=True)
                st.markdown("""
                <div class="timeline">
                    <div class="timeline-item left">
                        <div class="timeline-content">
                            <h4>Évaluation Initiale</h4>
                            <p>1-2 heures</p>
                        </div>
                    </div>
                    <div class="timeline-item right">
                        <div class="timeline-content">
                            <h4>Commande des Pièces</h4>
                            <p>1-2 jours</p>
                        </div>
                    </div>
                    <div class="timeline-item left">
                        <div class="timeline-content">
                            <h4>Travaux de Réparation</h4>
                            <p>{} heures</p>
                        </div>
                    </div>
                    <div class="timeline-item right">
                        <div class="timeline-content">
                            <h4>Inspection Finale</h4>
                            <p>1-2 heures</p>
                        </div>
                    </div>
                </div>
                """.format(labor_hours), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erreur lors du calcul du coût: {e}")
    else:
        st.info("Remplissez les détails et cliquez sur 'Calculer le Coût de Réparation' pour obtenir une estimation")

    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📸 Analyse d'Image</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='upload-box'>
        <h3>Déposez votre image ici</h3>
        <p>ou cliquez pour sélectionner un fichier</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Image téléchargée", use_container_width=True)
        with st.spinner("🔍 Analyse de l'image en cours..."):
            result = analyze_image_with_gemini(image)

        if result:
            st.success("✅ Image analysée avec succès!")
            with st.expander("🤖 Résultats de l'Analyse IA", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Marque:** {result.get('brand', 'Inconnue')}")
                    st.markdown(f"**Modèle:** {result.get('model', 'Inconnu')}")
                    st.markdown(f"**Année:** {result.get('year', 'Inconnue')}")
                with col2:
                    st.markdown(f"**Partie:** {car_parts_map.get(result.get('damaged_part', ''), result.get('damaged_part', 'Inconnue'))}")
                    st.markdown(f"**Sévérité:** {result.get('severity', 'Inconnue').capitalize()}")
                    st.markdown(f"**Description:** {result.get('damage_description', 'Non disponible')}")

            if 'estimated_cost' in result:
                st.markdown("### 💰 Estimation des Coûts")
                cost_category, emoji = get_cost_category(result['estimated_cost'])
                cost_class = "cost-low" if result['estimated_cost'] < 1000 else "cost-medium" if result['estimated_cost'] < 3000 else "cost-high"
                st.markdown(f"<div class='highlight'><h2 class='{cost_class}'>{emoji} {format_currency(result['estimated_cost'])}</h2></div>", unsafe_allow_html=True)
        else:
            st.warning("⚠ Impossible d'analyser l'image correctement. Veuillez essayer une autre image ou remplir les détails manuellement.")
    else:
        st.info("Téléchargez une image pour commencer l'analyse")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: #6B7280; margin-top: 30px; padding: 20px; border-top: 1px solid #E5E7EB;'>
    <p>Analyseur de Dommages Auto Pro v2.0 | Propulsé par IA</p>
    <p style='font-size: 0.8em;'>Dernière mise à jour: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
