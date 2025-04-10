# Analyseur de Dommages Auto Pro

Une application web moderne pour estimer les coûts de réparation automobile en utilisant l'IA.

## Fonctionnalités

- 📝 Saisie manuelle des détails du véhicule
- 📸 Analyse automatique des photos de dommages
- 💰 Estimation précise des coûts de réparation
- 📊 Détail des coûts (pièces, main d'œuvre, peinture)
- ⏱️ Calendrier de réparation

## Installation Locale

1. Clonez le dépôt :
```bash
git clone [url-du-repo]
cd [nom-du-dossier]
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurez votre clé API Gemini :
   - Copiez le fichier `.env.example` vers `.env`
   - Remplacez `your_api_key_here` par votre clé API Gemini

4. Lancez l'application :
```bash
streamlit run app.py
```

## Déploiement sur Streamlit Cloud

1. Créez un compte sur [Streamlit Cloud](https://streamlit.io/cloud)

2. Connectez votre compte GitHub

3. Cliquez sur "New app"

4. Sélectionnez votre dépôt et le fichier `app.py`

5. Dans les paramètres de déploiement :
   - Ajoutez votre clé API Gemini comme secret :
     - Clé : `GEMINI_API_KEY`
     - Valeur : Votre clé API Gemini

6. Cliquez sur "Deploy"

## Structure du Projet

- `app.py` : Application principale
- `gradient_boosting_model.pkl` : Modèle de prédiction
- `preprocessor.pkl` : Prétraitement des données
- `requirements.txt` : Dépendances Python
- `.env` : Configuration (clé API)

## Utilisation

1. **Saisie Manuelle** :
   - Sélectionnez la marque et le modèle
   - Entrez l'année
   - Choisissez la partie endommagée
   - Indiquez la sévérité

2. **Analyse par Photo** :
   - Téléchargez une photo du dommage
   - L'IA analysera automatiquement l'image
   - Obtenez une estimation détaillée

## Support

Pour toute question ou problème, veuillez ouvrir une issue sur GitHub. 