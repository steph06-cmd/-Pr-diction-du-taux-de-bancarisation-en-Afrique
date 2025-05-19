import streamlit as st
import joblib
import os
import pandas as pd

# ===================== Chargement du modèle =======================
@st.cache_resource
def load_model():
    model_path = "decision_tree_model.joblib"
    if not os.path.exists(model_path):
        st.error(f"❌ Le fichier {model_path} est introuvable.")
        return None
    return joblib.load(model_path)

model = load_model()

# ===================== Titre =======================
st.title("🔍 Prédiction du taux de bancarisation en Afrique")

if model is not None:
    # ===================== Saisie utilisateur =======================
    st.subheader("Veuillez renseigner les informations suivantes :")

    # Dictionnaires d'encodage
    country_map = {
        'Kenya': 0,
        'Rwanda': 1,
        'Tanzania': 2,
        'Uganda': 3
    }

    location_map = {
        'Rural': 0,
        'Urban': 1
    }

    cellphone_map = {
        'Yes': 1,
        'No': 0
    }

    gender_map = {
        'Female': 0,
        'Male': 1
    }

    relationship_map = {
        'Head of Household': 0,
        'Spouse': 1,
        'Child': 2,
        'Parent': 3,
        'Other relative': 4,
        'Other non-relatives': 5
    }

    marital_status_map = {
        'Married/Living together': 0,
        'Widowed': 1,
        'Single/Never Married': 2,
        'Divorced/Seperated': 3,
        'Dont know': 4
    }

    education_map = {
        'No formal education': 0,
        'Primary education': 1,
        'Secondary education': 2,
        'Vocational/Specialised training': 3,
        'Tertiary education': 4,
        'Other/Dont know/RTA': 5
    }

    job_type_map = {
        'Self employed': 0,
        'Government Dependent': 1,
        'Formally employed Private': 2,
        'Informally employed': 3,
        'Formally employed Government': 4,
        'Farming and Fishing': 5,
        'Remittance Dependent': 6,
        'Other Income': 7,
        'Dont Know/Refuse to answer': 8,
        'No Income': 9
    }

    # Interface utilisateur
    country_choice = st.selectbox("Pays", list(country_map.keys()))
    location_choice = st.selectbox("Type de lieu", list(location_map.keys()))
    cellphone_choice = st.selectbox("Accès à un téléphone portable", list(cellphone_map.keys()))
    gender_choice = st.selectbox("Genre", list(gender_map.keys()))
    relationship_choice = st.selectbox("Lien avec le chef de ménage", list(relationship_map.keys()))
    marital_status_choice = st.selectbox("Statut marital", list(marital_status_map.keys()))
    education_choice = st.selectbox("Niveau d’éducation", list(education_map.keys()))
    job_type_choice = st.selectbox("Type d'emploi", list(job_type_map.keys()))
    household_size = st.number_input("Taille du ménage", min_value=1)
    age = st.number_input("Âge du répondant", min_value=10)
    year = st.number_input("Année de l'enquête", min_value=2000, max_value=2100, value=2024)

    # Encodage numérique
    country = country_map[country_choice]
    location_type = location_map[location_choice]
    cellphone_access = cellphone_map[cellphone_choice]
    gender = gender_map[gender_choice]
    relationship = relationship_map[relationship_choice]
    marital_status = marital_status_map[marital_status_choice]
    education = education_map[education_choice]
    job_type = job_type_map[job_type_choice]

    # ===================== Création du DataFrame =======================
    input_df = pd.DataFrame([{
        "country": country,
        "year": year,
        "location_type": location_type,
        "cellphone_access": cellphone_access,
        "household_size": household_size,
        "age_of_respondent": age,
        "gender_of_respondent": gender,
        "relationship_with_head": relationship,
        "marital_status": marital_status,
        "education_level": education,
        "job_type": job_type
    }])

    # ===================== Encodage des variables (optionnel) =======================
    encoders_path = "label_encoders.joblib"
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
        for col in input_df.select_dtypes(include='object').columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])
            else:
                st.warning(f"⚠️ Pas d'encodeur pour {col}")
    else:
        st.warning("⚠️ Le fichier label_encoders.joblib est manquant. Les variables catégoriques pourraient être incorrectes.")

    # ===================== Réorganisation des colonnes =======================
    expected_order = [
        'country', 'year', 'location_type', 'cellphone_access', 'household_size',
        'age_of_respondent', 'gender_of_respondent', 'relationship_with_head',
        'marital_status', 'education_level', 'job_type'
    ]
    input_df = input_df[expected_order]

    # ===================== Affichage des données encodées =======================
    st.subheader("🧾 Données utilisateur encodées :")
    st.dataframe(input_df)

    # ===================== Prédiction =======================
    if st.button("Prédire"):
        try:
            prediction = model.predict(input_df)[0]
            if prediction == 1:
                st.success("✅ Le client a un compte bancaire.")
            else:
                st.warning("❌ Le client n’a PAS de compte bancaire.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

else:
    st.warning("⚠️ Le modèle n'a pas pu être chargé.")
