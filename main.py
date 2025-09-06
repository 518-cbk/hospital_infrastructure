import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

st.set_page_config(
    page_title="Healthcare Facility Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --error-color: #F44336;
        --background-color: #F8F9FA;
        --text-color: #2C3E50;
    }
    .css-1d391kg {
        background: linear-gradient(180deg, #2E86AB 0%, #1C5A7A 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary-color);
        transition: box-shadow 0.3s ease;
    }
    .info-card:hover {
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #E0E0E0;
        transition: border-color 0.3s ease;
    }
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(46, 134, 171, 0.2);
    }
    .stSuccess {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    .stError {
        background: linear-gradient(135deg, #F44336 0%, #d32f2f 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px 10px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .css-1lcbmhc .css-1outpf7 {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("PMC Hospital Infrastructure.csv")
        # Convert specific columns to numeric
        columns_to_convert = [
            'Number of Beds in Emergency Wards ',
            'Number of Doctors / Physicians',
            'Number of Nurses',
            'Number of Midwives Professional ',
            'Average Monthly Patient Footfall',
            'Count of Ambulance'
        ]
        for col in columns_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '')
        # Drop unnecessary columns
        for col in ['CityName', 'ZoneName', 'WardNo.', 'NumberofBedsinEmergencyWards']:
            if col in df.columns:
                df = df.drop(columns=col)
        df = df.dropna()
        df = df.drop_duplicates()

        # Standardize Yes/No columns
        for col in ['PharmacyAvailable:Yes/No', 'AmbulanceServiceAvailable']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col] = df[col].replace('n.a.', 'no')
                df[col] = df[col].map({'yes':1, 'no':0})
                
        # Facility type classification
        type_mapping = {
            'lab': 'Laboratory',
            'laboratory': 'Laboratory',
            'nursinghome': 'Nursing Home',
            'nursing home': 'Nursing Home',
            'hospital': 'General Hospital',
            'clinic': 'Clinic / Dispensary',
            'dispensary': 'Clinic / Dispensary'
        }
        
        specialty_keywords = ['ent', 'ophthalmology', 'cardiology', 'urology', 'orthopedics']
        ayurvedic_keywords = ['ayurvedic', 'homeopathic', 'ayurved']
        surgical_keywords = ['surgical']
        maternity_keywords = ['maternity']
        elderly_keywords = ['elderly', 'senior']
        
        def classify_facility(row):
            type_lower = row['Type(Hospital/NursingHome/Lab)'].lower()
            for key, category in type_mapping.items():
                if key in type_lower:
                    return category
            if any(keyword in type_lower for keyword in maternity_keywords):
                return 'Hospital & Maternity'
            elif any(keyword in type_lower for keyword in elderly_keywords):
                return 'Nursing Home'
            elif any(keyword in type_lower for keyword in specialty_keywords):
                return 'Specialty Clinic'
            elif any(keyword in type_lower for keyword in ayurvedic_keywords):
                return 'Ayurvedic / Homeopathic'
            elif any(keyword in type_lower for keyword in surgical_keywords):
                return 'Surgical / Procedural Facility'
            else:
                if 'surgical' in type_lower:
                    return 'Surgical / Procedural Facility'
                elif 'clinic' in type_lower or 'dispensary' in type_lower:
                    return 'Clinic / Dispensary'
                elif 'lab' in type_lower:
                    return 'Laboratory'
                elif 'nursing' in type_lower or 'elderly' in type_lower or 'senior' in type_lower:
                    return 'Nursing Home'
                elif 'hospital' in type_lower:
                    return 'General Hospital'
                else:
                    return 'Other'
        df['Type(Hospital/NursingHome/Lab)'] = df.apply(classify_facility, axis=1)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    models = {}
    try:
        models['regressor'] = joblib.load('best_patient_footfall_regressor.pkl')
        models['regressor_scaler'] = joblib.load('scaler_patient_footfall.pkl')
    except:
        st.warning("Regression model not found.")
    try:
        models['class_public_private'] = joblib.load('best_public_private_clf.pkl')
        models['class_scaler'] = joblib.load('scaler_public_private.pkl')
    except:
        st.warning("Public/Private classifier not found.")
    try:
        models['facility_type'] = joblib.load('best_facility_type_clf.pkl')
        models['facility_scaler'] = joblib.load('scaler_facility_type.pkl')
    except:
        st.warning("Facility Type classifier not found.")
    try:
        models['le_type'] = joblib.load('le_type.pkl')
        models['le_class'] = joblib.load('le_class.pkl')
        models['le_ward'] = joblib.load('le_ward.pkl')
        models['le_facility'] = joblib.load('le_facility.pkl')
    except:
        pass
    return models

def perform_clustering(df):
    features_clust = [
        'NumberofBedsinfacilitytype',
        'NumberofDoctors/Physicians',
        'NumberofNurses',
        'NumberofMidwivesProfessional',
        'CountofAmbulance'
    ]
    features_clust = [f for f in features_clust if f in df.columns]
    if not features_clust:
        return df, None, None

    integer_features = [
        'NumberofBedsinfacilitytype',
        'NumberofDoctors/Physicians',
        'NumberofNurses',
        'NumberofMidwivesProfessional',
        'CountofAmbulance'
    ]
    for feature in integer_features:
        if feature in df.columns:
            df[feature] = df[feature].round().astype(int)

    X_clust = df[features_clust]
    scaler_clust = StandardScaler()
    X_scaled = scaler_clust.fit_transform(X_clust)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = labels
    cluster_names = {
        0: 'Development Needs',
        1: 'Basic Infrastructure',
        2: 'Specialized Services'
    }
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)
    return df, kmeans, scaler_clust

# Load data and models once
df = load_data()
models = load_models()
if not df.empty:
    df, kmeans_model, clust_scaler = perform_clustering(df)

sidebar_pages = [
    "📊 Overview",
    "📈 Patient Footfall Prediction",
    "🏷️ Facility Classification",
    "🎯 Cluster Analysis",
    "⚠️ Anomaly Detection"
]
st.sidebar.markdown("# 🏥 Healthcare Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("🧭 Navigate", sidebar_pages)
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip**: This dashboard uses machine learning to analyze healthcare infrastructure data and provide insights for better decision making."
)

def create_metric_card(value, label, col):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# Start rendering pages
if page == "📊 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Healthcare Facility Infrastructure Analysis</h1>
        <p>Comprehensive analytics for healthcare facility management and planning</p>
    </div>
    """, unsafe_allow_html=True)
    if df.empty:
        st.error("❌ No data available. Please ensure the CSV file is in the correct location.")
        st.stop()
    col1, col2, col3, col4 = st.columns(4)
    create_metric_card(len(df), "Total Facilities", col1)
    hosp_count = len(df[df['Type(Hospital/NursingHome/Lab)'].str.contains('Hospital', case=False)]) if 'Type(Hospital/NursingHome/Lab)' in df.columns else 0
    create_metric_card(hosp_count, "Hospitals", col2)
    pub_count = len(df[df['Class:(Public/Private)'] == 'Public']) if 'Class:(Public/Private)' in df.columns else 0
    create_metric_card(pub_count, "Public Facilities", col3)
    if 'AverageMonthlyPatientFootfall' in df.columns:
        avg_footfall = df['AverageMonthlyPatientFootfall'].mean()
        create_metric_card(f"{avg_footfall:.0f}", "Avg Monthly Footfall", col4)
    else:
        create_metric_card("N/A", "Avg Monthly Footfall", col4)
    with st.container():
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📋 Data Overview")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("🏢 Facility Type Distribution")
        if 'Type(Hospital/NursingHome/Lab)' in df.columns:
            type_counts = df['Type(Hospital/NursingHome/Lab)'].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig.update_traces(
                textinfo='percent+label',
                textposition='inside',
                textfont=dict(color='white')
            )
            fig.update_layout(
                showlegend=True,
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("🏛️ Public vs Private Distribution")
        if 'Class:(Public/Private)' in df.columns:
            class_counts = df['Class:(Public/Private)'].value_counts()
            fig = px.bar(
                x=class_counts.index,
                y=class_counts.values,
                color=class_counts.index,
                color_discrete_sequence=['#667eea', '#764ba2']
            )
            fig.update_traces(
                text=None,
                textposition='outside'
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Facility Class",
                yaxis_title="Count",
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(range=[0, class_counts.max() * 1.1]),
                font=dict(size=12, color='black')
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🔥 Feature Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(color='black')
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Patient Footfall Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Patient Footfall Prediction</h1>
        <p>Predict average monthly patient footfall using facility characteristics</p>
    </div>
    """, unsafe_allow_html=True)
    models = load_models()
    df = load_data()
    if 'regressor' not in models:
        st.error("❌ Regression model not available. Please ensure the model files are in place.")
    else:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏢 Facility Information")
            if 'le_type' in models:
                facility_options = list(models['le_type'].classes_)
            else:
                facility_options = sorted(df['Type(Hospital/NursingHome/Lab)'].unique()) if 'Type(Hospital/NursingHome/Lab)' in df.columns else ['General Hospital', 'Nursing Home', 'Laboratory']
            facility_type = st.selectbox("🏥 Facility Type", options=facility_options)
            if 'le_class' in models:
                class_options = list(models['le_class'].classes_)
            else:
                class_options = sorted(df['Class:(Public/Private)'].unique()) if 'Class:(Public/Private)' in df.columns else ['Public', 'Private']
            facility_class = st.selectbox("🏛️ Facility Class", options=class_options)
            pharmacy_available = st.selectbox("💊 Pharmacy Available", options=["Yes", "No"])
            num_beds = st.number_input("🛏️ Number of Beds", min_value=0, value=50, step=1)
        with col2:
            st.subheader("👥 Staffing Information")
            num_doctors = st.number_input("👨‍⚕️ Number of Doctors", min_value=0, value=10, step=1)
            num_nurses = st.number_input("👩‍⚕️ Number of Nurses", min_value=0, value=15, step=1)
            num_midwives = st.number_input("🤱 Number of Midwives", min_value=0, value=5, step=1)
            ambulance_available = st.selectbox("🚑 Ambulance Service Available", options=["Yes", "No"])
            if ambulance_available == "Yes":
                ambulance_count = st.number_input("🚑 Ambulance Count", min_value=1, value=2, step=1, help="Must be at least 1 if ambulance service is available")
            else:
                ambulance_count = 0
                st.info("🚑 Ambulance count set to 0 since service is not available.")
        st.markdown('</div>', unsafe_allow_html=True)
        colA, colB, colC = st.columns([1, 2, 1])
        with colB:
            if st.button("🔮 Predict Patient Footfall", use_container_width=True):
                if ambulance_available == "Yes" and ambulance_count < 1:
                    st.error("❌ Ambulance count must be at least 1 when ambulance service is available")
                    st.stop()
                elif ambulance_available == "No" and ambulance_count != 0:
                    st.error("❌ Ambulance count must be 0 when ambulance service is not available")
                    st.stop()
                if 'le_type' in models:
                    encoded_type = models['le_type'].transform([facility_type])[0]
                else:
                    type_mapping = {type_name: i for i, type_name in enumerate(facility_options)}
                    encoded_type = type_mapping.get(facility_type, 0)
                if 'le_class' in models:
                    encoded_class = models['le_class'].transform([facility_class])[0]
                else:
                    encoded_class = 1 if facility_class == "Public" else 0
                pharmacy_encoded = 1 if pharmacy_available == "Yes" else 0
                ambulance_encoded = 1 if ambulance_available == "Yes" else 0
                features = np.array([[
                    encoded_type,
                    encoded_class,
                    pharmacy_encoded,
                    num_beds,
                    num_doctors,
                    num_nurses,
                    num_midwives,
                    ambulance_encoded,
                    ambulance_count
                ]])
                scaled_features = models['regressor'].transform(features)
                prediction = models['regressor'].predict(scaled_features)[0]
                st.success(f"🎯 **Predicted Average Monthly Patient Footfall: {int(prediction)}**")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Patient Footfall"},
                    delta={'reference': df['AverageMonthlyPatientFootfall'].mean() if 'AverageMonthlyPatientFootfall' in df.columns else prediction},
                    gauge={'axis': {'range': [0, max(prediction * 1.5, 1)]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, prediction * 0.5], 'color': "lightgray"},
                               {'range': [prediction * 0.5, prediction], 'color': "gray"}
                           ],
                           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': prediction * 1.2}}
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

elif page == "🏷️ Facility Classification":
    st.markdown("""
    <div class="main-header">
        <h1>🏷️ Facility Classification</h1>
        <p>Classify facilities based on their characteristics</p>
    </div>
    """, unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🏛️ Public/Private Classification", "🏥 Facility Type Classification"])
    with tab1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.header("🏛️ Public/Private Classification")
        models = load_models()
        df = load_data()
        if 'class_public_private' not in models:
            st.error("❌ Public/Private classification model not available.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if 'le_type' in models:
                    pp_facility_options = list(models['le_type'].classes_)
                else:
                    pp_facility_options = sorted(df['Type(Hospital/NursingHome/Lab)'].unique()) if 'Type(Hospital/NursingHome/Lab)' in df.columns else ['General Hospital', 'Nursing Home', 'Laboratory']
                pp_facility_type = st.selectbox("🏥 Facility Type", options=pp_facility_options, key="pp_type")
                pp_pharmacy = st.selectbox("💊 Pharmacy Available", options=["Yes", "No"], key="pp_pharmacy")
                pp_beds = st.number_input("🛏️ Number of Beds", min_value=0, value=50, key="pp_beds")
                pp_doctors = st.number_input("👨‍⚕️ Number of Doctors", min_value=0, value=10, key="pp_doctors")
            with col2:
                pp_nurses = st.number_input("👩‍⚕️ Number of Nurses", min_value=0, value=15, key="pp_nurses")
                pp_midwives = st.number_input("🤱 Number of Midwives", min_value=0, value=5, key="pp_midwives")
                pp_ambulance = st.selectbox("🚑 Ambulance Service Available", options=["Yes", "No"], key="pp_ambulance")
                if pp_ambulance == "Yes":
                    pp_ambulance_count = st.number_input("🚑 Ambulance Count", min_value=1, value=2, step=1, key="pp_count", help="Must be at least 1 if ambulance service is available")
                else:
                    pp_ambulance_count = 0
                    st.info("🚑 Ambulance count set to 0 since service is not available.")
            if st.button("🔍 Classify Public/Private", key="classify_pp"):
                if pp_ambulance == "Yes" and pp_ambulance_count < 1:
                    st.error("❌ Ambulance count must be at least 1 when ambulance service is available")
                    st.stop()
                elif pp_ambulance == "No" and pp_ambulance_count != 0:
                    st.error("❌ Ambulance count must be 0 when ambulance service is not available")
                    st.stop()
                if 'le_type' in models:
                    pp_encoded_type = models['le_type'].transform([pp_facility_type])[0]
                else:
                    type_mapping = {type_name: i for i, type_name in enumerate(pp_facility_options)}
                    pp_encoded_type = type_mapping.get(pp_facility_type, 0)
                pp_pharmacy_encoded = 1 if pp_pharmacy == "Yes" else 0
                pp_ambulance_encoded = 1 if pp_ambulance == "Yes" else 0
                pp_features = np.array([[
                    pp_encoded_type,
                    pp_pharmacy_encoded,
                    pp_beds,
                    pp_doctors,
                    pp_nurses,
                    pp_midwives,
                    pp_ambulance_encoded,
                    pp_ambulance_count
                ]])
                pp_scaled_features = models['class_scaler'].transform(pp_features)
                pp_prediction = models['class_public_private'].predict(pp_scaled_features)[0]
                if 'le_class' in models:
                    pp_class = models['le_class'].inverse_transform([pp_prediction])[0]
                else:
                    pp_class = "Public" if pp_prediction == 1 else "Private"
                icon = "🏛️" if pp_class == "Public" else "🏢"
                st.success(f"{icon} **Predicted Facility Class: {pp_class}**")
        st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.header("🏥 Facility Type Classification")
        models = load_models()
        df = load_data()
        if 'facility_type' not in models:
            st.error("❌ Facility type classification model not available.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                ft_pharmacy = st.selectbox("💊 Pharmacy Available", options=["Yes", "No"], key="ft_pharmacy")
                ft_beds = st.number_input("🛏️ Number of Beds", min_value=0, value=50, key="ft_beds")
                ft_doctors = st.number_input("👨‍⚕️ Number of Doctors", min_value=0, value=10, key="ft_doctors")
            with col2:
                ft_nurses = st.number_input("👩‍⚕️ Number of Nurses", min_value=0, value=15, key="ft_nurses")
                ft_midwives = st.number_input("🤱 Number of Midwives", min_value=0, value=5, key="ft_midwives")
                ft_ambulance = st.selectbox("🚑 Ambulance Service Available", options=["Yes", "No"], key="ft_ambulance")
                if ft_ambulance == "Yes":
                    ft_ambulance_count = st.number_input("🚑 Ambulance Count", min_value=1, value=2, step=1, key="ft_count", help="Must be at least 1 if ambulance service is available")
                else:
                    ft_ambulance_count = 0
                    st.info("🚑 Ambulance count set to 0 since service is not available.")
            if st.button("🔍 Classify Facility Type", key="classify_ft"):
                if ft_ambulance == "Yes" and ft_ambulance_count < 1:
                    st.error("❌ Ambulance count must be at least 1 when ambulance service is available")
                    st.stop()
                elif ft_ambulance == "No" and ft_ambulance_count != 0:
                    st.error("❌ Ambulance count must be 0 when ambulance is not available")
                    st.stop()
                ft_pharmacy_encoded = 1 if ft_pharmacy == "Yes" else 0
                ft_ambulance_encoded = 1 if ft_ambulance == "Yes" else 0
                ft_features = np.array([[
                    ft_pharmacy_encoded,
                    ft_beds,
                    ft_doctors,
                    ft_nurses,
                    ft_midwives,
                    ft_ambulance_encoded,
                    ft_ambulance_count
                ]])
                ft_scaled_features = models['facility_scaler'].transform(ft_features)
                ft_prediction = models['facility_type'].predict(ft_scaled_features)[0]
                if 'le_type' in models:
                    ft_type = models['le_type'].inverse_transform([ft_prediction])[0]
                else:
                    type_mapping = {
                        0: 'General Hospital',
                        1: 'Nursing Home',
                        2: 'Laboratory',
                        3: 'Clinic / Dispensary',
                        4: 'Hospital & Maternity',
                        5: 'Specialty Clinic',
                        6: 'Ayurvedic / Homeopathic',
                        7: 'Surgical / Procedural Facility',
                        8: 'Other'
                    }
                    ft_type = type_mapping.get(ft_prediction, 'Unknown')
                type_icons = {
                    "General Hospital": "🏥",
                    "Nursing Home": "🏠",
                    "Laboratory": "🔬",
                    "Clinic / Dispensary": "🏥",
                    "Hospital & Maternity": "🤱",
                    "Specialty Clinic": "⭐",
                    "Ayurvedic / Homeopathic": "🌿",
                    "Surgical / Procedural Facility": "🔪",
                    "Other": "🏢"
                }
                icon = type_icons.get(ft_type, "🏢")
                st.success(f"{icon} **Predicted Facility Type: {ft_type}**")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "🎯 Cluster Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Facility Cluster Analysis</h1>
        <p>Discover patterns and group similar facilities together</p>
    </div>
    """, unsafe_allow_html=True)
    cluster_names = ['📈 Development Needs', '🏗️ Basic Infrastructure', '⭐ Specialized Services']
    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Cluster Definitions:
    - **📈 Development Needs**: Facilities requiring infrastructure improvement
    - **🏗️ Basic Infrastructure**: Standard facilities with moderate resources
    - **⭐ Specialized Services**: Focused services with specific strengths
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    if 'Cluster' in df.columns:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig = px.bar(
            x=[cluster_names[i] for i in cluster_counts.index],
            y=cluster_counts.values,
            color=[cluster_names[i] for i in cluster_counts.index],
            color_discrete_sequence=cluster_colors,
            title="🏥 Facility Distribution Across Clusters"
        )
        fig.update_traces(
            text=cluster_counts.values,
            textposition='outside',
            textfont=dict(size=14, color='black')
        )
        fig.update_layout(
            showlegend=False,
            height=500,
            xaxis_title="Cluster Type",
            yaxis_title="Number of Facilities",
            margin=dict(l=20, r=20, t=60, b=20),
            font=dict(size=12, color='black')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📊 Cluster Characteristics")
        features_clust = [
            'NumberofBedsinfacilitytype',
            'NumberofDoctors/Physicians',
            'NumberofNurses',
            'NumberofMidwivesProfessional',
            'CountofAmbulance'
        ]
        features_clust = [f for f in features_clust if f in df.columns]
        if features_clust:
            cluster_summary = df.groupby('Cluster')[features_clust].mean()
            cluster_summary_formatted = cluster_summary.copy()
            for col in cluster_summary_formatted.columns:
                cluster_summary_formatted[col] = cluster_summary_formatted[col].round().astype(int)
            fig = px.imshow(
                cluster_summary_formatted.T,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                height=400,
                xaxis_title="Cluster",
                yaxis_title="Resource Type",
                font=dict(color='black')
            )
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(len(cluster_names))),
                ticktext=cluster_names
            )
            st.plotly_chart(fig, use_container_width=True)
            cluster_summary_formatted.index = [f"Cluster {i} ({cluster_names[i]})" for i in range(len(cluster_names))]
            st.dataframe(
                cluster_summary_formatted.style.format("{:.0f}"),
                use_container_width=True
            )
        else:
            st.warning("⚠️ No clustering features available in the dataset")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("❌ Cluster information not available")
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🔍 Find Cluster for Your Facility")
    st.markdown("*Enter your facility's characteristics to see which cluster it belongs to*")
    col1, col2 = st.columns(2)
    with col1:
        cluster_beds = st.number_input("🛏️ Number of Beds", min_value=0, value=50, step=1, key="cluster_beds")
        cluster_doctors = st.number_input("👨‍⚕️ Number of Doctors", min_value=0, value=10, step=1, key="cluster_docs")
        cluster_nurses = st.number_input("👩‍⚕️ Number of Nurses", min_value=0, value=15, step=1, key="cluster_nurses")
    with col2:
        cluster_midwives = st.number_input("🤱 Number of Midwives", min_value=0, value=5, step=1, key="cluster_midwives")
        cluster_ambulance_service = st.selectbox("🚑 Ambulance Service Available", options=["Yes", "No"], key="cluster_amb_service")
        if cluster_ambulance_service == "Yes":
            cluster_ambulance = st.number_input("🚑 Number of Ambulances", min_value=1, value=2, step=1, key="cluster_amb", help="Must be at least 1 if ambulance service is available")
        else:
            cluster_ambulance = 0
            st.info("🚑 Ambulance count set to 0 since service is not available.")
    colA, colB, colC = st.columns([1, 2, 1])
    with colB:
        if st.button("🎯 Find My Cluster", use_container_width=True):
            if cluster_ambulance_service == "Yes" and cluster_ambulance < 1:
                st.error("❌ Ambulance count must be at least 1 when ambulance service is available")
                st.stop()
            elif cluster_ambulance_service == "No" and cluster_ambulance != 0:
                st.error("❌ Ambulance count must be 0 when ambulance is not available")
                st.stop()
            if clust_scaler is not None and kmeans_model is not None:
                cluster_features = np.array([[
                    int(round(cluster_beds)),
                    int(round(cluster_doctors)),
                    int(round(cluster_nurses)),
                    int(round(cluster_midwives)),
                    int(round(cluster_ambulance))
                ]])
                scaled_features = clust_scaler.transform(cluster_features)
                pred = kmeans_model.predict(scaled_features)[0]
                cluster_name = cluster_names[pred]
                cluster_color = cluster_colors[pred]
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {cluster_color}40, {cluster_color}20);
                    border: 2px solid {cluster_color};
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    margin: 1rem 0;
                ">
                    <h2 style="color: {cluster_color}; margin-bottom: 1rem;">
                        🎯 Your Facility Cluster
                    </h2>
                    <h1 style="color: #2C3E50; margin: 0;">
                        {cluster_name}
                    </h1>
                </div>
                """, unsafe_allow_html=True)
                recommendations = {
                    0: ["🚀 Priority for resource allocation", "👥 Staff development programs", "🏗️ Infrastructure upgrades"],
                    1: ["📈 Focus on staff training", "🛏️ Consider expanding capacity", "💊 Enhance service offerings"],
                    2: ["⭐ Leverage specialized strengths", "🔗 Build partnerships", "📋 Document expertise"]
                }
                st.markdown("### 💡 Recommendations:")
                for rec in recommendations.get(pred, []):
                    st.markdown(f"• {rec}")
            else:
                st.error("❌ Clustering model not available")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "⚠️ Anomaly Detection":
    st.markdown("""
    <div class="main-header">
        <h1>⚠️ Anomaly Detection in Facilities</h1>
        <p>Identify facilities with unusual patterns that need attention</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 What we're looking for:")
    st.markdown("""
    - **📈 Unusual patient footfall** patterns (very high or very low)
    - **🚑 Ambulance service** inconsistencies
    - **👩‍⚕️ Staffing** anomalies compared to facility size
    - **📊 Resource allocation** inefficiencies
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    features_anomaly = [
        'AverageMonthlyPatientFootfall',
        'CountofAmbulance',
        'NumberofNurses'
    ]
    features_anomaly = [f for f in features_anomaly if f in df.columns]
    if not features_anomaly:
        st.error("❌ No anomaly detection features available in the dataset")
    else:
        from sklearn.ensemble import IsolationForest
        scaler_anomaly = StandardScaler()
        X_scaled_anomaly = scaler_anomaly.fit_transform(df[features_anomaly])
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_scores = iso_forest.fit_predict(X_scaled_anomaly)
        df_anomaly = df.copy()
        df_anomaly['AnomalyScore'] = anomaly_scores
        def generate_reason(row):
            reasons = []
            for feature in features_anomaly:
                val = row[feature]
                lower = df[feature].quantile(0.05)
                upper = df[feature].quantile(0.95)
                if val > upper:
                    reasons.append(f'📈 High {feature}')
                elif val < lower:
                    reasons.append(f'📉 Low {feature}')
            return '; '.join(reasons) if reasons else '🔍 Unusual pattern detected'
        anomalies = df_anomaly[df_anomaly['AnomalyScore'] == -1].copy()
        anomalies['AnomalyReason'] = anomalies.apply(generate_reason, axis=1)
        col1, col2, col3 = st.columns(3)
        create_metric_card(len(anomalies), "Anomalies Found", col1)
        create_metric_card(f"{len(anomalies)/len(df)*100:.1f}%", "Anomaly Rate", col2)
        create_metric_card(len(df) - len(anomalies), "Normal Facilities", col3)
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader(f"⚠️ Detected Anomalies")
        if len(anomalies) > 0:
            display_cols = ['FacilityName', 'Type(Hospital/NursingHome/Lab)', 'Class:(Public/Private)', 'AnomalyReason']
            for col in features_anomaly:
                if col in anomalies.columns:
                    display_cols.append(col)
            st.dataframe(anomalies[display_cols].style.background_gradient(cmap='Pastel1'), use_container_width=True, height=400)
            csv = anomalies.to_csv(index=False)
            st.download_button("📥 Download Anomaly Report as CSV", data=csv, file_name="healthcare_anomalies.csv", mime="text/csv")
        else:
            st.info("✅ No anomalies detected in the current dataset!")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📊 Anomaly Distribution Analysis")
        if len(features_anomaly) >= 2:
            fig = px.scatter(
                df_anomaly,
                x=features_anomaly[0],
                y=features_anomaly[1],
                color=df_anomaly['AnomalyScore'].map({1: 'Normal', -1: 'Anomaly'}),
                color_discrete_map={'Normal': '#4ECDC4', 'Anomaly': '#FF6B6B'},
                title=f"Anomaly Detection: {features_anomaly[0]} vs {features_anomaly[1]}",
                hover_data=['FacilityName'] if 'FacilityName' in df.columns else None
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        cols = st.columns(min(3, len(features_anomaly)))
        for i, feature in enumerate(features_anomaly[:3]):
            with cols[i]:
                fig = px.histogram(
                    df,
                    x=feature,
                    nbins=20,
                    title=f"Distribution: {feature}",
                    color_discrete_sequence=['#667eea']
                )
                if feature in df.columns:
                    q95 = df[feature].quantile(0.95)
                    q05 = df[feature].quantile(0.05)
                    fig.add_vline(x=q95, line_dash="dash", line_color="red", annotation_text="95th percentile", annotation_position="top right")
                    fig.add_vline(x=q05, line_dash="dash", line_color="red", annotation_text="5th percentile", annotation_position="bottom right")
                fig.update_layout(height=300, showlegend=False, font=dict(size=12, color='black'), margin=dict(l=10, r=10, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-top: 2rem;
">
    <h3>🏥 Healthcare Facility Analytics Dashboard</h3>
    <p>
        <small>Empowering healthcare decisions through data-driven insights</small>
    </p>
    <div style="margin-top: 1rem;">
        <span style="margin: 0 1rem;">📊 Analytics</span>
        <span style="margin: 0 1rem;">🤖 ML Predictions</span>
        <span style="margin: 0 1rem;">🎯 Clustering</span>
        <span style="margin: 0 1rem;">⚠️ Anomaly Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)