import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torchvision.models as tv_models
import json
import pickle
import joblib
import faiss
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
import os
import sys
import tempfile

# Page config
st.set_page_config(
    page_title="Car Vision & Sales Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM ANIMATED CSS & STYLING
# ============================================
st.markdown("""
<style>
    /* Smooth Transitions */
    * {
        transition: all 0.25s ease-in-out;
    }

    /* Custom Gradient Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        padding: 2.5rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: fadeIn 1s ease-in-out;
    }

    /* Floating Animation for Header Icons */
    .floating-icon {
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Pulsing Badge */
    .pulse-badge {
        display: inline-block;
        padding: 0.35em 0.8em;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 20px;
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Dynamic Card Styles */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
    }

    .info-card:hover {
        transform: translateY(-4px);
        border-color: #6366f1;
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.15);
    }

    /* Metric Cards Enhancement */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1rem;
        border-radius: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Feature schema for scalers
SCALER_FEATURES = [
    'day_of_week', 'week_of_year', 'season', 'cost', 
    'gross_sales', 'profit', 'rolling_mean_7', 'rolling_std_7', 
    'rolling_max_7', 'quantity', 'model', 'price_band'
]

QTY_MODEL_FEATURES = [
    'day_of_week', 'week_of_year', 'season', 'cost', 
    'gross_sales', 'profit', 'rolling_mean_7', 'rolling_std_7', 
    'rolling_max_7', 'model', 'price_band', 'gender', 
    'income_customer', 'discount'
]

# ============================================
# 1. LOAD MODELS & DATA
# ============================================

def load_all_models():
    """Load all models and data with caching"""
    BASE_DIR = Path.cwd().parent.parent
    MODELS_DIR = BASE_DIR / 'development' / 'models'
    DATA_DIR = BASE_DIR / 'development' / 'database'
    
    sales_data = pd.read_parquet(DATA_DIR / 'car_sales_prediction_sales.parquet')
    
    cv_dir = MODELS_DIR / 'computer_vision_2'
    with open(cv_dir / 'brand_mapping.json', 'r') as f:
        brand_mapping = json.load(f)
    
    index = faiss.read_index(str(cv_dir / 'car_index.faiss'))
    feature_data = pd.read_csv(cv_dir / 'feature_data.csv')
    feature_matrix = np.load(cv_dir / 'feature_matrix.npy')
    
    with open(cv_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    sales_dir = MODELS_DIR / 'sales_prediction'
    
    try:
        sales_scaler = joblib.load(sales_dir / 'scalers' / 'feature_scaler.pkl')
    except Exception:
        sales_scaler = None
    
    try:
        sales_target_scaler = joblib.load(sales_dir / 'scalers' / 'target_scaler.pkl')
    except Exception:
        sales_target_scaler = None
    
    sales_models = {}
    for name, filename in [('XGBoost', 'xgboost.pkl'), ('Random Forest', 'random_forest.pkl'), ('Decision Tree', 'decision_tree.pkl')]:
        try:
            sales_models[name] = joblib.load(sales_dir / 'models' / filename)
        except Exception:
            sales_models[name] = None
    
    try:
        from catboost import CatBoostRegressor
        sales_models['CatBoost'] = CatBoostRegressor()
        sales_models['CatBoost'].load_model(sales_dir / 'models' / 'catboost.cbm')
    except Exception:
        sales_models['CatBoost'] = None
    
    qty_dir = MODELS_DIR / 'quantity_prediction'
    
    try:
        qty_scaler = joblib.load(qty_dir / 'scalers' / 'feature_scaler.pkl')
    except Exception:
        qty_scaler = None

    try:
        qty_target_scaler = joblib.load(qty_dir / 'scalers' / 'target_scaler.pkl')
    except Exception:
        qty_target_scaler = None
    
    try:
        with open(qty_dir / 'parameters' / 'feature_columns.json', 'r') as f:
            qty_features = json.load(f)
    except Exception:
        qty_features = QTY_MODEL_FEATURES
    
    qty_models = {}
    for name, filename in [('XGBoost', 'xgboost.pkl'), ('Random Forest', 'random_forest.pkl'), ('Decision Tree', 'decision_tree.pkl')]:
        try:
            qty_models[name] = joblib.load(qty_dir / 'models' / filename)
        except Exception:
            qty_models[name] = None
            
    try:
        from catboost import CatBoostRegressor
        qty_models['CatBoost'] = CatBoostRegressor()
        qty_models['CatBoost'].load_model(qty_dir / 'models' / 'catboost.cbm')
    except Exception:
        qty_models['CatBoost'] = None
    
    sales_metrics = pd.read_csv(sales_dir / 'metrics' / 'model_metrics.csv')
    qty_metrics = pd.read_csv(qty_dir / 'metrics' / 'model_metrics.csv')
    
    return {
        'sales_data': sales_data,
        'brand_mapping': brand_mapping,
        'index': index,
        'feature_data': feature_data,
        'feature_matrix': feature_matrix,
        'metadata': metadata,
        'sales_scaler': sales_scaler,
        'sales_target_scaler': sales_target_scaler,
        'sales_models': sales_models,
        'sales_metrics': sales_metrics,
        'qty_scaler': qty_scaler,
        'qty_target_scaler': qty_target_scaler,
        'qty_models': qty_models,
        'qty_features': qty_features,
        'qty_metrics': qty_metrics
    }

@st.cache_resource
def init_feature_extractor():
    """Initialize Vision Transformer feature extractor"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT)
    model.heads = nn.Identity()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return {
        "device": device,
        "model": model,
        "feature_extractor": model,
        "transform": transform
    }

def extract_features(image, extractor):
    device = extractor["device"]
    transform = extractor["transform"]
    feature_extractor = extractor.get("feature_extractor", extractor.get("model"))

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = features.squeeze().cpu().numpy()

    return features

def normalize_features(features):
    return features / (np.linalg.norm(features) + 1e-8)

# Load resources
with st.spinner("🚀 Initializing AI Engine & Neural Weights..."):
    models = load_all_models()
    extractor = init_feature_extractor()

sales_data = models['sales_data']
brand_mapping = models['brand_mapping']
index = models['index']
feature_data = models['feature_data']

available_sales_models = {k: v for k, v in models['sales_models'].items() if v is not None}
available_qty_models = {k: v for k, v in models['qty_models'].items() if v is not None}

# ============================================
# 2. PREDICTION FUNCTIONS
# ============================================

def prepare_features_for_prediction(brand_info, model_type='sales'):
    avg_price = brand_info.get('avg_price', 0)
    qty = brand_info.get('total_quantity', 1)
    profit = brand_info.get('profit_margin', 0.2)
    
    input_dict = {
        'day_of_week': 0,
        'week_of_year': 1,
        'season': 1,
        'cost': float(avg_price),
        'gross_sales': float(avg_price * qty),
        'profit': float(profit),
        'rolling_mean_7': float(qty),
        'rolling_std_7': 0.0,
        'rolling_max_7': float(qty),
        'quantity': float(qty),
        'model': 0,
        'price_band': 0
    }
    return pd.DataFrame([input_dict])[SCALER_FEATURES]

def prepare_qty_features_for_prediction(brand_info, sales_data=None):
    avg_price = brand_info.get('avg_price', 0)
    qty = brand_info.get('total_quantity', 1)
    profit = brand_info.get('profit_margin', 0.2)
    
    input_dict = {
        'day_of_week': 0,
        'week_of_year': 1,
        'season': 1,
        'cost': float(avg_price),
        'gross_sales': float(avg_price * qty),
        'profit': float(profit),
        'rolling_mean_7': float(qty),
        'rolling_std_7': 0.0,
        'rolling_max_7': float(qty),
        'model': 0,
        'price_band': 0,
        'gender': 0,
        'income_customer': 0,
        'discount': 0.0
    }
    
    target_columns = models.get('qty_features', QTY_MODEL_FEATURES)
    for col in target_columns:
        if col not in input_dict:
            input_dict[col] = 0
            
    return pd.DataFrame([input_dict])[target_columns]

def predict_with_model(model, features_df, model_type='sales'):
    try:
        if model_type == 'sales':
            scaler = models['sales_scaler']
            target_scaler = models['sales_target_scaler']
        else:
            scaler = models['qty_scaler']
            target_scaler = models['qty_target_scaler']

        if scaler is not None:
            scaled_array = scaler.transform(features_df)
            features_input = pd.DataFrame(scaled_array, columns=features_df.columns)
        else:
            features_input = features_df

        prediction = model.predict(features_input)

        if target_scaler is not None:
            prediction = target_scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
            return max(0.0, float(prediction[0][0]))

        val = float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction)
        return max(0.0, val)

    except Exception as e:
        st.warning(f"Prediction failed ({model_type}): {e}")
        return None

def predict_all_models(brand_info):
    results = {'sales': {}, 'quantity': {}}

    sales_features_df = prepare_features_for_prediction(brand_info, model_type='sales')
    for model_name, model in available_sales_models.items():
        results['sales'][model_name] = predict_with_model(model, sales_features_df, model_type='sales')

    qty_features_df = prepare_qty_features_for_prediction(brand_info)
    for model_name, model in available_qty_models.items():
        results['quantity'][model_name] = predict_with_model(model, qty_features_df, model_type='quantity')

    return results

# ============================================
# 3. RECOMMENDATION ENGINE
# ============================================

def get_recommendations(image_path, k=5, price_range=None, selected_models=None):
    image = Image.open(image_path).convert('RGB')
    query_features = extract_features(image, extractor)
    query_features_norm = normalize_features(query_features).reshape(1, -1).astype(np.float32)

    distances, indices = index.search(query_features_norm, k * 2)

    results = []
    seen_brands = set()

    for idx, score in zip(indices[0], distances[0]):
        if idx < len(feature_data):
            brand = feature_data.iloc[idx]['brand']

            if brand in seen_brands:
                continue

            brand_info = brand_mapping.get(brand, {})
            brand_sales = sales_data[sales_data['company'].str.contains(brand, case=False, na=False)]

            result = {
                'brand': brand,
                'path': feature_data.iloc[idx]['path'],
                'similarity': float(score),
                'has_sales': brand_info.get('has_sales', False),
                'avg_price': brand_info.get('avg_price', 0),
                'total_quantity': brand_info.get('total_quantity', 0),
                'total_sales': brand_info.get('total_sales', 0),
                'profit_margin': brand_info.get('profit_margin', 0),
                'transaction_count': brand_info.get('transaction_count', 0),
                'models': brand_info.get('models', [])
            }

            if len(brand_sales) > 0:
                top_model = brand_sales.groupby('model')['quantity'].sum().idxmax()
                result['top_model'] = top_model
                result['top_model_qty'] = int(brand_sales[brand_sales['model'] == top_model]['quantity'].sum())

            if result['has_sales'] and selected_models:
                result['predictions'] = predict_all_models(brand_info)

            if price_range:
                min_price, max_price = price_range
                if result['avg_price'] and (result['avg_price'] < min_price or result['avg_price'] > max_price):
                    continue

            results.append(result)
            seen_brands.add(brand)

            if len(results) >= k:
                break

    return results

# ============================================
# 4. UI COMPONENTS
# ============================================

def render_hero_banner():
    st.markdown("""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="pulse-badge">● AI System Active</span>
                <h1 style="margin-top: 0.5rem; font-size: 2.2rem; font-weight: 800; color: white;">
                    <span class="floating-icon">🏎️</span> Car Vision & Sales Intelligence
                </h1>
                <p style="color: #94a3b8; font-size: 1.05rem; max-width: 650px;">
                    Upload any vehicle image to instantly perform <b>Visual Vector Search (ViT + FAISS)</b>, 
                    retrieve real transactional sales metrics, and generate predictive ML forecasts.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_empty_state_placeholders():
    st.markdown("### 🌟 Platform Capabilities & Guided Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>👁️ Visual Search Engine</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Powered by <b>Vision Transformer (ViT-B/16)</b> feature extraction paired with high-dimensional <b>FAISS Indexing</b> for sub-millisecond similarity matching.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Sales Forecasting</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Integrated multi-model forecasting engine combining <b>XGBoost, CatBoost, and Random Forest</b> to project revenue and unit sales.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>📈 Market Intelligence</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Gain deep analytical breakdowns of revenue distribution, profit margins, body-style trends, and historical transactions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Get Started:** Upload a vehicle image from the sidebar control panel to trigger live predictions!")

def render_dev_overlay(models, brand_info=None):
    with st.expander("🛠️ DEVELOPER DEBUG PANEL", expanded=True):
        tab1, tab2, tab3 = st.tabs(["📁 Model Load Status", "🔢 Feature Scaling Inspector", "🖼️ CV Vector Info"])

        with tab1:
            col_s, col_q = st.columns(2)
            with col_s:
                st.markdown("**Sales Models Status**")
                for k, v in models['sales_models'].items():
                    st.write(f"• `{k}`: {'✅ Loaded' if v is not None else '❌ Missing/Failed'}")
                st.write(f"• `Sales Scaler`: {'✅ Loaded' if models['sales_scaler'] is not None else '❌ Missing'}")
                
            with col_q:
                st.markdown("**Quantity Models Status**")
                for k, v in models['qty_models'].items():
                    st.write(f"• `{k}`: {'✅ Loaded' if v is not None else '❌ Missing/Failed'}")
                st.write(f"• `Qty Scaler`: {'✅ Loaded' if models['qty_scaler'] is not None else '❌ Missing'}")

        with tab2:
            col_sf, col_qf = st.columns(2)
            dummy_info = brand_info if brand_info else {'avg_price': 250000000, 'total_quantity': 50, 'profit_margin': 0.15}
            
            with col_sf:
                st.markdown("**Sales Features Input**")
                df_s = prepare_features_for_prediction(dummy_info)
                st.write(f"Shape: `{df_s.shape}`")
                st.dataframe(df_s, use_container_width=True)
                
            with col_qf:
                st.markdown("**Quantity Features Input**")
                df_q = prepare_qty_features_for_prediction(dummy_info)
                st.write(f"Shape: `{df_q.shape}`")
                st.dataframe(df_q, use_container_width=True)

        with tab3:
            st.json({
                "FAISS Total Vector Index Count": int(models['index'].ntotal),
                "Feature Matrix Dimension": list(models['feature_matrix'].shape),
                "Indexed Feature Data Rows": len(models['feature_data']),
                "Mapped Brand Records": len(models['brand_mapping'])
            })

def sidebar_controls():
    with st.sidebar:
        st.title("🏎️ Control Center")
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Upload Vehicle Image",
            type=['jpg', 'jpeg', 'png'],
            help="Select a clear vehicle photo to execute similarity search"
        )
        
        st.markdown("---")
        st.subheader("🤖 Model Engines")
        
        col1, col2 = st.columns(2)
        with col1:
            sales_model = st.selectbox("Sales Model", options=list(available_sales_models.keys()), index=0)
            sales_metric = models['sales_metrics'][models['sales_metrics']['Model'] == sales_model]
            if not sales_metric.empty:
                st.caption(f"R²: `{sales_metric['R2'].values[0]:.3f}`")
        
        with col2:
            qty_model = st.selectbox("Quantity Model", options=list(available_qty_models.keys()), index=0)
            qty_metric = models['qty_metrics'][models['qty_metrics']['Model'] == qty_model]
            if not qty_metric.empty:
                st.caption(f"R²: `{qty_metric['R2'].values[0]:.3f}`")
        
        st.markdown("---")
        st.subheader("⚙️ Query Parameters")
        
        k_results = st.slider("Max Results (K)", min_value=3, max_value=10, value=5)
        
        min_price = st.number_input("Min Price (IDR)", min_value=0, value=0, step=10000000)
        max_price = st.number_input("Max Price (IDR)", min_value=0, value=2000000000, step=50000000)
        price_range = (min_price, max_price) if min_price < max_price else None
        
        show_sales = st.toggle("📊 Sales Analytics", value=True)
        show_comparison = st.toggle("📈 Model Benchmark", value=False)
        dev_mode = st.toggle("🛠️ Developer Overlay", value=False)
        
        st.markdown("---")
        st.caption("⚡ Powered by Vision Transformer & FAISS")
        
        return uploaded_file, k_results, price_range, show_sales, show_comparison, sales_model, qty_model, dev_mode

def display_metrics(results, sales_data):
    valid_results = [r for r in results if r['has_sales']]
    
    if valid_results:
        avg_price = np.mean([r['avg_price'] for r in valid_results])
        total_sales = sum([r['total_quantity'] for r in valid_results])
        avg_margin = np.mean([r['profit_margin'] for r in valid_results])
        top_brand = max(valid_results, key=lambda x: x['total_sales'])
    else:
        avg_price, total_sales, avg_margin = 0, 0, 0
        top_brand = {'brand': 'N/A'}
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Avg Market Price", f"Rp {avg_price/1_000_000:,.1f}M")
    with col2:
        st.metric("📦 Cumulative Sales", f"{total_sales:,.0f} units")
    with col3:
        st.metric("📈 Profit Margin", f"{avg_margin*100:.1f}%")
    with col4:
        st.metric("🏆 Top Revenue Brand", top_brand.get('brand', 'N/A'))

def display_recommendation_cards(results, show_sales, selected_sales_model, selected_qty_model):
    cols = st.columns(min(len(results), 4))
    
    for idx, (col, result) in enumerate(zip(cols, results)):
        with col:
            try:
                img = Image.open(result['path']).convert('RGB')
                st.image(img, use_container_width=True)
            except Exception:
                st.image("https://via.placeholder.com/200x150?text=No+Image", use_container_width=True)
            
            st.markdown(f"### {result['brand']}")
            score = result['similarity']
            
            st.markdown(f"🟢 Match Similarity: **{score:.2%}**" if score > 0.70 else 
                        f"🟠 Match Similarity: **{score:.2%}**" if score > 0.50 else 
                        f"🔴 Match Similarity: **{score:.2%}**")
            
            if show_sales and result['has_sales']:
                if result.get('top_model'):
                    st.caption(f"🏷️ **Top Model:** {result['top_model']}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Volume", f"{result['total_quantity']:,.0f}")
                with c2:
                    st.metric("Price", f"Rp{result['avg_price']/1_000_000:,.0f}M")
                
                if 'predictions' in result:
                    st.markdown("---")
                    sales_pred = result['predictions']['sales'].get(selected_sales_model)
                    if sales_pred is not None:
                        st.metric(f"Pred. Sales ({selected_sales_model})", f"{sales_pred:,.0f}")
                    
                    qty_pred = result['predictions']['quantity'].get(selected_qty_model)
                    if qty_pred is not None:
                        st.metric(f"Pred. Qty ({selected_qty_model})", f"{qty_pred:,.0f}")
            else:
                st.caption("⚠️ No transactional records available")

def display_analytics_charts(results, sales_data, selected_sales_model):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Similarity vs Price Index", 
            "Actual vs Predicted Volume", 
            "Profit Margin vs Volume", 
            "Market Share Breakdown"
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy", "secondary_y": True}, {"type": "domain"}]]
    )
    
    valid_results = [r for r in results if r['has_sales']]
    
    if valid_results:
        brands = [r['brand'] for r in valid_results]
        similarities = [r['similarity'] for r in valid_results]
        prices = [r['avg_price'] / 1_000_000 for r in valid_results]
        
        fig.add_trace(
            go.Scatter(
                x=prices, y=similarities, mode='markers+text',
                text=brands, textposition="top center",
                marker=dict(size=14, color=similarities, colorscale='Plasma')
            ),
            row=1, col=1
        )
        
        actual_sales = [r['total_quantity'] for r in valid_results]
        predicted_sales = [
            r['predictions']['sales'].get(selected_sales_model, 0) if 'predictions' in r else 0 
            for r in valid_results
        ]
        
        fig.add_trace(go.Bar(x=brands, y=actual_sales, name='Actual Sales', marker_color='#6366f1'), row=1, col=2)
        fig.add_trace(go.Bar(x=brands, y=predicted_sales, name=f'Pred. ({selected_sales_model})', marker_color='#38bdf8'), row=1, col=2)
        
        margins = [r['profit_margin'] * 100 for r in valid_results]
        volumes = [r['total_quantity'] for r in valid_results]
        
        fig.add_trace(
            go.Scatter(x=brands, y=margins, mode='lines+markers', name='Margin %', marker=dict(size=8), line=dict(color='#10b981')),
            row=2, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Bar(x=brands, y=volumes, name='Sales Volume', marker_color='#f59e0b', opacity=0.4),
            row=2, col=1, secondary_y=True
        )
        
        transactions = [r['transaction_count'] for r in valid_results]
        fig.add_trace(go.Pie(labels=brands, values=transactions, hole=0.4), row=2, col=2)
    
    fig.update_layout(height=650, template='plotly_dark', showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

def display_transaction_data(results, sales_data, selected_sales_model, selected_qty_model):
    """Display expandable transaction data table"""
    
    st.subheader("📜 Transaction History Breakdown")
    
    # Prepare data
    table_data = []
    for result in results:
        if result['has_sales']:
            brand = result['brand']
            brand_sales = sales_data[
                sales_data['company'].str.contains(brand, case=False, na=False)
            ]
            
            # Get body style distribution
            body_styles = brand_sales['body_style'].value_counts().to_dict()
            top_style = max(body_styles.items(), key=lambda x: x[1]) if body_styles else ('N/A', 0)
            
            # Get predictions safely (handle None values)
            pred_sales = 'N/A'
            pred_qty = 'N/A'
            
            if 'predictions' in result:
                # Safely format sales prediction
                raw_sales = result['predictions']['sales'].get(selected_sales_model)
                if raw_sales is not None:
                    pred_sales = f"{raw_sales:,.0f}"
                
                # Safely format quantity prediction
                raw_qty = result['predictions']['quantity'].get(selected_qty_model)
                if raw_qty is not None:
                    pred_qty = f"{raw_qty:,.0f}"
            
            table_data.append({
                'Brand': brand,
                'Models': ', '.join(result.get('models', [])[:3]),
                'Total Quantity': f"{result['total_quantity']:,.0f}",
                'Total Sales': f"Rp{result['total_sales']:,.0f}",
                'Avg Price': f"Rp{result['avg_price']:,.0f}",
                'Profit Margin': f"{result['profit_margin']*100:.1f}%",
                'Transactions': f"{result['transaction_count']:,}",
                'Top Body Style': f"{top_style[0]} ({top_style[1]} units)",
                f'Pred Sales ({selected_sales_model})': pred_sales,
                f'Pred Qty ({selected_qty_model})': pred_qty
            })
        else:
            table_data.append({
                'Brand': result['brand'],
                'Models': '-',
                'Total Quantity': 'No Data',
                'Total Sales': 'No Data',
                'Avg Price': 'No Data',
                'Profit Margin': 'No Data',
                'Transactions': 'No Data',
                'Top Body Style': 'No Data',
                f'Pred Sales ({selected_sales_model})': 'N/A',
                f'Pred Qty ({selected_qty_model})': 'N/A'
            })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================
# 5. MAIN APP ENTRY
# ============================================

def main():
    uploaded_file, k_results, price_range, show_sales, show_comparison, selected_sales_model, selected_qty_model, dev_mode = sidebar_controls()
    
    render_hero_banner()
    
    if dev_mode:
        render_dev_overlay(models)
        st.markdown("---")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        
        with st.spinner("⚡ Executing Vision Search & Forecast Engine..."):
            results = get_recommendations(temp_path, k=k_results, price_range=price_range, selected_models=True)
        
        if results:
            if dev_mode:
                render_dev_overlay(models, brand_info=results[0])
                st.markdown("---")

            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(Image.open(temp_path).convert('RGB'), caption="🔍 Query Vehicle", use_container_width=True)
            with col2:
                st.subheader("📊 Analytics Summary")
                display_metrics(results, sales_data)
            
            st.markdown("---")
            st.subheader("🏎️ Top Matching Vehicles")
            display_recommendation_cards(results, show_sales, selected_sales_model, selected_qty_model)
            
            st.markdown("---")
            if show_sales:
                st.subheader("📈 Intelligence Analytics & Market Trends")
                display_analytics_charts(results, sales_data, selected_sales_model)

            st.markdown("---")
            display_transaction_data(results, sales_data, selected_sales_model, selected_qty_model)
        else:
            st.warning("No matching vehicle records found matching the configured price range.")
    else:
        render_empty_state_placeholders()

if __name__ == "__main__":
    main()