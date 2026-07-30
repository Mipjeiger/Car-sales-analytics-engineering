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
# 1. LOAD MODELS & DATA
# ============================================

def load_all_models():
    """Load all models and data with caching"""
    
    BASE_DIR = Path.cwd()
    MODELS_DIR = BASE_DIR / 'development' / 'models'
    DATA_DIR = BASE_DIR / 'development' / 'database'
    
    # Load sales data
    sales_data = pd.read_parquet(DATA_DIR /  'car_sales_prediction_sales.parquet')
    
    # Load computer vision models
    cv_dir = MODELS_DIR / 'computer_vision_2'
    with open(cv_dir / 'brand_mapping.json', 'r') as f:
        brand_mapping = json.load(f)
    
    index = faiss.read_index(str(cv_dir / 'car_index.faiss'))
    feature_data = pd.read_csv(cv_dir / 'feature_data.csv')
    feature_matrix = np.load(cv_dir / 'feature_matrix.npy')
    
    with open(cv_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load ALL sales prediction models
    sales_dir = MODELS_DIR / 'sales_prediction'
    
    # Load scalers
    try:
        sales_scaler = joblib.load(sales_dir / 'scalers' / 'feature_scaler.pkl')
    except Exception:
        sales_scaler = None
    
    # Load feature columns
    try:
        sales_target_scaler = joblib.load(sales_dir / 'scalers' / 'target_scaler.pkl')
    except Exception:
        sales_target_scaler = None
    
    # Load all sales models
    sales_models = {}
    
    # XGBoost
    try:
        sales_models['XGBoost'] = joblib.load(sales_dir / 'models' / 'xgboost.pkl')
    except:
        sales_models['XGBoost'] = None
    
    # Random Forest
    try:
        sales_models['Random Forest'] = joblib.load(sales_dir / 'models' / 'random_forest.pkl')
    except:
        sales_models['Random Forest'] = None
    
    # Decision Tree
    try:
        sales_models['Decision Tree'] = joblib.load(sales_dir / 'models' / 'decision_tree.pkl')
    except:
        sales_models['Decision Tree'] = None
    
    # CatBoost
    try:
        # CatBoost uses .cbm format
        from catboost import CatBoostRegressor
        sales_models['CatBoost'] = CatBoostRegressor()
        sales_models['CatBoost'].load_model(sales_dir / 'models' / 'catboost.cbm')
    except:
        sales_models['CatBoost'] = None
    
    # Load ALL quantity prediction models
    qty_dir = MODELS_DIR / 'quantity_prediction'
    
    # Load scalers
    try:
        qty_scaler = joblib.load(qty_dir / 'scalers' / 'feature_scaler.pkl')
    except Exception:
        qty_scaler = None

    try:
        qty_target_scaler = joblib.load(qty_dir / 'scalers' / 'target_scaler.pkl')
    except Exception:
        qty_target_scaler = None
    
    # Load feature columns
    with open(qty_dir / 'parameters' / 'feature_columns.json', 'r') as f:
        qty_features = json.load(f)
    
    # Load all quantity models
    qty_models = {}
    
    # XGBoost
    try:
        qty_models['XGBoost'] = joblib.load(qty_dir / 'models' / 'xgboost.pkl')
    except:
        qty_models['XGBoost'] = None
    
    # Random Forest
    try:
        qty_models['Random Forest'] = joblib.load(qty_dir / 'models' / 'random_forest.pkl')
    except:
        qty_models['Random Forest'] = None
    
    # Decision Tree
    try:
        qty_models['Decision Tree'] = joblib.load(qty_dir / 'models' / 'decision_tree.pkl')
    except:
        qty_models['Decision Tree'] = None
    
    # CatBoost
    try:
        from catboost import CatBoostRegressor
        qty_models['CatBoost'] = CatBoostRegressor()
        qty_models['CatBoost'].load_model(qty_dir / 'models' / 'catboost.cbm')
    except:
        qty_models['CatBoost'] = None
    
    # Load model metrics
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
        #'sales_features': sales_features,
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    return {
        "device": device,
        "model": model,
        "feature_extractor": model,
        "transform": transform
    }

def extract_features(image, extractor):
    """Extract features from PIL image"""
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

# Load all models and data
with st.spinner("Loading models and data..."):
    models = load_all_models()
    extractor = init_feature_extractor()

sales_data = models['sales_data']
brand_mapping = models['brand_mapping']
index = models['index']
feature_data = models['feature_data']

# Filter out None models
available_sales_models = {k: v for k, v in models['sales_models'].items() if v is not None}
available_qty_models = {k: v for k, v in models['qty_models'].items() if v is not None}

# ============================================
# 2. PREDICTION FUNCTIONS
# ============================================
def prepare_features_for_prediction(brand_info, model_type='sales'):
    """
    Prepare features for prediction based on brand info and model type 
    -> fetched on dataframe also to get comprehensive features
    """
    avg_price = brand_info.get('avg_price', 0)

    # Create feature array based on model type
    if model_type == 'sales':
        features = np.array([[
            avg_price / 1000000,
            brand_info.get('discount', 0.1),
            brand_info.get('profit_margin', 0.2),
            brand_info.get('income_customer', 20000000),
            brand_info.get('quantity', 1)
        ]])
        return features

    else: # model_type == 'quantity':
        features = np.array([[
            avg_price / 1000000,
            brand_info.get('discount', 0.1),
            brand_info.get('profit_margin', 0.2),
            brand_info.get('income_customer', 20000000)
        ]])
        return features

def predict_with_model(model, features, model_type='sales'):
    """Predict using the given model and features"""
    try:
        # Get appropriate scaler
        if model_type == 'sales':
            scaler = models['sales_scaler']
            target_scaler = models['sales_target_scaler']
        else:  # model_type == 'quantity'
            scaler = models['qty_scaler']
            target_scaler = models['qty_target_scaler']

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)

        # Inverse transform if target scaler exists
        if target_scaler is not None:
            prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))
            return float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])

        return float(prediction[0]) if prediction.ndim > 1 else float(prediction)

    except Exception as e:
        st.warning(f"Prediction failed: {e}")
        return None

def predict_all_models(brand_info):
    """Predict using all available models and return results"""
    results = {
        'sales': {},
        'quantity': {}
    }

    # Sales prediction
    sales_features = prepare_features_for_prediction(brand_info, model_type='sales')
    for model_name, model in available_sales_models.items():
        pred = predict_with_model(model, sales_features, model_type='sales')
        results['sales'][model_name] = pred

    # Quantity prediction
    qty_features = prepare_features_for_prediction(brand_info, model_type='quantity')
    for model_name, model in available_qty_models.items():
        pred = predict_with_model(model, qty_features, model_type='quantity')
        results['quantity'][model_name] = pred

    return results

# ============================================
# 3. Recommendation Engine
# ============================================

def get_recommendations(image_path, k=5, price_range=None, selected_models=None):
    """Get car recommendations with sales data"""

    # Load and process image
    image = Image.open(image_path).convert('RGB')

    # Extract features
    query_features = extract_features(image, extractor)
    query_features_norm = normalize_features(query_features).reshape(1, -1).astype(np.float32)

    # Search FAISS
    distances, indices = index.search(query_features_norm, k * 2)

    # Get results with sales data
    results = []
    seen_brands = set()

    for idx, score in zip(indices[0], distances[0]):
        if idx < len(feature_data):
            brand = feature_data.iloc[idx]['brand']

            if brand in seen_brands:
                continue  # Skip duplicate brands

            # Get brand info
            brand_info = brand_mapping.get(brand, {})

            # Get detailed sales info
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

            # Get model
            if len(brand_sales) > 0:
                top_model = brand_sales.groupby('model')['quantity'].sum().idxmax()
                result['top_model'] = top_model
                result['top_model_qty'] = int(brand_sales[brand_sales['model'] == top_model]['quantity'].sum())

            # Get prediction from all models if sales data exists
            if result['has_sales'] and selected_models:
                predictions = predict_all_models(brand_info)
                result['predictions'] = predictions

            # Apply price filter
            if price_range:
                min_price, max_price = price_range
                if result['avg_price'] and (result['avg_price'] < min_price or result['avg_price'] > max_price):
                    continue  # Skip if outside price range

            results.append(result)
            seen_brands.add(brand)

            if len(results) >= k:
                break  # Stop if we have enough results

    return results

# ============================================
# 4. UI COMPONENTS
# ============================================

def sidebar_controls():
    """Render sidebar controls with model selection"""
    with st.sidebar:
        st.title("🏎️ Car Vision")
        st.markdown("---")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload Car Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a car image to find similar vehicles"
        )
        
        st.markdown("---")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sales Prediction**")
            sales_model = st.selectbox(
                "Select Model",
                options=list(available_sales_models.keys()),
                index=0,
                key="sales_model"
            )
            
            # Show model metrics
            sales_metric = models['sales_metrics'][
                models['sales_metrics']['Model'] == sales_model
            ]
            if not sales_metric.empty:
                st.caption(f"R²: {sales_metric['R2'].values[0]:.3f} | RMSE: {sales_metric['RMSE'].values[0]:.0f}")
        
        with col2:
            st.markdown("**Quantity Prediction**")
            qty_model = st.selectbox(
                "Select Model",
                options=list(available_qty_models.keys()),
                index=0,
                key="qty_model"
            )
            
            # Show model metrics
            qty_metric = models['qty_metrics'][
                models['qty_metrics']['Model'] == qty_model
            ]
            if not qty_metric.empty:
                st.caption(f"R²: {qty_metric['R2'].values[0]:.3f} | RMSE: {qty_metric['RMSE'].values[0]:.0f}")
        
        st.markdown("---")
        
        # Controls
        st.subheader("⚙️ Controls")
        
        k_results = st.slider(
            "Number of Results",
            min_value=3,
            max_value=10,
            value=5,
            help="Select how many similar cars to display"
        )
        
        # Price range filter
        st.subheader("💰 Price Filter")
        min_price = st.number_input("Min Price (Rp)", min_value=0, value=0, step=1000000, format="%d")
        max_price = st.number_input("Max Price (Rp)", min_value=0, value=1000000000, step=1000000, format="%d")
        
        price_range = (min_price, max_price) if min_price < max_price else None
        
        # Toggle sales overlay
        show_sales = st.toggle("📊 Show Sales Data", value=True)
        
        # Show model comparison
        show_comparison = st.toggle("📈 Show Model Comparison", value=False)
        
        st.markdown("---")
        st.caption("Built with ❤️ using Vision Transformer + FAISS")
        
        return uploaded_file, k_results, price_range, show_sales, show_comparison, sales_model, qty_model

def display_query_image(image_path):
    """Display the query image"""
    image = Image.open(image_path).convert('RGB')
    st.image(image, caption="🔍 Query Image", use_container_width=True)

def display_metrics(results, sales_data):
    """Display KPI metrics"""
    
    # Calculate metrics
    valid_results = [r for r in results if r['has_sales']]
    
    if valid_results:
        avg_price = np.mean([r['avg_price'] for r in valid_results])
        total_sales = sum([r['total_quantity'] for r in valid_results])
        avg_margin = np.mean([r['profit_margin'] for r in valid_results])
        
        # Top revenue brand
        top_brand = max(valid_results, key=lambda x: x['total_sales'])
    else:
        avg_price = 0
        total_sales = 0
        avg_margin = 0
        top_brand = {'brand': 'N/A'}
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Average Price",
            f"Rp{avg_price/1_000_000:,.1f}M",
            help="Mean price across matched recommendations"
        )
    
    with col2:
        st.metric(
            "📦 Total Sales",
            f"{total_sales:,.0f} units",
            help="Total units sold for recommended brands"
        )
    
    with col3:
        st.metric(
            "📈 Avg Profit Margin",
            f"{avg_margin*100:.1f}%",
            help="Average margin from real transaction logs"
        )
    
    with col4:
        st.metric(
            "🏆 Top Revenue Brand",
            top_brand.get('brand', 'N/A'),
            help="Highest grossing brand in recommendations"
        )

def display_model_predictions(result):
    """Display predictions from selected models"""
    
    if 'predictions' not in result:
        return
    
    predictions = result['predictions']
    
    st.markdown("**Model Predictions:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("📊 Sales Prediction")
        for model_name, pred in predictions['sales'].items():
            if pred:
                st.metric(model_name, f"{pred:,.0f}", delta_color="off")
    
    with col2:
        st.markdown("📦 Quantity Prediction")
        for model_name, pred in predictions['quantity'].items():
            if pred:
                st.metric(model_name, f"{pred:,.0f}", delta_color="off")

def display_recommendation_cards(results, show_sales, selected_sales_model, selected_qty_model):
    """Display recommendations as image cards with model predictions"""
    
    cols = st.columns(min(len(results), 4))
    
    for idx, (col, result) in enumerate(zip(cols, results)):
        with col:
            # Display image
            try:
                img = Image.open(result['path']).convert('RGB')
                st.image(img, use_container_width=True)
            except:
                st.image("https://via.placeholder.com/200x150?text=No+Image", use_container_width=True)
            
            # Brand name
            st.markdown(f"**{result['brand']}**")
            
            # Similarity score with color
            score = result['similarity']
            color = "green" if score > 0.70 else "orange" if score > 0.50 else "red"
            st.markdown(f"🟢 Similarity: `{score:.2%}`" if score > 0.70 else 
                       f"🟠 Similarity: `{score:.2%}`" if score > 0.50 else 
                       f"🔴 Similarity: `{score:.2%}`")
            
            # Sales data badges
            if show_sales and result['has_sales']:
                if result.get('top_model'):
                    st.markdown(f"🏷️ **Top Model:** {result['top_model']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sales", f"{result['total_quantity']:,.0f}", delta_color="off")
                with col2:
                    st.metric("Price", f"Rp{result['avg_price']/1_000_000:,.0f}M", delta_color="off")
                
                # Display model predictions
                if 'predictions' in result:
                    st.markdown("---")
                    st.markdown("**📊 Predictions**")
                    
                    # Sales prediction
                    sales_pred = result['predictions']['sales'].get(selected_sales_model)
                    if sales_pred:
                        st.metric(f"Sales ({selected_sales_model})", f"{sales_pred:,.0f}", delta_color="off")
                    
                    # Quantity prediction
                    qty_pred = result['predictions']['quantity'].get(selected_qty_model)
                    if qty_pred:
                        st.metric(f"Qty ({selected_qty_model})", f"{qty_pred:,.0f}", delta_color="off")
                
                # Sales status badge
                st.markdown("✅ **Sales Data Available**")
            else:
                st.caption("⚠️ No sales data available")

def display_model_comparison(results, selected_sales_model, selected_qty_model):
    """Display model comparison charts"""
    
    st.subheader("📊 Model Prediction Comparison")
    
    # Filter results with predictions
    valid_results = [r for r in results if 'predictions' in r and r['has_sales']]
    
    if not valid_results:
        st.info("No prediction data available for comparison")
        return
    
    # Prepare data for comparison
    brands = [r['brand'] for r in valid_results]
    
    # Sales predictions comparison
    fig1 = go.Figure()
    
    for model_name in available_sales_models.keys():
        predictions = []
        for r in valid_results:
            pred = r['predictions']['sales'].get(model_name)
            predictions.append(pred if pred else 0)
        
        fig1.add_trace(go.Bar(
            name=model_name,
            x=brands,
            y=predictions,
            text=[f"{p:,.0f}" if p else 'N/A' for p in predictions],
            textposition='auto'
        ))
    
    fig1.update_layout(
        title="Sales Predictions by Model",
        xaxis_title="Brand",
        yaxis_title="Predicted Sales",
        barmode='group',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Quantity predictions comparison
    fig2 = go.Figure()
    
    for model_name in available_qty_models.keys():
        predictions = []
        for r in valid_results:
            pred = r['predictions']['quantity'].get(model_name)
            predictions.append(pred if pred else 0)
        
        fig2.add_trace(go.Bar(
            name=model_name,
            x=brands,
            y=predictions,
            text=[f"{p:,.0f}" if p else 'N/A' for p in predictions],
            textposition='auto'
        ))
    
    fig2.update_layout(
        title="Quantity Predictions by Model",
        xaxis_title="Brand",
        yaxis_title="Predicted Quantity",
        barmode='group',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Model metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Model Metrics")
        st.dataframe(models['sales_metrics'][['Model', 'R2', 'RMSE', 'MAE']], use_container_width=True)
    
    with col2:
        st.subheader("Quantity Model Metrics")
        st.dataframe(models['qty_metrics'][['Model', 'R2', 'RMSE', 'MAE']], use_container_width=True)

def display_analytics_charts(results, sales_data, selected_sales_model):
    # Fixed specs: Set secondary_y: True for row 2, col 1 and domain for row 2, col 2
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=(
            "Similarity vs Price", 
            "Actual vs Predicted Sales", 
            "Profit Margin & Volume", 
            "Transaction Distribution"
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "secondary_y": True}, {"type": "domain"}]
        ]
    )
    
    # Filter valid results
    valid_results = [r for r in results if r['has_sales']]
    
    if valid_results:
        # 1. Similarity vs Price (Row 1, Col 1)
        brands = [r['brand'] for r in valid_results]
        similarities = [r['similarity'] for r in valid_results]
        prices = [r['avg_price'] / 1_000_000 for r in valid_results]
        
        fig.add_trace(
            go.Scatter(
                x=prices,
                y=similarities,
                mode='markers+text',
                text=brands,
                textposition="top center",
                marker=dict(size=15, color=similarities, colorscale='Viridis'),
                name='Similarity vs Price'
            ),
            row=1, col=1
        )
        
        # 2. Actual vs Predicted (Row 1, Col 2)
        actual_sales = [r['total_quantity'] for r in valid_results]
        predicted_sales = []
        for r in valid_results:
            if 'predictions' in r:
                pred = r['predictions']['sales'].get(selected_sales_model)
                predicted_sales.append(pred if pred else 0)
            else:
                predicted_sales.append(0)
        
        fig.add_trace(
            go.Bar(x=brands, y=actual_sales, name='Actual Sales', marker_color='lightcoral'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=brands, y=predicted_sales, name=f'Predicted ({selected_sales_model})', marker_color='lightblue'),
            row=1, col=2
        )
        
        # 3. Profit Margin & Sales Volume (Row 2, Col 1)
        margins = [r['profit_margin'] * 100 for r in valid_results]
        volumes = [r['total_quantity'] for r in valid_results]
        
        # Add primary trace (Profit Margin %)
        fig.add_trace(
            go.Scatter(
                x=brands,
                y=margins,
                mode='lines+markers',
                name='Profit Margin %',
                marker=dict(size=10),
                line=dict(color='green')
            ),
            row=2, col=1, secondary_y=False
        )
        
        # Add secondary trace (Sales Volume)
        fig.add_trace(
            go.Bar(
                x=brands,
                y=volumes,
                name='Sales Volume',
                marker_color='orange'
            ),
            row=2, col=1, secondary_y=True
        )
        
        # 4. Transaction distribution (Row 2, Col 2 - Pie Chart)
        transactions = [r['transaction_count'] for r in valid_results]
        
        fig.add_trace(
            go.Pie(
                labels=brands,
                values=transactions,
                name='Transaction Distribution'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        template='plotly_white'
    )
    
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
# 5. MAIN APP
# ============================================

def main():
    # Sidebar
    uploaded_file, k_results, price_range, show_sales, show_comparison, selected_sales_model, selected_qty_model = sidebar_controls()
    
    # Main content
    st.title("🚗 Car Vision & Sales Intelligence Dashboard")
    st.markdown("---")
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        
        # Get recommendations
        with st.spinner("🔍 Finding similar cars..."):
            results = get_recommendations(
                temp_path, 
                k=k_results, 
                price_range=price_range,
                selected_models=True
            )
        
        if results:
            # Display query image and metrics
            col1, col2 = st.columns([1, 3])
            
            with col1:
                display_query_image(temp_path)
            
            with col2:
                st.subheader("📊 Key Metrics")
                display_metrics(results, sales_data)
            
            st.markdown("---")
            
            # Display recommendation cards
            st.subheader("🖼️ Top Recommendations")
            display_recommendation_cards(results, show_sales, selected_sales_model, selected_qty_model)
            
            st.markdown("---")
            
            # Display model comparison if enabled
            if show_comparison:
                display_model_comparison(results, selected_sales_model, selected_qty_model)
                st.markdown("---")
            
            # Display analytics
            if show_sales:
                display_analytics_charts(results, sales_data, selected_sales_model)
                st.markdown("---")
                display_transaction_data(results, sales_data, selected_sales_model, selected_qty_model)
        else:
            st.warning("No recommendations found. Try adjusting the filters.")
        
        # Cleanup
        os.unlink(temp_path)
        
    else:
        # Welcome screen
        st.info("👈 Please upload a car image from the sidebar to get started!")
        
        # Show model information
        with st.expander("📊 Model Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sales Prediction Models")
                st.dataframe(models['sales_metrics'][['Model', 'R2', 'RMSE', 'MAE']], use_container_width=True)
            
            with col2:
                st.subheader("Quantity Prediction Models")
                st.dataframe(models['qty_metrics'][['Model', 'R2', 'RMSE', 'MAE']], use_container_width=True)
        
        # Show sample data preview
        with st.expander("📊 Preview Sales Data"):
            st.dataframe(sales_data.head(10), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", f"{len(sales_data):,}")
            with col2:
                st.metric("Total Sales", f"Rp{sales_data['sales'].sum():,.0f}")

if __name__ == "__main__":
    main()