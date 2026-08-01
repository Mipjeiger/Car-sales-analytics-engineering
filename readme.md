## 🏎️ Project structure for enterprise Car analytics Engineering

- Build PostgreSQL database for Car sales projection
    ![alt text](development/database/images/F89AEB6E-A1BC-492B-A2B6-36E768EAA494.png)
- Metabase as Data Visualization to ensure data analytics projection
    - Sum of Sales and Sum of Profit by Date Month
    ![alt text](development/database/images/7BDB70CE-D9DC-4C65-BCAE-FDA87DD4A757.png)
    - Sum of Sales and Sum of Profit by Date Week
    ![alt text](development/database/images/3E42B322-15CA-4F30-8E38-F763A477F534.png)
    - Sum of Sales and Sum of Profit by Dealer Region
    ![alt text](development/database/images/4A9CB78D-E566-4968-83BB-67C679E253F0.png)
- ML Prediction divide into 2 model
    - Model 1 - Demand Prediction
    Business Question:
    How many cars will a customer purchase?
    - Model 2 - Revenue Forecasting
- Sales prediction on visual graph data visualization
    - Comparison ML models for sales prediction
    ![alt text](development/database/images/CB069150-EDDB-4346-A4C9-A7CD245995AB_1_105_c.jpeg)
- Streamlit UI production to get car recommendation system AI Dashboard
    ![alt text](development/database/images/1FBC6837-7934-43D3-B330-C4F01CBCF1BD_1_105_c.jpeg)
    ![alt text](development/database/images/1A097914-282B-4459-B464-68F6FC2841EF_1_105_c.jpeg)
    ![alt text](development/database/images/E70EAEE7-DBD6-4A8B-AF02-12C41F0EEF3E_1_105_c.jpeg)
    ![alt text](development/database/images/C7723BB0-8CFE-4C7E-85B9-1F9834F443A8_1_105_c.jpeg)
- DVC (Data Version Control)
    ⚙️ DVC ensures data and pipeline determinism
    ![alt text](development/database/images/D22ECAF4-20A6-40F0-93AE-DE9FA358F1E4.png)
- Minio as S3 MLflow models storage
    MLFLow models storage
    ![alt text](development/database/images/E1176D1B-9384-4404-B1BF-BB4AD18C72DA_1_105_c.jpeg)
    DVC data storage
    ![alt text](<development/database/images/Screenshot 2026-08-01 at 22.11.06.png>)
- MLFlow for experiment tracking and registry models
    📈 Chart metrics model in MLflow
    ![alt text](development/database/images/A0CAF509-0154-4347-952A-44EAC594E7E4_1_105_c.jpeg)
    📊 Parallel coordinat in 9 models comparison with each models metrics
    ![alt text](development/database/images/90417958-A154-4D14-A741-6573BF614ED9_1_105_c.jpeg)
- Build computer vision deep learning for car recommendation system
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🚗 COMPUTER VISION CAR RECOMMENDATION SYSTEM           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

✅ SYSTEM COMPONENTS:
────────────────────
1. Vision Transformer Feature Extractor
2. FAISS Index for Fast Similarity Search
3. Integration with Existing Prediction Models
4. Enhanced Recommendation with Sales Data
5. Production-Ready API Interface

📊 SYSTEM STATISTICS:
────────────────────
- Total Images Processed: {len(image_features)}
- Feature Dimension: {image_features.shape[1]}
- Car Brands: {len(feature_df['brand'].unique())}
- FAISS Index Size: {similarity_search.index.ntotal} vectors

🎯 KEY FEATURES:
────────────────────
1. Similar Car Search by Image
2. Visual Similarity Scoring
3. Sales Data Integration
4. Price and Sales Predictions
5. Brand Mapping and Analysis

📁 EXPORTED MODELS:
────────────────────
Location: {cv_export_dir}
- car_index.faiss (FAISS index)
- feature_data.csv (Image metadata)
- feature_matrix.npy (Image embeddings)
- metadata.json (System metadata)
- brand_mapping.json (Brand mappings)

🚀 PRODUCTION USAGE:
────────────────────
from production_recommender import ProductionCarRecommender

# Initialize
recommender = ProductionCarRecommender()

# Get recommendations
results = recommender.recommend('car_image.jpg', k=5)

# Display results
for result in results:
    print(f"Brand: {result['brand']}")
    print(f"Similarity: {result['similarity']:.3f}")

═══════════════════════════════════════════════════════════════


🔍 Quick Validation:
==================================================
✅ Test successful! Found 3 recommendations
   Top brand: Tata_Safari
   Similarity: 0.767

✅ System ready for deployment!