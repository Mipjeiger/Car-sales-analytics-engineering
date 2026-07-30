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

## 📝 Important notes
- Demand Prediction
    Your current R² (~0.34) suggests there is room for improvement. I would:
- Apply a log transform to `income_customer` if it's highly skewed.
- Consider treating high-cardinality categorical features (`dealer_name`, `model`) with CatBoost's native categorical handling rather than label encoding.
- Add richer business features if available (financing type, dealership inventory, holiday indicators, prior customer purchases).

- Revenue Forecasting
    Decide which business problem you want to solve:
- **Forecast future revenue**: remove features that wouldn't be known at prediction time (`quantity`, `customers`, `avg_price`, `avg_discount`) and rely on historical lags and calendar information.
- **Estimate revenue during operations**: your current feature set is appropriate, and the very high R² is expected because `quantity` and `customers` are strong real-time indicators of revenue.