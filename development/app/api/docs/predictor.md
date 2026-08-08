Two specific logic issues in predictor.py caused the high output:

  1. car_id Match Missing in Reference Row Resolution:
  _find_reference_row() only checked "model": "Expedition", which took the median across all ~500 Expedition records in the database (where rolling means and standard deviations reach up to 2.11 Billion). It ignored the specific
  "car_id": "C_CND_000001" passed in the payload.
  2. Feature Mutation of gross_sales:
  In predict(), when quantity predicted 6, the method executed:
  features["gross_sales"] = price * quantity
  This multiplied the user-supplied gross_sales (467,740,000) by 6, mutating gross_sales to 2,806,440,000 (2.8 Billion) before passing it to the Sales Model. The Sales Model then predicted 2,491,160,644.85.
  ──────
  ### Solution Implemented

  Updated predictor.py:

  1. Prioritized car_id Match:
  Updated _find_reference_row() to check for "car_id" match first before falling back to "model". Matching "C_CND_000001" directly pulls the exact rolling statistics for that car.
  2. Preserved Explicit gross_sales:
  Updated predict() to preserve user-provided gross_sales and profit values rather than multiplying them by predicted_quantity.
  ──────
  ### Verification Output for Your Exact JSON Request

  Running your payload again through the updated API:

    {
        "model_type": "both",
        "predicted_sales": 455316953.15,
        "sales_confidence": 0.99,
        "predicted_quantity": 2,
        "quantity_confidence": 0.559
    }

  • Actual Sales: 444,353,000.00
  • Previous Output: 2,491,160,644.85 (2.49 Billion)
  • New Output: 455,316,953.15 (Gap reduced from >2 Billion down to 2.4%)