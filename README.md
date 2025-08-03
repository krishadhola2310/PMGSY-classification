#  PMGSY Scheme Classification using ML

This project classifies **PMGSY (Pradhan Mantri Gram Sadak Yojana)** schemes using a **Random Forest Classifier**, with **SMOTE** for class balancing and **StandardScaler** for feature normalization.

## ✅ Key Highlights
- **Model**: Random Forest (with default hyperparameters)
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling)
- **Feature Scaling**: StandardScaler
- **Evaluation**:
  - Training Accuracy: ~XX%
  - Testing Accuracy: ~XX%
  - 7-Fold Cross-Validation: ~XX%
- **Artifacts Saved**:  
  `pmgsy_rf_model.pkl`, `scaler.pkl`, `label_encoder.pkl`

## 📊 Visualizations
- Confusion Matrix Heatmap  
- Actual vs Predicted Class Count (Bar Graph)  
- Top 10 Feature Importances

## 📁 Files Included
- `PMGSY_Classification.ipynb` – Full ML pipeline (Google Colab)
- `PMGSY_DATASET.csv` – Input dataset
- `.pkl` files – Trained model + scaler + label encoder

## 💻 Platform
- Developed on **Google Colab**  
- Compatible with **IBM WatsonX Studio** (upload `.ipynb` as local notebook)

