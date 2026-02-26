import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Charger le CSV
df = pd.read_csv("../creditcard.csv")

# Colonnes features
features_columns = [col for col in df.columns if col.startswith('V')] + ['Amount']

X = df[features_columns]
y = df['Class']

corr_matrix = X.corr().abs()

# Supprimer la diagonale
corr_no_diag = corr_matrix.where(
    ~np.eye(corr_matrix.shape[0], dtype=bool)
)

top_corr = (
    corr_no_diag
    .stack()
    .sort_values(ascending=False)
    .head(10)
)

print(top_corr)


"""
# Modèle pour importance
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Récupérer les importances
importances = pd.Series(model.feature_importances_, index=features_columns)
selected_features = importances[importances > 0.02].index  
print("Features sélectionnées :", selected_features)

#df_selected = df[selected_features]
#df_selected.to_csv("transactions_features_selected.csv", index=False)
"""
