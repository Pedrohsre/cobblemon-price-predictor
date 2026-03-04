import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Cobblemon Price Predictor",
    page_icon="🦞",
    layout="wide"
)

# Título
st.title(" Scizor Gauntlets Price Predictor")
st.markdown("### Machine Learning price prediction for **play.cobblemondelta.com**")

# Função para calcular DPS
def calcular_dps(atk_base, atk_speed_percent, sharpness_level, base_aps=1.6):
    sharpness_bonus = 0.5 * sharpness_level + 1.0
    fist_bonus = 0.5
    atk_efetivo = atk_base + sharpness_bonus + fist_bonus
    
    multiplier = 1 + (atk_speed_percent / 100)
    aps = base_aps * multiplier
    dano_medio_por_hit = atk_efetivo * (5 / 3)
    dps_com_passiva = aps * dano_medio_por_hit
    
    return round(dps_com_passiva, 2)

# Dataset
@st.cache_data
def load_data():
    data = [
        {'sharp': 4, 'ad': 6.94, 'as': 133.21, 'mending': 0, 'price': 1500000},
        {'sharp': 4, 'ad': 6.7, 'as': 142.69, 'mending': 0, 'price': 1500000},
        {'sharp': 4, 'ad': 6.88, 'as': 130.56, 'mending': 0, 'price': 1200000},
        {'sharp': 4, 'ad': 6.95, 'as': 120.56, 'mending': 0, 'price': 1200000},
        {'sharp': 4, 'ad': 6.89, 'as': 127.33, 'mending': 0, 'price': 1100000},
        {'sharp': 5, 'ad': 5.86, 'as': 132.22, 'mending': 0, 'price': 670000},
        {'sharp': 4, 'ad': 6.55, 'as': 148.29, 'mending': 1, 'price': 3000000},
        {'sharp': 4, 'ad': 6.83, 'as': 128.38, 'mending': 1, 'price': 2250000},
        {'sharp': 5, 'ad': 6.0, 'as': 122.14, 'mending': 0, 'price': 1250000},
        {'sharp': 4, 'ad': 6.36, 'as': 146.13, 'mending': 0, 'price': 900000},
        {'sharp': 5, 'ad': 5.89, 'as': 122.67, 'mending': 0, 'price': 850000},
        {'sharp': 5, 'ad': 6.23, 'as': 129.71, 'mending': 0, 'price': 1200000},
        {'sharp': 4, 'ad': 6.6, 'as': 120.88, 'mending': 1, 'price': 1900000},
        {'sharp': 5, 'ad': 5.9, 'as': 149.08, 'mending': 0, 'price': 1500000},
        {'sharp': 4, 'ad': 6.25, 'as': 125.19, 'mending': 0, 'price': 650000},
        {'sharp': 4, 'ad': 6.59, 'as': 129.08, 'mending': 0, 'price': 1000000},
        {'sharp': 5, 'ad': 6.08, 'as': 147.43, 'mending': 0, 'price': 1200000},
        {'sharp': 4, 'ad': 5.82, 'as': 127.41, 'mending': 1, 'price': 1000000},
        {'sharp': 4, 'ad': 6.61, 'as': 131.73, 'mending': 1, 'price': 2000000},
        {'sharp': 5, 'ad': 5.12, 'as': 140.45, 'mending': 1, 'price': 1000000},
        {'sharp': 5, 'ad': 6.32, 'as': 140.14, 'mending': 1, 'price': 2500000},
        {'sharp': 4, 'ad': 6.27, 'as': 129.04, 'mending': 0, 'price': 800000},
        {'sharp': 5, 'ad': 6.88, 'as': 147.18, 'mending': 0, 'price': 5000000},
        {'sharp': 4, 'ad': 5.79, 'as': 134.55, 'mending': 1, 'price': 900000},
        {'sharp': 4, 'ad': 6.43, 'as': 123.55, 'mending': 0, 'price': 550000},
        {'sharp': 4, 'ad': 6.43, 'as': 126.55, 'mending': 0, 'price': 750000},
        {'sharp': 5, 'ad': 5.7, 'as': 143.49, 'mending': 0, 'price': 900000},
        {'sharp': 4, 'ad': 6.81, 'as': 132.66, 'mending': 0, 'price': 1200000},
        {'sharp': 5, 'ad': 6.24, 'as': 121.02, 'mending': 0, 'price': 800000},
        {'sharp': 4, 'ad': 5.71, 'as': 140.39, 'mending': 1, 'price': 900000},
        {'sharp': 4, 'ad': 6.9, 'as': 129.14, 'mending': 0, 'price': 1000000},
        {'sharp': 5, 'ad': 6.65, 'as': 127.93, 'mending': 0, 'price': 1200000},
        {'sharp': 5, 'ad': 6.82, 'as': 147.58, 'mending': 1, 'price': 8500000},
        {'sharp': 5, 'ad': 6.43, 'as': 140.38, 'mending': 1, 'price': 4500000},
        {'sharp': 5, 'ad': 6.45, 'as': 146.00, 'mending': 1, 'price': 6000000},
        {'sharp': 5, 'ad': 5.46, 'as': 127.89, 'mending': 1, 'price': 1000000},
        {'sharp': 4, 'ad': 6.27, 'as': 125.69, 'mending': 0, 'price': 570000},
        {'sharp': 5, 'ad': 6.65, 'as': 141.06, 'mending': 0, 'price': 2000000},
        {'sharp': 4, 'ad': 7.00, 'as': 147.34, 'mending': 0, 'price': 4500000},
        {'sharp': 5, 'ad': 6.61, 'as': 131.05, 'mending': 1, 'price': 2100000},
        {'sharp': 5, 'ad': 6.09, 'as': 137.25, 'mending': 0, 'price': 800000},
        {'sharp': 4, 'ad': 5.28, 'as': 134.85, 'mending': 1, 'price': 1200000},
        {'sharp': 5, 'ad': 6.09, 'as': 137.25, 'mending': 0, 'price': 800000},
        {'sharp': 4, 'ad': 5.28, 'as': 134.85, 'mending': 1, 'price': 1200000},
        {'sharp': 4, 'ad': 5.08, 'as': 132.7, 'mending': 1, 'price': 700000},
        {'sharp': 4, 'ad': 5.35, 'as': 142.5, 'mending': 1, 'price': 800000},
        {'sharp': 4, 'ad': 5.32, 'as': 139.2, 'mending': 0, 'price': 800000},
        {'sharp': 4, 'ad': 5.29, 'as': 129.2, 'mending': 0, 'price': 600000},
        {'sharp': 4, 'ad': 6.91, 'as': 128.3, 'mending': 1, 'price': 2750000},
        {'sharp': 4, 'ad': 5.84, 'as': 129.6, 'mending': 1, 'price': 850000},
        {'sharp': 4, 'ad': 6, 'as': 128.1, 'mending': 0, 'price': 750000},
        {'sharp': 5, 'ad': 5, 'as': 149.2, 'mending': 0, 'price': 550000},
        {'sharp': 4, 'ad': 6.02, 'as': 136.7, 'mending': 0, 'price': 800000},
        {'sharp': 4, 'ad': 6.96, 'as': 132, 'mending': 1, 'price': 2700000},
        {'sharp': 4, 'ad': 6.42, 'as': 132.8, 'mending': 0, 'price': 900000},
        {'sharp': 5, 'ad': 6.68, 'as': 136, 'mending': 1, 'price': 7000000},
        {'sharp': 5, 'ad': 6.68, 'as': 128, 'mending': 1, 'price': 7250000},
        {'sharp': 4, 'ad': 7, 'as': 138.09, 'mending': 0, 'price': 2500000},
        {'sharp': 5, 'ad': 6.72, 'as': 146.8, 'mending': 1, 'price': 8000000},
    ]
    
    df = pd.DataFrame(data)
    df['dps'] = df.apply(lambda row: calcular_dps(row['ad'], row['as'], row['sharp']), axis=1)
    return df

# Treinar modelos
@st.cache_resource
def train_models(df):
    features = ['dps', 'mending']
    X = df[features]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    return models, results

# Carregar dados e treinar modelos
df = load_data()
models, results = train_models(df)

# Sidebar - Inputs
st.sidebar.header("Item Stats")
st.sidebar.markdown("Enter the Scizor Gauntlets attributes:")

sharp = st.sidebar.selectbox("Sharpness", [4, 5], index=1)
ad = st.sidebar.number_input("Attack Damage (AD)", min_value=5.0, max_value=7.5, value=6.5, step=0.01)
as_value = st.sidebar.number_input("Attack Speed (AS%)", min_value=120.0, max_value=150.0, value=140.0, step=0.1)
mending = st.sidebar.checkbox("Mending", value=True)

# Calcular DPS
dps = calcular_dps(ad, as_value, sharp)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Calculated DPS:** `{dps}`")

# Fazer previsões
new_item = pd.DataFrame({
    'dps': [dps],
    'mending': [1 if mending else 0]
})

predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(new_item)[0]

# Exibir resultados
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Price Predictions")
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    bars = ax.bar(predictions.keys(), predictions.values(), color=colors)
    
    media = np.mean(list(predictions.values()))
    ax.axhline(y=media, color='purple', linestyle='--', linewidth=2, label=f'Average: PD {media:,.0f}')
    
    ax.set_ylabel('Predicted Price (PD)', fontsize=12)
    ax.set_title('Model Predictions Comparison', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'PD {height:,.0f}',
                ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig)

with col2:
    st.subheader("💰 Results")
    
    for name, price in predictions.items():
        st.metric(
            label=name,
            value=f"PD {price:,.0f}",
            delta=f"{((price - media) / media * 100):+.1f}%"
        )
    
    st.markdown("---")
    st.metric("Average Price", f"PD {media:,.0f}")
    st.metric("Std Deviation", f"PD {np.std(list(predictions.values())):,.0f}")

# Seção de métricas dos modelos
st.markdown("---")
st.subheader("Model Performance Metrics")

metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[name]['rmse'] for name in results.keys()],
    'MAE': [results[name]['mae'] for name in results.keys()],
    'R²': [results[name]['r2'] for name in results.keys()]
})

st.dataframe(
    metrics_df.style.format({
        'RMSE': 'PD {:,.0f}',
        'MAE': 'PD {:,.0f}',
        'R²': '{:.4f}'
    }).background_gradient(subset=['R²'], cmap='Greens'),
    use_container_width=True
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with Machine Learning | Dataset: 59 historical items</p>
        <p>Server: <strong>play.cobblemondelta.com</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)
