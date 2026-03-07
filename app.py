import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
from github import Github
import base64

# ============================================================================
# GITHUB INTEGRATION
# ============================================================================

@st.cache_resource
def connect_to_github():
    """Conecta ao GitHub usando token do Streamlit Secrets"""
    try:
        # Verificar se há configuração do GitHub
        if "github_token" in st.secrets:
            token = st.secrets["github_token"]
            g = Github(token)
            
            # Pegar repo e arquivo configurados
            repo_name = st.secrets.get("github_repo", "")
            file_path = st.secrets.get("github_file_path", "user_submissions.json")
            
            if repo_name:
                repo = g.get_repo(repo_name)
                return {"github": g, "repo": repo, "file_path": file_path}
        
        return None
    except Exception as e:
        st.warning(f"GitHub não configurado: {e}")
        return None

def load_from_github(github_config):
    """Carrega dados do arquivo JSON no GitHub"""
    try:
        if not github_config:
            return []
        
        repo = github_config["repo"]
        file_path = github_config["file_path"]
        
        # Tentar obter o arquivo
        try:
            file_content = repo.get_contents(file_path)
            content = base64.b64decode(file_content.content).decode('utf-8')
            return json.loads(content)
        except Exception:
            # Arquivo não existe ainda
            return []
            
    except Exception as e:
        st.warning(f"GitHub não configurado: {e}")
        return []

def save_to_github(github_config, data, commit_message="Add new submission"):
    """Salva dados no arquivo JSON no GitHub"""
    try:
        if not github_config:
            return False
        
        repo = github_config["repo"]
        file_path = github_config["file_path"]
        
        # Converter dados para JSON
        json_content = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Verificar se arquivo já existe
        try:
            file_content = repo.get_contents(file_path)
            # Atualizar arquivo existente
            repo.update_file(
                file_path,
                commit_message,
                json_content,
                file_content.sha
            )
        except Exception:
            # Criar novo arquivo
            repo.create_file(
                file_path,
                commit_message,
                json_content
            )
        
        return True
        
    except Exception as e:
        st.error(f"Erro ao salvar no GitHub: {e}")
        return False

# Conectar ao GitHub (ou None se não configurado)
github_config = connect_to_github()
if github_config:
    st.sidebar.success("GitHub conectado")
else:
    st.sidebar.info("ℹUsando armazenamento local (JSON)")

# ============================================================================

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
def load_data():
    # Dataset base (histórico)
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
    
    # Carregar dados submetidos pelos usuários
    user_data = []
    
    # Tentar carregar do GitHub primeiro
    if github_config:
        try:
            user_data = load_from_github(github_config)
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar do GitHub: {e}")
    
    # Fallback para JSON local se GitHub não estiver configurado ou falhar
    if not user_data:
        database_file = 'user_submissions.json'
        if os.path.exists(database_file):
            try:
                with open(database_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
            except Exception as e:
                st.warning(f"⚠️ Não foi possível carregar dados do JSON: {str(e)}")
    
    # Adicionar dados dos usuários ao dataset
    if user_data:
        try:
            # Extrair apenas as colunas necessárias para o treinamento
            user_df = pd.DataFrame([{
                'sharp': item['sharp'],
                'ad': item['ad'],
                'as': item['as'],
                'mending': item['mending'],
                'price': item['price']
            } for item in user_data])
            
            # Combinar com o dataset base
            df = pd.concat([df, user_df], ignore_index=True)
        except Exception as e:
            st.warning(f"⚠️ Erro ao processar dados dos usuários: {str(e)}")
    
    # Calcular DPS para todos os dados
    df['dps'] = df.apply(lambda row: calcular_dps(row['ad'], row['as'], row['sharp']), axis=1)
    return df

# Treinar modelos
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
ad = st.sidebar.number_input("Attack Damage (AD)", min_value=5.0, max_value=7.0, value=6.5, step=0.01)
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

# Seção das últimas 10 adições
#st.markdown("---")
#st.subheader("🔥 Recent Community Submissions")

# database_file = 'user_submissions.json'
# if os.path.exists(database_file):
#     with open(database_file, 'r', encoding='utf-8') as f:
#         submissions = json.load(f)
#     
#     if submissions:
#         # Pegar as últimas 10 submissões (mais recentes primeiro)
#         recent_submissions = submissions[-10:][::-1]
#         
#         # Criar um card para cada submissão
#         for idx, item in enumerate(recent_submissions, 1):
#             with st.container():
#                 col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
#                 
#                 with col1:
#                     st.markdown(f"**👤 {item['submitted_by']}**")
#                     st.caption(item['submitted_at'])
#                 
#                 with col2:
#                     mending_emoji = "✨" if item['mending'] == 1 else "❌"
#                     st.markdown(f"Sharp: **{item['sharp']}** | AD: **{item['ad']}** | AS: **{item['as']}%** | Mend: {mending_emoji}")
#                 
#                 with col3:
#                     st.markdown(f"DPS: **{item.get('dps', 'N/A')}**")
#                 
#                 with col4:
#                     st.markdown(f"💰 **PD {item['price']:,.0f}**")
#                 
#                 if idx < len(recent_submissions):
#                     st.divider()
#     else:
#         st.info("No submissions yet. Be the first to contribute!")
# else:
#     st.info("No submissions yet. Be the first to contribute!")

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
    width='stretch'
)

# Seção para adicionar novos dados ao database
st.markdown("---")
st.subheader("➕ Add New Item to Database")
st.markdown("Contribute to the dataset by adding new item sales data")

with st.expander("📝 Submit New Item Data", expanded=False):
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Item Attributes:**")
        new_sharp = st.selectbox("Sharpness Level", [4, 5], key="new_sharp")
        new_ad = st.number_input("Attack Damage (AD)", min_value=5.0, max_value=7.0, value=6.0, step=0.01, key="new_ad")
        new_as = st.number_input("Attack Speed (AS%)", min_value=120.0, max_value=150.0, value=130.0, step=0.1, key="new_as")
        new_mending = st.checkbox("Has Mending", key="new_mending")
        new_price = st.number_input("Sale Price (PD)", min_value=0, max_value=100000000, value=1000000, step=50000, key="new_price")
    
    with col_b:
        st.markdown("**Contributor Information:**")
        nickname = st.text_input("Your Nickname/IGN", placeholder="Enter your name or in-game name", key="nickname")
        
        # Calcular DPS do novo item
        new_dps = calcular_dps(new_ad, new_as, new_sharp)
        st.info(f"**Calculated DPS:** {new_dps}")
        
        st.markdown("---")
        
        if st.button("💾 Submit to Database", type="primary", use_container_width=True):
            if nickname.strip() == "":
                st.error("⚠️ Please enter your nickname before submitting!")
            else:
                # Preparar os dados para salvar
                new_entry = {
                    'sharp': new_sharp,
                    'ad': new_ad,
                    'as': new_as,
                    'mending': 1 if new_mending else 0,
                    'price': new_price,
                    'dps': new_dps,
                    'submitted_by': nickname.strip(),
                    'submitted_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                success = False
                
                # Tentar salvar no GitHub primeiro
                if github_config:
                    try:
                        # Carregar dados existentes do GitHub
                        submissions = load_from_github(github_config)
                        if submissions is None:
                            submissions = []
                        
                        # Adicionar nova entrada
                        submissions.append(new_entry)
                        
                        # Salvar no GitHub
                        commit_msg = f"Add submission from {nickname.strip()}"
                        if save_to_github(github_config, submissions, commit_msg):
                            st.success(f"✅ Data successfully added to GitHub!\n\nSubmitted by: **{nickname}**\nDateTime: **{new_entry['submitted_at']}**")
                            success = True
                    except Exception as e:
                        st.warning(f"⚠️ Erro ao salvar no GitHub: {e}")
                
                # Fallback para JSON local se GitHub não estiver configurado ou falhar
                if not success:
                    database_file = 'user_submissions.json'
                    try:
                        # Carregar dados existentes ou criar lista vazia
                        if os.path.exists(database_file):
                            with open(database_file, 'r', encoding='utf-8') as f:
                                submissions = json.load(f)
                        else:
                            submissions = []
                        
                        # Adicionar nova entrada
                        submissions.append(new_entry)
                        
                        # Salvar de volta ao arquivo
                        with open(database_file, 'w', encoding='utf-8') as f:
                            json.dump(submissions, f, indent=2, ensure_ascii=False)
                        
                        st.success(f"✅ Data successfully added to local database!\n\nSubmitted by: **{nickname}**\nDateTime: **{new_entry['submitted_at']}**")
                        success = True
                        
                    except Exception as e:
                        st.error(f"❌ Error saving to database: {str(e)}")
                
                if success:
                    st.info("🔄 Models will be retrained with your data on next prediction!")
                    st.balloons()

# Mostrar dados submetidos pelos usuários
submissions = []

# Tentar carregar do GitHub primeiro
if github_config:
    try:
        submissions = load_from_github(github_config)
    except Exception as e:
        st.warning(f"⚠️ Erro ao carregar submissions do GitHub: {e}")

# Fallback para JSON local se GitHub não estiver configurado ou falhar
if not submissions and os.path.exists('user_submissions.json'):
    try:
        with open('user_submissions.json', 'r', encoding='utf-8') as f:
            submissions = json.load(f)
    except Exception as e:
        st.warning(f"⚠️ Erro ao carregar submissions do JSON: {e}")

# Exibir submissions se houver
if submissions:
    with st.expander("📋 View User Submissions", expanded=False):
        submissions_df = pd.DataFrame(submissions)
        # Reordenar colunas para melhor visualização
        cols_order = ['submitted_at', 'submitted_by', 'sharp', 'ad', 'as', 'mending', 'dps', 'price']
        submissions_df = submissions_df[cols_order]
        
        st.dataframe(
            submissions_df.style.format({
                'ad': '{:.2f}',
                'as': '{:.2f}',
                'dps': '{:.2f}',
                'price': 'PD {:,.0f}'
            }),
            width='stretch'
        )
        
        storage_type = "GitHub" if github_config else "Local JSON"
        st.markdown(f"**Total Submissions:** {len(submissions)} | **Storage:** {storage_type}")

# Footer
st.markdown("---")

# Contabilizar total de itens no dataset
base_items = 59
user_items = 0

# Tentar contar do GitHub primeiro
if github_config:
    try:
        github_data = load_from_github(github_config)
        user_items = len(github_data) if github_data else 0
    except:
        pass

# Fallback para JSON local
if user_items == 0 and os.path.exists('user_submissions.json'):
    try:
        with open('user_submissions.json', 'r', encoding='utf-8') as f:
            user_submissions_data = json.load(f)
            user_items = len(user_submissions_data)
    except:
        pass

total_items = base_items + user_items

storage_info = "GitHub" if github_config else "Local storage"
st.markdown(
    f"""
    <div style='text-align: center; color: #666;'>
        <p>Built with Machine Learning | Dataset: <strong>{total_items} items</strong> ({base_items} historical + {user_items} community)</p>
        <p>Server: <strong>play.cobblemondelta.com</strong></p>
        <p style='font-size: 0.8em; margin-top: 10px;'>🤖 Models are automatically retrained with community contributions</p>
        <p style='font-size: 0.8em;'>💾 Storage: {storage_info}</p>
    </div>
    """,
    unsafe_allow_html=True
)
