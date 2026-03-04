# Price Prediction

Machine Learning project to predict *Scizor Gauntlets* prices.

## 📋 About

Predicts item prices based on combat stats (Attack Damage, Attack Speed, Sharpness, and Mending) for the **play.cobblemondelta.com** server.

## 🎯 What it does

- Calculates DPS (Damage Per Second) automatically, including combat passives
- Compares 4 different ML models:
  - Random Forest
  - Gradient Boosting
  - Ridge Regression
  - Linear Regression
- Interactive widgets to test your own predictions
- Visual charts comparing model performance

## 📊 Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

## 🔧 Built with

Python 3, pandas, scikit-learn, numpy, matplotlib, ipywidgets

## 🚀 Usage

1. Open `priceprediction.ipynb`
2. Run the cells
3. Play with the interactive widgets

Or try it on website: https://cobblemon-price-predictor.streamlit.app/

## 📈 Item Stats

- **Sharp**: Sharpness level (4-5)
- **AD**: Attack Damage (base value)
- **AS**: Attack Speed (%)
- **Mending**: Enchantment (0/1)

## 💰 Currency

Prices in **PD** (Pokédollar)

## 📝 Notes

The model includes fist base bonus (+0.5) and calculates DPS using the 3-attack passive cycle. Dataset has 59 items with historical prices (growing with community contributions).

---

# Price Prediction

Projeto para prever preços de *Scizor Gauntlets* usando Machine Learning.

## 📋 Sobre

Prevê preços de itens baseado nas estatísticas de combate (Attack Damage, Attack Speed, Sharpness e Mending) para o servidor **play.cobblemondelta.com**.

## 🎯 O que faz

- Calcula DPS (Damage Per Second) automaticamente, incluindo passivas de combate
- Compara 4 modelos diferentes de ML:
  - Random Forest
  - Gradient Boosting
  - Ridge Regression
  - Linear Regression
- Widgets interativos para testar suas próprias predições
- Gráficos visuais comparando o desempenho dos modelos

## 📊 Métricas

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coeficiente de Determinação)

## 🔧 Feito com

Python 3, pandas, scikit-learn, numpy, matplotlib, ipywidgets

## 🚀 Como usar

1. Abra o `priceprediction.ipynb`
2. Execute as células
3. Brinque com os widgets interativos

Ou teste no site: https://cobblemon-price-predictor.streamlit.app/

## 📈 Stats do Item

- **Sharp**: Nível de Sharpness (4-5)
- **AD**: Attack Damage (valor base)
- **AS**: Attack Speed (%)
- **Mending**: Encantamento (0/1)

## 💰 Moeda

Preços em **PD** (Pokédollar)

## 📝 Observações

O modelo inclui o bônus de fist base (+0.5) e calcula o DPS usando o ciclo de 3 ataques da passiva. Dataset tem 59 itens com preços históricos (aumentando com contribuições da comunidade).
