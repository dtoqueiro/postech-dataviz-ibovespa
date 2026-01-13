# 📈 Visualização de Previsão de Tendência do IBOVESPA com LSTM

Este projeto apresenta uma aplicação interativa desenvolvida em **Streamlit** para **previsão de tendência do IBOVESPA (alta ou baixa)** utilizando um modelo **LSTM (Long Short-Term Memory)** treinado em dados históricos do índice e do dólar.

A aplicação inclui:
- Previsão atual sob demanda
- Backtesting visual
- Candlestick com marcação das previsões
- Heatmap de acertos e erros
- Análises históricas interativas

---

## 🚀 Demonstração

A aplicação foi projetada para rodar no [**Streamlit Cloud**](https://postech-dataviz-ibovespa.streamlit.app/).

> ⚠️ Observação: devido a bloqueios recentes do Yahoo Finance em ambientes cloud, os dados são obtidos via **Arquivo Local** quando a conexão com o Yahoo Finance é perdida, garantindo estabilidade no deploy.

Foi utilizado CSS costumizado para melhorar a aparência da aplicação.



---

## 🧠 Modelo Utilizado

- Arquitetura: **LSTM**
- Framework: **PyTorch**
- Janela temporal: **20 dias**
- Camadas LSTM: **3**
- Hidden size: **64**
- Saída: **Classificação binária**
  - `0` → tendência de baixa
  - `1` → tendência de alta

O modelo foi treinado em notebook Jupyter e salvo em:
- `melhor_modelo.pth`

---

## 📊 Features Utilizadas

As features são recriadas automaticamente na aplicação e correspondem exatamente às usadas no treinamento:

- Variação diária do IBOVESPA
- Variação diária do dólar (USD/BRL)
- Lags da variação diária (2, 5, 10, 15, 20, 25, 30)
- Médias móveis (MA) da variação diária
- Médias móveis exponenciais (EMA) da variação diária

A lista de features é carregada via:
- `features.pkl`

A normalização é feita com:
- `scaler.pkl`

---


## 📈 Funcionalidades da Aplicação

### 🔮 Previsão Atual
- Botão para gerar a previsão mais recente
- Feedback visual com spinner
- Resultado claro (📈 Alta / 📉 Baixa)

---

### 📉 Backtesting
- Indicadores de performance
- Performance Recente (30 dias)
- Evolução da Acurácia
- Mapa de Acertos e Erros
- Heatmap Semanal de Performance
- é possível selecional o período do backtesting e Exportar Dados

---

### 🕯️ Candlestick com Previsões
- Gráfico de velas do IBOVESPA
- Marcação das previsões do modelo:
  - Verde → previsão de alta
  - Vermelho → previsão de baixa
- Correlação IBOVESPA x USD/BRL
- Distribuição de Variações Diárias

---


