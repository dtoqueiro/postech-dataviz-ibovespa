import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import pandas_datareader.data as web
import logging
# import appdirs as ad

# ad.user_cache_dir = lambda *args: "/tmp"
import yfinance as yf

# ==========================================================
# CONFIGURAÇÃO DA PÁGINA E ESTILO
# ==========================================================
st.set_page_config(
    page_title="Previsão IBOVESPA",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CSS customizado para melhorar estética
st.markdown(
    """
<style>
    /* Paleta de cores profissional */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ecc71;
        --danger-color: #e74c3c;
        --warning-color: #f39c12;
        --bg-dark: #0e1117;
        --bg-light: #262730;
    }
    
    /* Melhorar espaçamento geral */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Títulos mais elegantes */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(120deg, #1f77b4 0%, #2ecc71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #1f77b4;
    }
    
    h3 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        color: #1f77b4 !important;
    }
    
    /* Cards de métricas mais elegantes */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
        opacity: 0.8;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }
    
    /* Botões mais modernos */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1f77b4 0%, #2ecc71 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.4);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 204, 113, 0.4);
    }
    
    /* Expanders mais elegantes */
    .streamlit-expanderHeader {
        background-color: rgba(31, 119, 180, 0.1);
        border-radius: 8px;
        padding: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Inputs de data */
    .stDateInput > div > div > input {
        border-radius: 8px;
        border: 2px solid rgba(31, 119, 180, 0.3);
        padding: 0.5rem;
        font-size: 1rem;
    }
    
    /* Alertas personalizados */
    .stAlert {
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
    }
    
    /* Caption mais sutil */
    .caption-custom {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-top: 1rem;
        padding: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Containers de destaque */
    .highlight-box {
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.1) 0%, rgba(46, 204, 113, 0.1) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(31, 119, 180, 0.2);
    }
    
    /* Separadores */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #1f77b4, transparent);
    }
</style>
""",
    unsafe_allow_html=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 3
OUTPUT_SIZE = 2

# Paleta de cores consistente para gráficos
CHART_COLORS = {
    "primary": "#1f77b4",
    "success": "#2ecc71",
    "danger": "#e74c3c",
    "warning": "#f39c12",
    "secondary": "#95a5a6",
    "info": "#3498db",
    "gradient": ["#1f77b4", "#2ecc71", "#3498db"],
}


# ==========================================================
# MODELO
# ==========================================================
class ModeloTendenciaIBOV(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE, device=x.device)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ==========================================================
# LOADERS COM TRATAMENTO DE ERROS
# ==========================================================
@st.cache_resource
def carregar_modelo():
    try:
        feature_cols = joblib.load("features.pkl")
        model = ModeloTendenciaIBOV(
            len(feature_cols), HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE
        ).to(device)
        model.load_state_dict(torch.load("melhor_modelo.pth", map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(
            "❌ Arquivos do modelo não encontrados. Verifique se 'melhor_modelo.pth' e 'features.pkl' existem."
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        st.stop()


@st.cache_resource
def carregar_scaler():
    try:
        scaler = joblib.load("scaler.pkl")
        scaler["std"][scaler["std"] == 0] = 1.0
        return scaler
    except FileNotFoundError:
        st.error("❌ Arquivo 'scaler.pkl' não encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar scaler: {str(e)}")
        st.stop()


@st.cache_resource
def carregar_features():
    try:
        return joblib.load("features.pkl")
    except FileNotFoundError:
        st.error("❌ Arquivo 'features.pkl' não encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar features: {str(e)}")
        st.stop()


FEATURE_COLUMNS = carregar_features()
modelo = carregar_modelo()
scaler = carregar_scaler()


# ==========================================================
# DADOS
# ==========================================================
@st.cache_data()
def carregar_dados(periodo="4y"):
    try:
        ibov = yf.download("^BVSP", period=periodo, interval="1d", progress=False)
        dolar = yf.download("USDBRL=X", period=periodo, interval="1d", progress=False)

        if ibov.empty or dolar.empty:
            raise ValueError("O Yahoo Finance retornou dados vazios.")

        df = pd.concat([ibov[["Open", "High", "Low", "Close"]], dolar["Close"]], axis=1)

        df.columns = ["Open", "High", "Low", "Close", "Close_Dolar"]
        df.dropna(inplace=True)

        df.to_csv("data.csv")

        return df
    except Exception as e:
        try:
            logging.warning(
                f"⚠️ Falha na conexão ou download ({str(e)}). Tentando carregar dados locais..."
            )

            # Lê o CSV definindo a primeira coluna como índice (datas)
            df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

            logging.info("✅ Dados carregados do arquivo local (data.csv).")
            return df
        except FileNotFoundError:
            logging.error(
                "❌ Erro crítico: Não foi possível baixar os dados e o arquivo 'data.csv' não foi encontrado."
            )
            st.stop()


# ==========================================================
# FEATURES
# ==========================================================
def criar_features(df):
    df = df.copy()

    df["Daily_Change"] = df["Close"].pct_change()
    df["Daily_Change_Dolar"] = df["Close_Dolar"].pct_change()

    lags = [2, 5, 10, 15, 20, 25, 30]
    for lag in lags:
        df[f"Daily_Change_lag_{lag}"] = df["Daily_Change"].shift(lag)

    windows = [5, 10, 15, 20, 25, 30]
    for w in windows:
        df[f"MA_{w}_Daily_Change"] = df["Daily_Change"].rolling(w).mean()
        df[f"EMA_{w}_Daily_Change"] = (
            df["Daily_Change"].ewm(span=w, adjust=False).mean()
        )

    df["Target"] = (df["Daily_Change"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df


# ==========================================================
# INPUT
# ==========================================================
def preparar_input(janela):
    valores = janela[FEATURE_COLUMNS].values.astype(np.float32)
    valores_norm = (valores - scaler["mean"]) / scaler["std"]
    return torch.tensor(valores_norm).unsqueeze(0).to(device)


# ==========================================================
# BACKTEST COM CACHE
# ==========================================================
@st.cache_data(show_spinner="🔄 Executando backtest...")
def executar_backtest(_df_feat):
    resultados = []

    for i in range(WINDOW_SIZE, len(_df_feat)):
        janela = _df_feat.iloc[i - WINDOW_SIZE : i]
        x = preparar_input(janela)

        with torch.no_grad():
            pred = torch.argmax(modelo(x), dim=1).item()

        real = int(_df_feat.iloc[i]["Target"])

        resultados.append(
            {
                "Data": _df_feat.index[i],
                "Predicao": pred,
                "Real": real,
                "Acerto": pred == real,
                "Close": _df_feat.iloc[i]["Close"],
            }
        )

    return pd.DataFrame(resultados)


# ==========================================================
# FUNÇÃO AUXILIAR - STREAK
# ==========================================================
def calcular_streak(df):
    if len(df) == 0:
        return {"tipo": "N/A", "valor": 0}

    streak = 0
    tipo = "acertos" if df.iloc[-1]["Acerto"] else "erros"

    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]["Acerto"] == df.iloc[-1]["Acerto"]:
            streak += 1
        else:
            break

    return {"tipo": tipo.capitalize(), "valor": streak}


# ==========================================================
# APP - HEADER
# ==========================================================
st.title("📈 Previsão de Tendência IBOVESPA")
st.markdown(
    """
<div style='text-align: center; padding: 1rem 0; margin-bottom: 2rem;'>
    <span style='font-size: 1.1rem; color: #888;'>
        🧙‍♂️ Modelo LSTM • 📊 IBOV + Dólar • 🔄 Atualização em tempo real
    </span>
</div>
""",
    unsafe_allow_html=True,
)

df = carregar_dados()
df_feat = criar_features(df)

# ==========================================================
# INFORMAÇÕES DOS DADOS
# ==========================================================
st.markdown("## 📊 Informações dos Dados")
with st.expander("ℹ️ Ver detalhes dos dados", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Última cotação:** {df.index[-1].strftime('%d/%m/%Y')}")
        st.write(f"**IBOV:** {df['Close'].iloc[-1]:,.2f} pts")
        st.write(f"**Variação (dia):** {df_feat['Daily_Change'].iloc[-1]:.2%}")
    with col2:
        st.write(f"**Total de dias:** {len(df)}")
        st.write(f"**USD/BRL:** R$ {df['Close_Dolar'].iloc[-1]:.4f}")
        st.write(
            f"**Período:** {df.index[0].strftime('%d/%m/%Y')} a {df.index[-1].strftime('%d/%m/%Y')}"
        )

# ==========================================================
# SEÇÃO 1: PREVISÃO ATUAL (DESTAQUE)
# ==========================================================
# st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
st.markdown("## 🔮 Previsão para o Próximo Pregão")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("📅 Última Atualização", df.index[-1].strftime("%d/%m/%Y"))
with col_info2:
    variacao = df_feat["Daily_Change"].iloc[-1]
    st.metric(
        "📊 IBOV Atual",
        f"{df['Close'].iloc[-1]:,.0f}",
        delta=f"{variacao:.2%}",
        delta_color="normal",
    )
with col_info3:
    st.metric("💵 USD/BRL", f"R$ {df['Close_Dolar'].iloc[-1]:.4f}")

st.markdown("")

if st.button("🚀 Gerar Previsão Agora", width="stretch"):
    with st.spinner("⏳ Analisando padrões do mercado..."):
        janela_atual = df_feat.iloc[-WINDOW_SIZE:]
        x_atual = preparar_input(janela_atual)

        with torch.no_grad():
            probs = torch.softmax(modelo(x_atual), dim=1).cpu().numpy()[0]

    st.markdown("### 🎯 Resultado da Análise")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if probs[1] > probs[0]:
            st.markdown(
                f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(46, 204, 113, 0.2) 0%, rgba(46, 204, 113, 0.05) 100%); 
                        border-radius: 16px; border: 2px solid #2ecc71;'>
                <h2 style='color: #2ecc71; margin: 0; font-size: 2.5rem;'>📈 ALTA</h2>
                <p style='font-size: 1.5rem; margin: 0.5rem 0 0 0;'>Confiança: <strong>{probs[1]:.1%}</strong></p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(231, 76, 60, 0.2) 0%, rgba(231, 76, 60, 0.05) 100%); 
                        border-radius: 16px; border: 2px solid #e74c3c;'>
                <h2 style='color: #e74c3c; margin: 0; font-size: 2.5rem;'>📉 BAIXA</h2>
                <p style='font-size: 1.5rem; margin: 0.5rem 0 0 0;'>Confiança: <strong>{probs[0]:.1%}</strong></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# SEÇÃO 2: ANÁLISE DE BACKTEST
# ==========================================================
st.markdown("---")
st.markdown("## 🧠 Análise de Performance (Backtest)")

try:
    df_backtest_full = executar_backtest(df_feat)
    df_backtest_full["Data"] = pd.to_datetime(df_backtest_full["Data"])
except Exception as e:
    st.error(f"Erro ao executar backtest: {e}")
    st.stop()

# Seleção de período
col_date1, col_date2, col_spacer = st.columns([2, 2, 1])
with col_date1:
    data_ini = st.date_input("📅 Data Inicial", df_feat.index.min().date())
with col_date2:
    data_fim = st.date_input("📅 Data Final", df_feat.index.max().date())

# Validação de datas
if data_ini >= data_fim:
    st.warning("⚠️ A data inicial deve ser anterior à data final.")
    st.stop()

df_bt = df_feat.loc[str(data_ini) : str(data_fim)]

if len(df_bt) <= WINDOW_SIZE:
    st.warning(f"⚠️ Período muito curto. Necessário mínimo de **{WINDOW_SIZE} dias**.")
    st.stop()

# Executar backtest
with st.spinner("🔄 Processando backtest..."):
    df_backtest = df_backtest_full[
        (df_backtest_full["Data"].dt.date >= data_ini)
        & (df_backtest_full["Data"].dt.date <= data_fim)
    ]
    df_backtest["Acerto_Int"] = df_backtest["Acerto"].astype(int)
    df_backtest["Acuracia"] = df_backtest["Acerto_Int"].expanding().mean()

# ==========================================================
# MÉTRICAS PRINCIPAIS
# ==========================================================
acuracia_total = df_backtest["Acerto"].mean()
total_previsoes = len(df_backtest)
acertos = df_backtest["Acerto"].sum()
erros = total_previsoes - acertos

st.markdown("### 📊 Indicadores de Performance")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Acurácia Geral", f"{acuracia_total:.1%}")
with col2:
    st.metric("✅ Total de Acertos", f"{acertos:,}")
with col3:
    st.metric("❌ Total de Erros", f"{erros:,}")
with col4:
    st.metric("📈 Previsões", f"{total_previsoes:,}")

# ==========================================================
# ANÁLISE RECENTE (30 DIAS)
# ==========================================================
if len(df_backtest) >= 30:
    st.markdown("### 🔥 Performance Recente (30 dias)")

    df_recente = df_backtest.tail(30)
    acertos_recentes = df_recente["Acerto"].mean()
    streak_atual = calcular_streak(df_recente)

    col1, col2, col3 = st.columns(3)
    with col1:
        delta_acuracia = acertos_recentes - acuracia_total
        st.metric(
            "📊 Acurácia (30d)",
            f"{acertos_recentes:.1%}",
            delta=f"{delta_acuracia:.1%}",
        )
    with col2:
        st.metric(
            "🔄 Sequência Atual", f"{streak_atual['valor']} {streak_atual['tipo']}"
        )
    with col3:
        acertos_30d = df_recente["Acerto"].sum()
        st.metric("✅ Acertos (30d)", f"{acertos_30d}/30")

# ==========================================================
# GRÁFICO DE ACURÁCIA ACUMULADA
# ==========================================================
st.markdown("### 📈 Evolução da Acurácia")

fig = go.Figure()

# Linha principal de acurácia
fig.add_trace(
    go.Scatter(
        x=df_backtest["Data"],
        y=df_backtest["Acuracia"],
        mode="lines",
        line=dict(width=3, color=CHART_COLORS["primary"]),
        name="Acurácia Acumulada",
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.1)",
    )
)

# Linha de referência 50%
fig.add_hline(
    y=0.5,
    line_dash="dash",
    line_color=CHART_COLORS["secondary"],
    annotation_text="Baseline (50%)",
    annotation_position="right",
)

fig.update_layout(
    yaxis=dict(range=[0, 1], tickformat=".0%", title="Acurácia"),
    xaxis=dict(title="Data"),
    hovermode="x unified",
    height=450,
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, width="stretch")

# ==========================================================
# GRÁFICO PREVISÃO VS REALIDADE
# ==========================================================
st.markdown("### 🎯 Mapa de Acertos e Erros")

fig = go.Figure()

# Linha do IBOV (fundo)
fig.add_trace(
    go.Scatter(
        x=df_backtest["Data"],
        y=df_backtest["Close"],
        mode="lines",
        name="IBOV",
        line=dict(color=CHART_COLORS["secondary"], width=2, dash="dot"),
        opacity=0.4,
    )
)

# Pontos de acerto (verde)
acertos_df = df_backtest[df_backtest["Acerto"]]
fig.add_trace(
    go.Scatter(
        x=acertos_df["Data"],
        y=acertos_df["Close"],
        mode="markers",
        name="✅ Acertos",
        marker=dict(
            color=CHART_COLORS["success"],
            size=8,
            symbol="circle",
            line=dict(color="white", width=1),
        ),
    )
)

# Pontos de erro (vermelho)
erros_df = df_backtest[~df_backtest["Acerto"]]
fig.add_trace(
    go.Scatter(
        x=erros_df["Data"],
        y=erros_df["Close"],
        mode="markers",
        name="❌ Erros",
        marker=dict(
            color=CHART_COLORS["danger"],
            size=8,
            symbol="x",
            line=dict(color="white", width=1),
        ),
    )
)

fig.update_layout(
    hovermode="x unified",
    height=450,
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(title="Pontos IBOV"),
    xaxis=dict(title="Data"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, width="stretch")

# ==========================================================
# HEATMAP DE ACERTOS
# ==========================================================
st.markdown("### 🔥 Heatmap Semanal de Performance")

df_backtest["Ano"] = df_backtest["Data"].dt.year
df_backtest["Semana"] = df_backtest["Data"].dt.isocalendar().week

heatmap = df_backtest.groupby(["Ano", "Semana"])["Acerto"].mean().reset_index()

fig = px.density_heatmap(
    heatmap,
    x="Semana",
    y="Ano",
    z="Acerto",
    color_continuous_scale="RdYlGn",
    labels={"Acerto": "Taxa de Acerto", "Semana": "Semana do Ano", "Ano": "Ano"},
)

fig.update_traces(texttemplate="%{z:.0%}", textfont_size=9)
fig.update_layout(
    height=350,
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, width="stretch")

# ==========================================================
# DOWNLOAD DOS RESULTADOS
# ==========================================================
st.markdown("### 💾 Exportar Dados")

csv = df_backtest.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Baixar Resultados do Backtest (CSV)",
    data=csv,
    file_name=f"backtest_ibov_{data_ini.strftime('%Y%m%d')}_{data_fim.strftime('%Y%m%d')}.csv",
    mime="text/csv",
    width="stretch",
)

# ==========================================================
# ANÁLISES HISTÓRICAS EXPANDIDAS
# ==========================================================
st.markdown("---")
with st.expander("📊 **Ver Análises Históricas Completas**", expanded=False):
    # --- Candlestick ---
    st.markdown("#### 🕯️ Gráfico de Candlestick - IBOVESPA")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="IBOV",
                increasing_line_color=CHART_COLORS["success"],
                decreasing_line_color=CHART_COLORS["danger"],
            )
        ]
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=450,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    # --- IBOV x Dólar ---
    st.markdown("#### 🌎 Correlação IBOVESPA x USD/BRL")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="IBOV",
            line=dict(color=CHART_COLORS["primary"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close_Dolar"],
            name="USD/BRL",
            yaxis="y2",
            line=dict(dash="dash", color=CHART_COLORS["warning"], width=2),
        )
    )
    fig.update_layout(
        yaxis=dict(title="IBOV (pontos)"),
        yaxis2=dict(title="USD/BRL (R$)", overlaying="y", side="right"),
        hovermode="x unified",
        height=450,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    # --- Variação diária ---
    st.markdown("#### 📊 Distribuição de Variações Diárias")

    col_hist1, col_hist2 = st.columns(2)

    with col_hist1:
        # Gráfico de barras
        fig = go.Figure()
        colors = [
            CHART_COLORS["success"] if x > 0 else CHART_COLORS["danger"]
            for x in df_feat["Daily_Change"]
        ]
        fig.add_trace(
            go.Bar(
                x=df_feat.index,
                y=df_feat["Daily_Change"] * 100,
                name="Variação Diária",
                marker_color=colors,
            )
        )
        fig.update_layout(
            yaxis_title="Variação (%)",
            hovermode="x unified",
            height=400,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    with col_hist2:
        # Histograma
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df_feat["Daily_Change"] * 100,
                nbinsx=50,
                name="Distribuição",
                marker_color=CHART_COLORS["info"],
            )
        )
        fig.update_layout(
            xaxis_title="Variação Diária (%)",
            yaxis_title="Frequência",
            height=400,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

# ==========================================================
# RODAPÉ
# ==========================================================
st.markdown("---")
st.markdown(
    """
<div class='caption-custom'>
    <strong>Dashboard de Previsão IBOVESPA</strong><br>
    Powered by PyTorch LSTM • Design by UX Best Practices • Data from Yahoo Finance<br>
    <em>Este modelo é apenas para fins educacionais. Não constitui recomendação de investimento.</em>
</div>
""",
    unsafe_allow_html=True,
)
