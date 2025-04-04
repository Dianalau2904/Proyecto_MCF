import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import kurtosis, skew, norm, t

# Configuración inicial de la página
st.set_page_config(page_title="Análisis de Riesgo Financiero", layout="wide")
st.title("Proyecto 1: Cálculo de Value-At-Risk y Expected Shortfall")

# Funciones para descarga y cálculo de datos

def obtener_datos(stocks):
    df = yf.download(stocks, start="2010-01-01")["Close"]
    return df

def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Datos de entrada y cálculos base
M7 = ['NVDA']
df_precios = obtener_datos(M7)
df_rendimientos = calcular_rendimientos(df_precios)

# Estadísticas básicas
promedio_rendi_diario = df_rendimientos['NVDA'].mean()
kurt = kurtosis(df_rendimientos['NVDA'])
skewness = skew(df_rendimientos['NVDA'])

# Mostrar métricas de forma ordenada
st.subheader("Estadísticas Descriptivas")
col1, col2, col3 = st.columns(3)
col1.metric("Rendimiento Diario Promedio", f"{promedio_rendi_diario:.4%}")
col2.metric("Curtosis", f"{kurt:.2f}")
col3.metric("Sesgo", f"{skewness:.2f}")

# Funciones de cálculo de riesgo

def calcular_var_parametrico(returns, alpha, distrib='normal'):
    mean = np.mean(returns)
    stdev = np.std(returns)
    if distrib == 'normal':
        return norm.ppf(1 - alpha, mean, stdev)
    elif distrib == 't':
        df_t = 10
        return t.ppf(1 - alpha, df_t, mean, stdev)

def calcular_var_historico(returns, alpha):
    return returns.quantile(1 - alpha)

def calcular_var_montecarlo(returns, alpha, n_sims=100000):
    mean = np.mean(returns)
    stdev = np.std(returns)
    sim_returns = np.random.normal(mean, stdev, n_sims)
    return np.percentile(sim_returns, (1 - alpha) * 100)

def calcular_cvar(returns, hVaR):
    return returns[returns <= hVaR].mean()

# Calcular y almacenar VaR y CVaR
alpha_vals = [0.95, 0.975, 0.99]
resultados_df = pd.DataFrame(columns=['VaR (Normal)', 'VaR (t-Student)', 'VaR (Histórico)', 'VaR (Monte Carlo)',
                                      'CVaR (Normal)', 'CVaR (t-Student)', 'CVaR (Histórico)', 'CVaR (Monte Carlo)'])

for alpha in alpha_vals:
    var_normal = calcular_var_parametrico(df_rendimientos['NVDA'], alpha, 'normal')
    var_t = calcular_var_parametrico(df_rendimientos['NVDA'], alpha, 't')
    var_historico = calcular_var_historico(df_rendimientos['NVDA'], alpha)
    var_montecarlo = calcular_var_montecarlo(df_rendimientos['NVDA'], alpha)

    cvar_normal = calcular_cvar(df_rendimientos['NVDA'], var_normal)
    cvar_t = calcular_cvar(df_rendimientos['NVDA'], var_t)
    cvar_historico = calcular_cvar(df_rendimientos['NVDA'], var_historico)
    cvar_montecarlo = calcular_cvar(df_rendimientos['NVDA'], var_montecarlo)

    resultados_df.loc[f'{int(alpha * 100)}% Confidence'] = [
        var_normal * 100, var_t * 100, var_historico * 100, var_montecarlo * 100,
        cvar_normal * 100, cvar_t * 100, cvar_historico * 100, cvar_montecarlo * 100
    ]

# Histograma de rendimientos
fig, ax = plt.subplots(figsize=(10, 6))
n, bins, patches = ax.hist(df_rendimientos['NVDA'], bins=50, color='dodgerblue', alpha=0.7, label='Rendimientos')
for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
    if bin_left < var_historico:
        patch.set_facecolor('salmon')

# Líneas de VaR y CVaR en el histograma
ax.axvline(x=var_normal, color='lightseagreen', linestyle='--', label='VaR 95% (Normal)')
ax.axvline(x=var_montecarlo, color='darkgrey', linestyle='--', label='VaR 95% (Monte Carlo)')
ax.axvline(x=var_historico, color='forestgreen', linestyle='--', label='VaR 95% (Histórico)')
ax.axvline(x=cvar_normal, color='purple', linestyle='-.', label='CVaR 95% (Normal)')

ax.set_title('Histograma de Rendimientos con VaR y CVaR')
ax.set_xlabel('Rendimientos')
ax.set_ylabel('Frecuencia')
ax.legend()

# Rolling VaR y Expected Shortfall
window_size = 252
rolling_mean = df_rendimientos['NVDA'].rolling(window=window_size).mean()
rolling_std = df_rendimientos['NVDA'].rolling(window=window_size).std()

VaR_rolling = {
    95: norm.ppf(1 - 0.95, rolling_mean, rolling_std),
    99: norm.ppf(1 - 0.99, rolling_mean, rolling_std)
}

VaR_hist_rolling = {
    95: df_rendimientos['NVDA'].rolling(window=window_size).quantile(0.05),
    99: df_rendimientos['NVDA'].rolling(window=window_size).quantile(0.01)
}

def calcular_ES(returns, var_rolling):
    return returns[returns <= var_rolling].mean()

ES_rolling = {alpha: [calcular_ES(df_rendimientos['NVDA'][i - window_size:i], VaR_rolling[alpha][i]) for i in range(window_size, len(df_rendimientos))] for alpha in [95, 99]}
ES_hist_rolling = {alpha: [calcular_ES(df_rendimientos['NVDA'][i - window_size:i], VaR_hist_rolling[alpha][i]) for i in range(window_size, len(df_rendimientos))] for alpha in [95, 99]}

rolling_results_df = pd.DataFrame({
    'Date': df_rendimientos.index[window_size:],
    **{f'VaR_{alpha}_Rolling': VaR_rolling[alpha][window_size:] for alpha in [95, 99]},
    **{f'VaR_{alpha}_Rolling_Hist': VaR_hist_rolling[alpha][window_size:] for alpha in [95, 99]},
    **{f'ES_{alpha}_Rolling': ES_rolling[alpha] for alpha in [95, 99]},
    **{f'ES_{alpha}_Rolling_Hist': ES_hist_rolling[alpha] for alpha in [95, 99]}
})
rolling_results_df.set_index('Date', inplace=True)

fig2, ax2 = plt.subplots(figsize=(14, 7))
ax2.plot(df_rendimientos.index, df_rendimientos['NVDA'] * 100, label='Rendimientos Diarios (%)', color='blue', alpha=0.5)
for alpha in [95, 99]:
    ax2.plot(rolling_results_df.index, rolling_results_df[f'VaR_{alpha}_Rolling'] * 100, label=f'VaR {alpha}% Rolling Paramétrico')
    ax2.plot(rolling_results_df.index, rolling_results_df[f'VaR_{alpha}_Rolling_Hist'] * 100, label=f'VaR {alpha}% Rolling Histórico')
    ax2.plot(rolling_results_df.index, rolling_results_df[f'ES_{alpha}_Rolling'] * 100, label=f'ES {alpha}% Rolling Paramétrico')
    ax2.plot(rolling_results_df.index, rolling_results_df[f'ES_{alpha}_Rolling_Hist'] * 100, label=f'ES {alpha}% Rolling Histórico')
ax2.set_title('Rendimientos Diarios y VaR/ES Rolling Window (252 días)')
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Valor (%)')
ax2.legend()

# Evaluación de violaciones

def contar_violaciones(returns, risk_measure):
    violations = returns < risk_measure
    num_violations = violations.sum()
    violation_percentage = (num_violations / len(returns)) * 100
    return num_violations, violation_percentage

returns_for_test = df_rendimientos['NVDA'].iloc[window_size:]
risk_measures = {
    'VaR 95% Paramétrico': rolling_results_df['VaR_95_Rolling'],
    'VaR 99% Paramétrico': rolling_results_df['VaR_99_Rolling'],
    'VaR 95% Histórico': rolling_results_df['VaR_95_Rolling_Hist'],
    'VaR 99% Histórico': rolling_results_df['VaR_99_Rolling_Hist'],
    'ES 95% Paramétrico': rolling_results_df['ES_95_Rolling'],
    'ES 99% Paramétrico': rolling_results_df['ES_99_Rolling'],
    'ES 95% Histórico': rolling_results_df['ES_95_Rolling_Hist'],
    'ES 99% Histórico': rolling_results_df['ES_99_Rolling_Hist']
}

violation_results = {'Número de violaciones': [], 'Porcentaje de violaciones': []}
for name, values in risk_measures.items():
    violations, percent = contar_violaciones(returns_for_test, values)
    violation_results['Número de violaciones'].append(violations)
    violation_results['Porcentaje de violaciones'].append(percent)
violation_results_df = pd.DataFrame(violation_results, index=risk_measures.keys())

# Volatilidad móvil y violaciones
q_95 = norm.ppf(0.05)
q_99 = norm.ppf(0.01)
rolling_vol = df_rendimientos['NVDA'].rolling(window=window_size).std()
rolling_var_95 = q_95 * rolling_vol
rolling_var_99 = q_99 * rolling_vol

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(df_rendimientos.index, df_rendimientos['NVDA'], label='Retornos', alpha=0.5)
ax3.plot(df_rendimientos.index, rolling_var_95, label='VaR 95% (Volatilidad Móvil)', linestyle='dashed')
ax3.plot(df_rendimientos.index, rolling_var_99, label='VaR 99% (Volatilidad Móvil)', linestyle='dashed')
ax3.legend()
ax3.set_title('VaR con Volatilidad Móvil')
ax3.set_xlabel('Fecha')
ax3.set_ylabel('VaR')

violaciones = {'VaR_95': 0, 'VaR_99': 0}
total_pruebas = len(df_rendimientos) - window_size
for i in range(window_size, len(df_rendimientos)):
    r_t = df_rendimientos['NVDA'].iloc[i]
    if r_t < rolling_var_95.iloc[i]:
        violaciones['VaR_95'] += 1
    if r_t < rolling_var_99.iloc[i]:
        violaciones['VaR_99'] += 1
violaciones_pct = {key: (value / total_pruebas) * 100 for key, value in violaciones.items()}
violaciones_df = pd.DataFrame([violaciones, violaciones_pct], index=['Violaciones', 'Porcentaje (%)']).T

# Interfaz con pestañas Streamlit

tabs = st.tabs(["📊 Resultados VaR/CVaR", "📈 Histograma VaR", "📉 VaR/ES Rolling", "⚠️ Violaciones"])

with tabs[0]:
    st.subheader("Tabla de Resultados de VaR y CVaR")
    st.dataframe(resultados_df.style.format("{:.2f}"))

with tabs[1]:
    st.subheader("Histograma de Rendimientos con VaR y CVaR")
    st.pyplot(fig)

with tabs[2]:
    st.subheader("Rendimientos Diarios y Rolling VaR / Expected Shortfall")
    st.pyplot(fig2)

with tabs[3]:
    st.subheader("Violaciones a VaR y Expected Shortfall")
    st.caption("Violaciones en medidas rolling")
    st.dataframe(violation_results_df.style.format("{:.2f}"))
    st.caption("Violaciones usando volatilidad móvil")
    st.dataframe(violaciones_df.style.format("{:.2f}"))
    st.subheader("Gráfica de VaR con Volatilidad Móvil")
    st.pyplot(fig3)
