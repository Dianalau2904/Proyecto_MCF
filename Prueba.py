import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import kurtosis, skew, norm, t

# Funciones MCF integradas 

def obtener_datos(stocks):
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Datos y parámetros
M7 = ['NVDA']
df_precios = obtener_datos(M7)
df_rendimientos = calcular_rendimientos(df_precios)

# Promedio de rendimientos
promedio_rendi_diario = df_rendimientos['NVDA'].mean()
st.write(f"Promedio de rendimiento diario: {promedio_rendi_diario:.4f}")

# Estadísticas
kurt = kurtosis(df_rendimientos['NVDA'])
skewness = skew(df_rendimientos['NVDA'])
st.write(f"Kurtois: {kurt:.4f}")
st.write(f"Sesgo: {skewness:.4f}")

# Funciones para calcular VaR y CVaR
def calcular_var_parametrico(returns, alpha, distrib='normal'):
    mean = np.mean(returns)
    stdev = np.std(returns)
    
    if distrib == 'normal':
        VaR = norm.ppf(1-alpha, mean, stdev)
    elif distrib == 't':
        df_t = 10  # Grados de libertad (es ajustable)
        VaR = t.ppf(1-alpha, df_t, mean, stdev)
    return VaR

def calcular_var_historico(returns, alpha):
    hVaR = returns.quantile(1-alpha)
    return hVaR

def calcular_var_montecarlo(returns, alpha, n_sims=100000):
    mean = np.mean(returns)
    stdev = np.std(returns)
    sim_returns = np.random.normal(mean, stdev, n_sims)
    return np.percentile(sim_returns, (1-alpha)*100)

def calcular_cvar(returns, hVaR):
    return returns[returns <= hVaR].mean()

# Cálculo de VaR y CVaR
alpha_vals = [0.95, 0.975, 0.99]
resultados_df = pd.DataFrame(columns=['VaR (Normal)', 'VaR (t-Student)', 'VaR (Histórico)', 'VaR (Monte Carlo)', 
                                      'CVaR (Normal)', 'CVaR (t-Student)', 'CVaR (Histórico)', 'CVaR (Monte Carlo)'])

# Funciones de cálculo de VaR y CVaR
for alpha in alpha_vals:
    var_normal = calcular_var_parametrico(df_rendimientos['NVDA'], alpha, distrib='normal')
    var_t = calcular_var_parametrico(df_rendimientos['NVDA'], alpha, distrib='t')
    var_historico = calcular_var_historico(df_rendimientos['NVDA'], alpha)
    var_montecarlo = calcular_var_montecarlo(df_rendimientos['NVDA'], alpha)
    
    cvar_normal = calcular_cvar(df_rendimientos['NVDA'], var_normal)
    cvar_t = calcular_cvar(df_rendimientos['NVDA'], var_t)
    cvar_historico = calcular_cvar(df_rendimientos['NVDA'], var_historico)
    cvar_montecarlo = calcular_cvar(df_rendimientos['NVDA'], var_montecarlo)
    
    # Resultados
    resultados_df.loc[f'{int(alpha*100)}% Confidence'] = [
        var_normal * 100, var_t * 100, var_historico * 100, var_montecarlo * 100,
        cvar_normal * 100, cvar_t * 100, cvar_historico * 100, cvar_montecarlo * 100
    ]

# Mostrar resultados en formato dataframe
st.write("Resultados de VaR y CVaR por intervalo de confianza:")
st.dataframe(resultados_df)

# Graficar el histograma de rendimientos y los valores de VaR y CVaR
fig, ax = plt.subplots(figsize=(10, 6))
n, bins, patches = ax.hist(df_rendimientos['NVDA'], bins=50, color='dodgerblue', alpha=0.7, label='Rendimientos')

# Colorear
for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
    if bin_left < var_historico:
        patch.set_facecolor('salmon')

# Marcar los diferentes VaR y CVaR
ax.axvline(x=var_normal, color='lightseagreen', linestyle='--', label='VaR 95% (Normal)')
ax.axvline(x=var_montecarlo, color='darkgrey', linestyle='--', label='VaR 95% (Monte Carlo)')
ax.axvline(x=var_historico, color='forestgreen', linestyle='--', label='VaR 95% (Histórico)')
ax.axvline(x=cvar_normal, color='purple', linestyle='-.', label='CVaR 95% (Normal)')

ax.set_title('Histograma de Rendimientos con VaR y CVaR', fontsize=14)
ax.set_xlabel('Rendimientos', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.legend()

# Mostrar gráfico
st.pyplot(fig)

# Definir tamaño de ventana móvil
window_size = 252

# Calcular la media y desviación estándar de los retornos en la ventana móvil
rolling_mean = df_rendimientos['NVDA'].rolling(window=window_size).mean()
rolling_std = df_rendimientos['NVDA'].rolling(window=window_size).std()

# Calcular VaR paramétrico para 95% y 99% usando distribución normal
VaR_rolling = {
    95: norm.ppf(1 - 0.95, rolling_mean, rolling_std),
    99: norm.ppf(1 - 0.99, rolling_mean, rolling_std)
}

# Calcular VaR histórico para 95% y 99%
VaR_hist_rolling = {
    95: df_rendimientos['NVDA'].rolling(window=window_size).quantile(0.05),
    99: df_rendimientos['NVDA'].rolling(window=window_size).quantile(0.01)
}

# Función para calcular Expected Shortfall (ES)
def calcular_ES(returns, var_rolling):
    return returns[returns <= var_rolling].mean()

# Calcular ES para 95% y 99% (paramétrico y histórico)
ES_rolling = {alpha: [calcular_ES(df_rendimientos['NVDA'][i-window_size:i], VaR_rolling[alpha][i]) for i in range(window_size, len(df_rendimientos))]
              for alpha in [95, 99]}

ES_hist_rolling = {alpha: [calcular_ES(df_rendimientos['NVDA'][i-window_size:i], VaR_hist_rolling[alpha][i]) for i in range(window_size, len(df_rendimientos))]
                   for alpha in [95, 99]}

# Crear DataFrame con resultados
rolling_results_df = pd.DataFrame({
    'Date': df_rendimientos.index[window_size:],
    **{f'VaR_{alpha}_Rolling': VaR_rolling[alpha][window_size:] for alpha in [95, 99]},
    **{f'VaR_{alpha}_Rolling_Hist': VaR_hist_rolling[alpha][window_size:] for alpha in [95, 99]},
    **{f'ES_{alpha}_Rolling': ES_rolling[alpha] for alpha in [95, 99]},
    **{f'ES_{alpha}_Rolling_Hist': ES_hist_rolling[alpha] for alpha in [95, 99]}
})

rolling_results_df.set_index('Date', inplace=True)

# Mostrar gráfico de rendimientos y VaR/ES rolling
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df_rendimientos.index, df_rendimientos['NVDA'] * 100, label='Rendimientos Diarios (%)', color='blue', alpha=0.5)

for alpha in [95, 99]:
    ax.plot(rolling_results_df.index, rolling_results_df[f'VaR_{alpha}_Rolling'] * 100, label=f'VaR {alpha}% Rolling Paramétrico')
    ax.plot(rolling_results_df.index, rolling_results_df[f'VaR_{alpha}_Rolling_Hist'] * 100, label=f'VaR {alpha}% Rolling Histórico')
    ax.plot(rolling_results_df.index, rolling_results_df[f'ES_{alpha}_Rolling'] * 100, label=f'ES {alpha}% Rolling Paramétrico')
    ax.plot(rolling_results_df.index, rolling_results_df[f'ES_{alpha}_Rolling_Hist'] * 100, label=f'ES {alpha}% Rolling Histórico')

ax.set_title('Rendimientos Diarios y VaR/ES Rolling Window (252 días)')
ax.set_xlabel('Fecha')
ax.set_ylabel('Valor (%)')
ax.legend()

# Mostrar gráfico
st.pyplot(fig)
