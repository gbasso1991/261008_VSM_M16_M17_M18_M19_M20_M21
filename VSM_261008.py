#%% VSM 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt
import re
from uncertainties import ufloat
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


#%% Funciones
def lineal(x,m,n):
    return m*x+n
def coercive_field(H, M):
    """
    Devuelve los valores de campo coercitivo (Hc) donde la magnetización
    M cruza por cero.
    
    Parámetros:
    - H: np.array, campo magnético (en A/m o kA/m)
    - M: np.array, magnetización (en emu/g)
    
    Retorna:
    - hc_values: lista de valores Hc
    """
    H = np.asarray(H)
    M = np.asarray(M)
    hc_values = []

    for i in range(len(M)-1):
        if M[i]*M[i+1] < 0:
            h1, h2 = H[i], H[i+1]
            m1, m2 = M[i], M[i+1]

            hc = h1 - m1 * (h2 - h1) / (m2 - m1)
            hc_values.append(hc)

    return hc_values
#%% Parámetros de las muestras

nombres = ['M18', 'M20', 'M16', 'M17', 'M21', 'M19']
temperatura = {
    'M18': 215,
    'M20': 226,
    'M16': 235,
    'M17': 237,
    'M21': 243,
    'M19': 250,
}
# Concentración de magnetita
# unidades: g/L
concentracion = {
    'M18': 22.5,
    'M20': 20.0,
    'M16': 16.5,
    'M17': 23.9,
    'M21': 20.0,
    'M19': 22.0,}

# Masa del sachet con 50 uL de ferrofluido menos masa del sachet vacío
# unidades: g
masas_FF = {
    'M18': 0.1192 - 0.0677,
    'M20': 0.1162 - 0.0668,
    'M16': 0.1198 - 0.0670,
    'M17': 0.1097 - 0.0588,
    'M21': 0.1227 - 0.0714,
    'M19': 0.1249 - 0.0741,}

# Volumen de ferrofluido utilizado en cada muestra
volumen = 50e-6  # L

archivos = {nombre: f'{nombre}.txt' for nombre in nombres}
#%% Lectura de archivos y normalización
datos_VSM = {}
for nombre in nombres:
    data = np.loadtxt(os.path.join('data', archivos[nombre]),skiprows=12)
    H = data[:, 0]       # Gauss
    m = data[:, 1]       # emu
    
    masa_FF = masas_FF[nombre]# Masa de ferrofluido contenida en el sachet
    conc = concentracion[nombre]  # g/L
    conc_mm = conc/1000  # 
    masa_NP = masas_FF[nombre]*conc_mm # g

    # Magnetización normalizada por masa de magnetita
    m_norm = m / masa_NP         # emu/g

    datos_VSM[nombre] = {
        'H': H,
        'm': m,
        'masa_FF': masa_FF,
        'volumen': volumen,
        'concentracion': conc,
        'masa_NP': masa_NP,
        'm_norm': m_norm,}

#%% Ploteo normalizado por masa de magnetita

fig01, axs = plt.subplots(2, 3,figsize=(16,9),constrained_layout=True,sharex=True,sharey=True)

axs = axs.flatten()

for ax, nombre in zip(axs, nombres):
    H = datos_VSM[nombre]['H']
    m_norm = datos_VSM[nombre]['m_norm']
    masa_NP = datos_VSM[nombre]['masa_NP']

    ax.plot(H,m_norm,'-',label=f'{concentracion[nombre]} g/L\n'
              f'{masa_NP*1000:.3f} mg')

    ax.grid()
    ax.legend(ncol=1,frameon=True,shadow=True)
    ax.set_title(f'{nombre} - T = {temperatura[nombre]} °C',loc='left')
# Etiquetas de los ejes
for ax in axs[3:]:
    ax.set_xlabel('H (G)')

for ax in [axs[0], axs[3]]:
    ax.set_ylabel('m (emu/g)')
plt.suptitle('VSM - Magnetización normalizada por masa de Fe3O4\n'+str(), fontsize=16)
plt.savefig('VSM_magnetizacion_normalizada.png',dpi=300)
plt.show()


#%% Fiteo con fit3
resultados_fit = {}
H_fit_arrays = {}
m_fit_arrays = {}

for nombre in nombres:
    H = datos_VSM[nombre]['H']
    m = datos_VSM[nombre]['m']
    mass = datos_VSM[nombre]['masa_NP']
    # Obtengo la curva anhistérica
    H_anhist, m_anhist = mt.anhysteretic(H, m)
    # Ajuste normalizando por masa
    fit = fit3.session(H_anhist,m_anhist,fname=nombre,
        divbymass=True,mass=mass)

    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()

    fit.free('sig0')
    fit.free('mu0')
    fit.set_yE_as('sep')
    fit.fit()
    fit.update()

    fit.save()
    fit.print_pars()

    # Contribución lineal
    C = fit.params['C'].value
    dc = fit.params['dc'].value

    linear_contrib = lineal(fit.X, C, dc)
    m_fit_sin_lineal = fit.Yfit - linear_contrib

    resultados_fit[nombre] = {
        'H_anhist': H_anhist,
        'm_anhist': m_anhist,
        'H_fit': fit.X,
        'm_fit': fit.Yfit,
        'm_fit_sin_lineal': m_fit_sin_lineal,
        'linear_contrib': linear_contrib,
        'Ms': fit.derived_parameters()['m_s'],
        'fit': fit}
    H_fit_arrays[nombre] = fit.X
    m_fit_arrays[nombre] = fit.Yfit

#%% Ploteo individual de los fits

for nombre in nombres:
    H = datos_VSM[nombre]['H']
    m_norm = datos_VSM[nombre]['m_norm']
    masa_NP = datos_VSM[nombre]['masa_NP']
    conc = datos_VSM[nombre]['concentracion']

    fit = resultados_fit[nombre]['fit']
    H_fit = resultados_fit[nombre]['H_fit']
    m_fit = resultados_fit[nombre]['m_fit']
    Ms = resultados_fit[nombre]['Ms']

    fig, ax = plt.subplots(figsize=(8, 6),constrained_layout=True)

    ax.plot(H,m_norm,'o-',c='C0',alpha=0.4,
        label=f'Datos originales\n'
              f'C = {conc} g/L\n'
              f'm = {masa_NP*1000:.3f} mg')

    ax.plot(H_fit,m_fit,'.-',c='C1',
        label=f'Ajuste\n'
              f'm$_s$ = {Ms:.2uS} emu/g\n'
              f'$<\\mu_\\mu>$ = '
              f'{fit.derived_parameters()["<mu>_mu"]:.1uS} $\\mu_B$')
    ax.set_ylabel('m (emu/g)')
    ax.set_xlabel('H (G)')
    ax.legend(title=f'{nombre} — T = {temperatura[nombre]} °C',
        ncol=1,frameon=True,shadow=True,labelspacing=1.2)
    ax.grid()
    #ax.set_title(nombre)

    # Inset
    axins = inset_axes(ax,width="40%",height="40%",loc='lower right',bbox_to_anchor=(-0.01, 0.08, 0.98, 1),bbox_transform=ax.transAxes,borderpad=0)
    axins.plot(H,m_norm,'o-',c='C0',alpha=0.4)
    axins.plot(H_fit,m_fit,'.-',c='C1')
    axins.set_xlim(-30, 30)
    axins.set_ylim(-10, 10)
    axins.grid()
    plt.savefig(f'{nombre}_fit.png',dpi=300)
    plt.show()


# %%# %% Ploteo de los fits completos

fig02, axs = plt.subplots(2, 3,figsize=(16, 9),constrained_layout=True,sharex=True,sharey=True)
axs = axs.flatten()
for ax, nombre in zip(axs, nombres):
    H = datos_VSM[nombre]['H']
    m_norm = datos_VSM[nombre]['m_norm']

    H_fit = resultados_fit[nombre]['H_fit']
    m_fit = resultados_fit[nombre]['m_fit']

    fit = resultados_fit[nombre]['fit']
    Ms = resultados_fit[nombre]['Ms']
    mu_mu = fit.derived_parameters()['<mu>_mu']

    # Datos experimentales
    ax.plot(H,m_norm,'o-',alpha=0.35,label='Datos')

    # Fit completo
    ax.plot(H_fit,m_fit,'.-',
        label=f'Ajuste\n'
              f'm$_s$ = {Ms:.2uS} emu/g\n'
              f'$<\\mu_\\mu>$ = {mu_mu:.2uS} $\\mu_B$')

    ax.grid()

    ax.legend(ncol=1,frameon=True,shadow=True,loc='upper left')
    ax.set_title(f'{nombre} - T = {temperatura[nombre]} °C',loc='left')

for ax in axs[3:]:
    ax.set_xlabel('H (G)')

for ax in [axs[0], axs[3]]:
    ax.set_ylabel('m (emu/g)')

plt.suptitle('VSM - Ajuste de magnetización',fontsize=16)
plt.savefig('VSM_fits_completos.png',dpi=300)
plt.show()

# %% Ploteo de los fits sin contribución lineal
fig03, axs = plt.subplots(2, 3,figsize=(16, 9),constrained_layout=True,sharex=True,sharey=True)
axs = axs.flatten()
for ax, nombre in zip(axs, nombres):

    H_fit = resultados_fit[nombre]['H_fit']
    m_fit_sin_lineal = resultados_fit[nombre]['m_fit_sin_lineal']

    fit = resultados_fit[nombre]['fit']
    Ms = resultados_fit[nombre]['Ms']
    mu_mu = fit.derived_parameters()['<mu>_mu']

    # Fit sin contribución lineal
    ax.plot(H_fit,m_fit_sin_lineal,'-',
        label=f'm$_s$ = {Ms:.2uS} emu/g\n'
              f'$<\\mu_\\mu>$ = {mu_mu:.2uS} $\\mu_B$')

    ax.grid()

    ax.legend(ncol=1,frameon=True,shadow=True,loc='upper left')

    ax.set_title(f'{nombre} - T = {temperatura[nombre]} °C',loc='left')

for ax in axs[3:]:
    ax.set_xlabel('H (G)')

for ax in [axs[0], axs[3]]:
    ax.set_ylabel('m (emu/g)')

plt.suptitle('VSM - Fits sin contribución lineal',fontsize=16)
plt.savefig('VSM_fits_sin_lineal.png',dpi=300)
plt.show()
# %% Ms vs temperatura de síntesis
T = np.array([temperatura[nombre] for nombre in nombres])
Ms = np.array([resultados_fit[nombre]['Ms'].n for nombre in nombres])
Ms_err = np.array([resultados_fit[nombre]['Ms'].s for nombre in nombres])

fig04, ax = plt.subplots(figsize=(8, 5),constrained_layout=True)

ax.errorbar(T,Ms,yerr=Ms_err,fmt='o-',capsize=4)
ax.set_xlabel('Temperatura de síntesis (°C)')
ax.set_ylabel('$m_s$ (emu/g)')

ax.grid()

plt.title('Magnetización de saturación vs temperatura de síntesis')

plt.savefig('Ms_vs_temperatura_sintesis.png',dpi=300)

plt.show()
# %%
