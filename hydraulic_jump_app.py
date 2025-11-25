            import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import io

# ============================================================================
# 1. CONFIGURACIÓN VISUAL "PAPER SPRINGER" (ACADÉMICO / SOBRIO)
# ============================================================================
st.set_page_config(
    page_title="Detección de Saltos - Şen (2021)",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# TIPOGRAFÍA Y ESTILO GENERAL
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = ':' 
rcParams['axes.linewidth'] = 0.8
rcParams['figure.dpi'] = 300 
rcParams['mathtext.fontset'] = 'cm'

# PALETA DE COLORES
C_OBSERVED = 'black'       
C_MODEL = '#cc0000'        
C_GRID = '#bfbfbf'         

# ============================================================================
# 2. LÓGICA MATEMÁTICA (ALGORITMO DE ŞEN)
# ============================================================================

def contar_cruces_ascendentes(serie, umbral):
    """Cuenta cruces: X_{i-1} < T y X_i >= T"""
    anterior = serie[:-1]
    actual = serie[1:]
    condicion = (anterior < umbral) & (actual >= umbral)
    return np.sum(condicion)

def calcular_perfil_cruces(serie, n_umbrales=100):
    """Genera el perfil de cruces barriendo el rango"""
    min_val = np.min(serie)
    max_val = np.max(serie)
    umbrales = np.linspace(min_val, max_val, n_umbrales)
    cruces = np.array([contar_cruces_ascendentes(serie, u) for u in umbrales])
    return umbrales, cruces

def suavizado_armonico_fourier(cruces, n_armonicos=15):
    """Ajuste de Fourier para suavizar la curva"""
    N = len(cruces)
    t = np.arange(N)
    a0 = np.mean(cruces)
    suavizado = np.full(N, a0)
    for i in range(1, n_armonicos + 1):
        arg = 2 * np.pi * i * t / N
        ai = (2/N) * np.sum(cruces * np.cos(arg))
        bi = (2/N) * np.sum(cruces * np.sin(arg))
        suavizado += ai * np.cos(arg) + bi * np.sin(arg)
    return suavizado

def detectar_saltos_significativos(umbrales, cruces_suavizados, umbral_error_min=5.0):
    """
    Detecta saltos según Şen (2021):
    
    Para CADA punto i del ajuste de Fourier, calcula:
    Error % = |Fourier[i] - Fourier[i-1]| / Fourier[i-1] × 100
    
    Si Error % > umbral, ese punto es un salto (independiente de si sube o baja).
    Esto detecta saltos en AMBOS lados de la curva.
    """
    n = len(cruces_suavizados)
    saltos_detectados = []
    
    # Calcular error relativo para CADA punto respecto al anterior
    for i in range(1, n):
        fourier_actual = cruces_suavizados[i]
        fourier_anterior = cruces_suavizados[i - 1]
        
        # Calcular error relativo: |Fourier[i] - Fourier[i-1]| / Fourier[i-1] × 100
        if fourier_anterior > 0:
            error_relativo = abs(fourier_actual - fourier_anterior) / fourier_anterior * 100
        else:
            error_relativo = 0.0
        
        # Detectar salto si error > umbral (sin importar dirección)
        if error_relativo > umbral_error_min:
            saltos_detectados.append({
                'indice': i,
                'nivel_salto': umbrales[i],
                'fourier_actual': fourier_actual,
                'fourier_anterior': fourier_anterior,
                'error_relativo': error_relativo,
                'tipo': 'Caída' if fourier_actual < fourier_anterior else 'Subida',
                'es_valido': True
            })
    
    # Ordenar por nivel (de mayor a menor)
    saltos_detectados.sort(key=lambda x: x['nivel_salto'], reverse=True)
    
    return saltos_detectados

# ============================================================================
# 3. INTERFAZ DE USUARIO
# ============================================================================

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuración")
        n_umbrales = st.slider("Niveles de Truncación (k)", 50, 300, 100, 10)
        n_armonicos = st.slider("Armónicos Fourier (m)", 1, 50, 15, 1)
        st.markdown("---")
        umbral_error = st.number_input("Error Relativo Mín. (%)", 1.0, 50.0, 5.0, 0.5, 
                                     help="Error % = |Fourier[i] - Fourier[i-1]| / Fourier[i-1] × 100")
        
        # --- CRÉDITOS AL GRUPO G ---
        st.markdown("---")
        st.markdown("### 👨‍💻 Créditos")
        st.info("**Desarrollado por:**\n\n🎓 **Grupo G**\n*Ingeniería de Recursos Hídricos*")
        
        return n_umbrales, n_armonicos, umbral_error

def cargar_datos():
    archivo = st.file_uploader("📂 Cargar Datos (Excel)", type=['xlsx', 'xls'])
    if archivo:
        try:
            df = pd.read_excel(archivo)
            if df.shape[1] < 2:
                st.error("Error: Columnas requeridas [Año, Valor]")
                return None, None
            years = df.iloc[:, 0].values
            values = df.iloc[:, 1].values
            mask = ~np.isnan(values)
            return years[mask], values[mask]
        except Exception as e:
            st.error(f"Error: {e}")
    return None, None

def generar_sinteticos():
    np.random.seed(42)
    years = np.arange(1950, 2050)
    p1 = np.random.normal(100, 10, 50)
    p2 = np.random.normal(140, 12, 50)
    return years, np.concatenate([p1, p2])

# ============================================================================
# 4. APP PRINCIPAL
# ============================================================================

def main():
    st.title("Jump Point Identification (Crossing Methodology)")
    st.markdown("**Implementation based on Şen (2021)**")
    
    n_umbrales, n_armonicos, umbral_error = render_sidebar()
    tab1, tab2 = st.tabs(["📈 Gráficos del Paper", "📄 Tabla de Datos"])
    
    with st.expander("Fuente de Datos / Data Source", expanded=True):
        col_a, col_b = st.columns([1, 3])
        with col_a:
            modo = st.radio("Input:", ["Datos Prueba", "Subir Archivo"])
        with col_b:
            years, serie = None, None
            if modo == "Subir Archivo":
                years, serie = cargar_datos()
            else:
                years, serie = generar_sinteticos()
                if modo == "Datos Prueba": st.info("Usando datos sintéticos para demostración.")

    if serie is not None and len(serie) > 2:
        # Cálculos
        umbrales, cruces_raw = calcular_perfil_cruces(serie, n_umbrales)
        cruces_smooth = suavizado_armonico_fourier(cruces_raw, n_armonicos)
        saltos = detectar_saltos_significativos(umbrales, cruces_smooth, umbral_error)
        
        # --- PESTAÑA 1: GRÁFICOS ---
        with tab1:
            if len(saltos) > 0:
                st.success(f"✅ **{len(saltos)} Salto(s) Detectado(s)**")
                for idx, s in enumerate(saltos[:5], 1):
                    st.info(f"**Salto {idx}:** Nivel = {s['nivel_salto']:.2f} | Error = {s['error_relativo']:.2f}% | {s['tipo']} (F={s['fourier_actual']:.1f}, F_prev={s['fourier_anterior']:.1f})")
            else:
                st.warning(f"⚠️ **Sin Saltos Detectados** (ningún cambio con error > {umbral_error}%)")

            # --- FIGURA 1: PANELES INDIVIDUALES ---
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            
            # (a) TIME SERIES
            ax1.plot(years, serie, color=C_OBSERVED, lw=0.8, label='Observed Series')
            ax1.scatter(years, serie, color=C_OBSERVED, s=12, marker='o', alpha=0.7)
            
            # Marcar niveles de salto en la serie temporal
            for salto in saltos[:3]:
                ax1.axhline(salto['nivel_salto'], color=C_MODEL, linestyle='--', lw=1.0, alpha=0.6)
            
            ax1.set_xlabel("Time (Years)")
            ax1.set_ylabel("Hydrological Variable")
            ax1.set_title("(a) Time Series Record", loc='center')
            ax1.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=9)

            # (b) CROSSING PROFILE
            ax2.scatter(cruces_raw, umbrales, 
                       facecolors='none', edgecolors=C_OBSERVED, marker='^', s=45, lw=0.8, 
                       label='Actual Crossings')
            
            ax2.plot(cruces_smooth, umbrales, color=C_MODEL, lw=2.0, 
                    label='Harmonic Fit (Model)')
            
            # Marcar TODOS los puntos de salto con círculos verdes
            for salto in saltos:
                ax2.plot(salto['fourier_actual'], salto['nivel_salto'], 
                        marker='o', color='green', markersize=10, 
                        markeredgecolor='darkgreen', markeredgewidth=2, alpha=0.8)
                ax2.axhline(salto['nivel_salto'], color='green', linestyle=':', lw=1.0, alpha=0.3)

            if len(saltos) > 0:
                ax2.plot([], [], marker='o', color='green', markersize=10, 
                        markeredgecolor='darkgreen', markeredgewidth=2,
                        label=f'Jump Points ({len(saltos)})', linestyle='none')

            ax2.set_xlabel("Number of Up-Crossings")
            ax2.set_ylabel("Truncation Level")
            ax2.set_title("(b) Crossing Profile", loc='center')
            ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=9)

            st.pyplot(fig1)
            
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
            st.download_button("💾 Descargar Figuras (PNG Alta Calidad)", buf1.getvalue(), "figure1_sen_paper.png", "image/png")

            # --- FIGURA 2: COMPOSITE PLOT ---
            st.markdown("---")
            st.subheader("(c) Composite Diagnostic Plot")
            
            fig2, ax_main = plt.subplots(figsize=(10, 6))
            
            ax_main.plot(years, serie, color='#555555', lw=1.0, alpha=0.8, label='Time Series')
            ax_main.set_xlabel("Time (Years)", fontsize=12)
            ax_main.set_ylabel("Magnitude / Truncation Level", fontsize=12)
            
            ax_top = ax_main.twiny()
            ax_top.plot(cruces_smooth, umbrales, color=C_MODEL, lw=2.5, label='Crossing Profile (Fourier)')
            ax_top.fill_betweenx(umbrales, 0, cruces_smooth, color=C_MODEL, alpha=0.05)
            
            ax_top.set_xlabel("Number of Crossings", fontsize=12, color=C_MODEL)
            ax_top.tick_params(axis='x', colors=C_MODEL)

            # Escala visual para no tapar datos
            max_cruces = np.max(cruces_raw) if len(cruces_raw) > 0 else 10
            ax_top.set_xlim(0, max_cruces * 2.8) 
            
            # Marcar saltos en composite
            for salto in saltos[:3]:
                ax_main.axhline(salto['nivel_salto'], color='green', linestyle='--', lw=1.2, alpha=0.6)

            lines1, labels1 = ax_main.get_legend_handles_labels()
            lines2, labels2 = ax_top.get_legend_handles_labels()
            ax_main.legend(lines1 + lines2, labels1 + labels2, loc='lower center', ncol=2, frameon=True)

            st.pyplot(fig2)
            
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
            st.download_button("💾 Descargar Composite Plot", buf2.getvalue(), "figure2_composite.png", "image/png")

        # --- PESTAÑA 2: DATOS ---
        with tab2:
            st.markdown("### Numerical Results")
            df_res = pd.DataFrame({
                "Level (T)": umbrales,
                "Actual Crossings": cruces_raw,
                "Fourier Fit": np.round(cruces_smooth, 3)
            })
            
            # Error de ajuste: (Fourier - Real) / Fourier * 100
            df_res["Fitting Error (%)"] = np.where(
                df_res["Fourier Fit"] > 0, 
                np.abs(df_res["Fourier Fit"] - df_res["Actual Crossings"]) / df_res["Fourier Fit"] * 100, 
                0
            ).round(2)
            
            # Calcular error relativo punto a punto (para detección de saltos)
            df_res["Relative Error (%)"] = 0.0
            for i in range(1, len(df_res)):
                N_actual = df_res.iloc[i]["Fourier Fit"]
                N_anterior = df_res.iloc[i-1]["Fourier Fit"]
                if N_anterior > 0:
                    df_res.loc[i, "Relative Error (%)"] = round(
                        abs(N_actual - N_anterior) / N_anterior * 100, 2
                    )
            
            # Marcar saltos detectados
            df_res["Jump Status"] = ""
            for salto in saltos:
                idx_salto = salto['indice']
                df_res.loc[idx_salto, "Jump Status"] = f"<<< JUMP (Err: {salto['error_relativo']:.2f}%)"
            
            df_res = df_res.sort_values("Level (T)", ascending=False).reset_index(drop=True)
            
            st.dataframe(df_res, use_container_width=True, height=500)
            st.download_button("📥 Descargar CSV", df_res.to_csv(index=False).encode('utf-8'), "sen_results.csv", "text/csv")
            
            # Mostrar resumen de saltos
            if len(saltos) > 0:
                st.markdown("### 📊 Resumen de Saltos Detectados")
                df_saltos = pd.DataFrame([{
                    'Nivel': s['nivel_salto'],
                    'Fourier Actual': s['fourier_actual'],
                    'Fourier Anterior': s['fourier_anterior'],
                    'Error (%)': s['error_relativo'],
                    'Tipo': s['tipo']
                } for s in saltos])
                st.dataframe(df_saltos, use_container_width=True)

    elif serie is not None and len(serie) <= 2:
        st.error("Error: La serie temporal debe tener al menos 3 puntos de datos para el análisis.")
    
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>© 2023 Grupo G - Algoritmo de Detección de Saltos Hidrológicos</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
