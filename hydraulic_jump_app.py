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

# TIPOGRAFÍA Y ESTILO GENERAL (Serif para estilo Paper)
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
rcParams['mathtext.fontset'] = 'cm' # Estilo matemático LaTeX

# PALETA DE COLORES (Springer Standard: Black & Dark Red)
C_OBSERVED = 'black'      # Datos observados (Puntos, Triángulos)
C_MODEL = '#cc0000'       # Rojo Académico (Dark Red) para Modelo/Salto
C_GRID = '#bfbfbf'        # Gris suave

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

def detectar_salto_significativo(umbrales, cruces_suavizados, umbral_error_min=5.0):
    """Detecta el mínimo y valida si la caída es significativa (>5%)"""
    idx_min = np.argmin(cruces_suavizados)
    valor_min = cruces_suavizados[idx_min]
    valor_medio = np.mean(cruces_suavizados)
    
    if valor_medio == 0: significancia = 0
    else: significancia = abs(valor_medio - valor_min) / valor_medio * 100
    
    es_significativo = significancia > umbral_error_min
    
    return {
        'indice': idx_min,
        'nivel_salto': umbrales[idx_min],
        'valor_cruces': valor_min,
        'significancia': significancia,
        'es_valido': es_significativo
    }

# ============================================================================
# 3. INTERFAZ DE USUARIO
# ============================================================================

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuración")
        n_umbrales = st.slider("Niveles de Truncación (k)", 50, 300, 100, 10)
        n_armonicos = st.slider("Armónicos Fourier (m)", 1, 50, 15, 1)
        st.markdown("---")
        umbral_error = st.number_input("Significancia Mín. Salto (%)", 1.0, 20.0, 5.0, 0.5, 
                                     help="Mínima caída % requerida para considerar un salto válido.")
        
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
            if modo == "Subir Archivo":
                years, serie = cargar_datos()
            else:
                years, serie = generar_sinteticos()
                if modo == "Datos Prueba": st.info("Usando datos sintéticos para demostración.")

    if serie is not None:
        # Cálculos
        umbrales, cruces_raw = calcular_perfil_cruces(serie, n_umbrales)
        cruces_smooth = suavizado_armonico_fourier(cruces_raw, n_armonicos)
        res = detectar_salto_significativo(umbrales, cruces_smooth, umbral_error)
        
        # --- PESTAÑA 1: GRÁFICOS ---
        with tab1:
            if res['es_valido']:
                st.success(f"✅ **Salto Detectado** | Nivel: {res['nivel_salto']:.2f} | Significancia: {res['significancia']:.2f}%")
            else:
                st.warning(f"⚠️ **Sin Salto Significativo** (Caída {res['significancia']:.2f}% < Umbral {umbral_error}%)")

            # --- FIGURA 1: PANELES INDIVIDUALES ---
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            
            # (a) TIME SERIES
            ax1.plot(years, serie, color=C_OBSERVED, lw=0.8, label='Observed Series')
            ax1.scatter(years, serie, color=C_OBSERVED, s=12, marker='o', alpha=0.7)
            
            if res['es_valido']:
                # Línea de Salto ROJA
                ax1.axhline(res['nivel_salto'], color=C_MODEL, linestyle='--', lw=1.5, 
                           label=f'Jump Level ({res["nivel_salto"]:.1f})')
            
            ax1.set_xlabel("Time (Years)")
            ax1.set_ylabel("Hydrological Variable")
            ax1.set_title("(a) Time Series Record", loc='center')
            ax1.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=9)

            # (b) CROSSING PROFILE (TRIÁNGULOS VACÍOS)
            ax2.scatter(cruces_raw, umbrales, 
                       facecolors='none', edgecolors=C_OBSERVED, marker='^', s=45, lw=0.8, 
                       label='Actual Crossings')
            
            ax2.plot(cruces_smooth, umbrales, color=C_MODEL, lw=2.0, 
                    label='Harmonic Fit (Model)')
            
            if res['es_valido']:
                ax2.axhline(res['nivel_salto'], color=C_MODEL, linestyle='--', lw=1.2, alpha=0.7)
                ax2.plot(res['valor_cruces'], res['nivel_salto'], 
                        marker='o', color=C_MODEL, markersize=7, 
                        label='Jump Point (Min)')
                
                ax2.text(res['valor_cruces'] + (max(cruces_raw)*0.05), res['nivel_salto'], 
                         f"Min @ {res['nivel_salto']:.1f}", fontsize=9, color=C_MODEL, verticalalignment='center')

            ax2.set_xlabel("Number of Up-Crossings")
            ax2.set_ylabel("Truncation Level")
            ax2.set_title("(b) Crossing Profile", loc='center')
            ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=9)

            st.pyplot(fig1)
            
            # Botón Descarga Fig 1
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
            st.download_button("💾 Descargar Figuras (PNG Alta Calidad)", buf1.getvalue(), "figure1_sen_paper.png", "image/png")

            # --- FIGURA 2: COMPOSITE PLOT (ESCALADO) ---
            st.markdown("---")
            st.subheader("(c) Composite Diagnostic Plot")
            
            fig2, ax_main = plt.subplots(figsize=(10, 6))
            
            # Serie Temporal (Abajo) - Gris Oscuro
            ax_main.plot(years, serie, color='#555555', lw=1.0, alpha=0.8, label='Time Series')
            ax_main.set_xlabel("Time (Years)", fontsize=12)
            ax_main.set_ylabel("Magnitude / Truncation Level", fontsize=12)
            
            # Cruces (Arriba) - Rojo
            ax_top = ax_main.twiny()
            ax_top.plot(cruces_smooth, umbrales, color=C_MODEL, lw=2.5, label='Crossing Profile (Fourier)')
            # Relleno rojo muy suave
            ax_top.fill_betweenx(umbrales, 0, cruces_smooth, color=C_MODEL, alpha=0.05)
            
            ax_top.set_xlabel("Number of Crossings", fontsize=12, color=C_MODEL)
            ax_top.tick_params(axis='x', colors=C_MODEL)

            # === ESCALA VISUAL: MANTENER LA CURVA A LA IZQUIERDA ===
            # El límite X superior es el triple del máximo de cruces.
            max_cruces = np.max(cruces_raw) if len(cruces_raw) > 0 else 10
            ax_top.set_xlim(0, max_cruces * 2.8) 
            
            # Línea de Salto
            if res['es_valido']:
                ax_main.axhline(res['nivel_salto'], color='black', linestyle='--', lw=1.5)
                ax_main.text(years[0], res['nivel_salto'] + (max(serie)-min(serie))*0.02, 
                             f" Jump Level ({res['nivel_salto']:.1f})", 
                             color='black', fontsize=10, fontweight='bold')

            # Leyenda Combinada
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
            
            # Fitting Error (Diferencia entre Curva y Puntos)
            df_res["Fitting Error (%)"] = np.where(
                df_res["Actual Crossings"] > 0, 
                np.abs(df_res["Fourier Fit"] - df_res["Actual Crossings"]) / df_res["Actual Crossings"] * 100, 
                0
            ).round(2)
            
            # Marca el Salto en el CSV para evitar confusiones
            df_res["Status"] = ""
            if res['es_valido']:
                # Encontramos el índice del salto y le ponemos una marca
                df_res.loc[df_res["Level (T)"] == res['nivel_salto'], "Status"] = f"<<< JUMP POINT (Sig: {res['significancia']:.2f}%)"
            
            df_res = df_res.sort_values("Level (T)", ascending=False)
            
            st.dataframe(df_res, use_container_width=True, height=500)
            st.download_button("📥 Descargar CSV", df_res.to_csv(index=False).encode('utf-8'), "sen_results.csv", "text/csv")

    # Footer Discreto
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>© 2023 Grupo G - Algoritmo de Detección de Saltos Hidrológicos</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
