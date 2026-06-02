
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ==================================================
# CONFIGURACION
# ==================================================

st.set_page_config(
    page_title="Centro de Navegación GPS",
    layout="wide"
)

st.title("🛰️ Centro de Navegación GPS para Drones")

# ==================================================
# PANEL LATERAL
# ==================================================

st.sidebar.header("🎮 Control del Dron")

x = st.sidebar.slider("Posición X", 0, 200, 30)
y = st.sidebar.slider("Posición Y", 0, 200, 40)
z = st.sidebar.slider("Posición Z", 0, 150, 20)

receiver = np.array([x, y, z])

st.sidebar.success(
"""
✅ GPS ACTIVO

🛰️ 4 Satélites Detectados

📡 Señal Estable
"""
)

# ==================================================
# SATELITES
# ==================================================

satellites = np.array([
    [0, 0, 100],
    [120, 20, 80],
    [20, 140, 130],
    [150, 100, 60]
])

# ==================================================
# DISTANCIAS
# ==================================================

distances = np.linalg.norm(
    satellites - receiver,
    axis=1
)

# ==================================================
# SISTEMA AX = B
# ==================================================

x1, y1, z1 = satellites[0]
d1 = distances[0]

A = []
B = []

for i in range(1, 4):

    xi, yi, zi = satellites[i]
    di = distances[i]

    A.append([
        2 * (xi - x1),
        2 * (yi - y1),
        2 * (zi - z1)
    ])

    B.append(
        d1**2 - di**2
        + xi**2 - x1**2
        + yi**2 - y1**2
        + zi**2 - z1**2
    )

A = np.array(A)
B = np.array(B)

augmented = np.column_stack((A, B))

position_calculated = np.linalg.solve(A, B)

detA = np.linalg.det(A)

inverseA = np.linalg.inv(A)

adjA = detA * inverseA

# ==================================================
# VECTORES
# ==================================================

vectors = receiver - satellites

v1 = vectors[0]
v2 = vectors[1]

cos_theta = np.dot(v1, v2) / (
    np.linalg.norm(v1)
    * np.linalg.norm(v2)
)

angle = np.degrees(
    np.arccos(cos_theta)
)

projection = (
    np.dot(v1, v2)
    /
    np.dot(v2, v2)
) * v2

error = np.linalg.norm(
    receiver - position_calculated
)

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3 = st.tabs([
    "🛰️ GPS",
    "📐 Álgebra Lineal",
    "➡️ Vectores"
])

# ==================================================
# TAB GPS
# ==================================================

with tab1:

    col1, col2 = st.columns([2,1])

    fig = go.Figure()

    # SATELITES

    fig.add_trace(
        go.Scatter3d(
            x=satellites[:,0],
            y=satellites[:,1],
            z=satellites[:,2],
            mode='markers+text',
            text=[
                f'🛰️ S1\n{distances[0]:.1f}',
                f'🛰️ S2\n{distances[1]:.1f}',
                f'🛰️ S3\n{distances[2]:.1f}',
                f'🛰️ S4\n{distances[3]:.1f}'
            ],
            textposition='top center',
            marker=dict(
                size=12,
                color='cyan'
            ),
            name='Satélites'
        )
    )

    # DRON

    fig.add_trace(
        go.Scatter3d(
            x=[receiver[0]],
            y=[receiver[1]],
            z=[receiver[2]],
            mode='markers+text',
            text=['🚁 Dron'],
            textposition='top center',
            marker=dict(
                size=18,
                color='red'
            ),
            name='Dron'
        )
    )

    # SEÑALES GPS

    colors = [
        "red",
        "green",
        "cyan",
        "orange"
    ]

    for i, sat in enumerate(satellites):

        fig.add_trace(
            go.Scatter3d(
                x=[sat[0], receiver[0]],
                y=[sat[1], receiver[1]],
                z=[sat[2], receiver[2]],
                mode='lines',
                line=dict(
                    width=6,
                    color=colors[i]
                ),
                showlegend=False
            )
        )

    fig.update_layout(
        height=700,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    with col1:

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.metric(
            "Determinante",
            f"{detA:.0f}"
        )

        st.metric(
            "Ángulo",
            f"{angle:.2f}°"
        )

        st.metric(
            "Error GPS",
            f"{error:.8f}"
        )

        if abs(detA) > 1e-6:

            st.success(
                "✅ Existe solución única"
            )

        else:

            st.error(
                "❌ Matriz singular"
            )

        st.subheader("🚁 Posición Real")

        st.success(
            f"""
            X = {receiver[0]}

            Y = {receiver[1]}

            Z = {receiver[2]}
            """
        )

        st.subheader("🛰️ Posición Calculada")

        st.info(
            f"""
            X = {position_calculated[0]:.2f}

            Y = {position_calculated[1]:.2f}

            Z = {position_calculated[2]:.2f}
            """
        )

# ==================================================
# TAB ALGEBRA
# ==================================================

with tab2:

    st.header("Modelo Matemático")

    st.latex(r"AX=B")

    st.latex(r"X=A^{-1}B")

    st.info(
        """
        El sistema GPS genera un sistema
        de ecuaciones lineales que permite
        determinar la posición del dron.
        """
    )

    st.subheader("Matriz A")
    st.write(A)

    st.subheader("Vector B")
    st.write(B)

    st.subheader("Matriz Aumentada [A|B]")
    st.write(augmented)

    st.subheader("Determinante")
    st.write(detA)

    st.subheader("Inversa")
    st.write(inverseA)

    st.subheader("Adjunta")
    st.write(adjA)

    st.subheader("Método de Gauss-Jordan")

    st.info(
        """
        La matriz aumentada [A|B] puede
        reducirse mediante operaciones
        elementales por filas hasta obtener
        la forma reducida y resolver la
        posición del dron.
        """
    )

# ==================================================
# TAB VECTORES
# ==================================================

with tab3:

    st.subheader("Vectores Satélite → Dron")
    st.write(vectors)

    st.subheader("Ángulo entre S1 y S2")
    st.write(angle)

    st.subheader("Proyección de S1 sobre S2")
    st.write(projection)

    st.latex(
        r"\cos(\theta)=\frac{u\cdot v}{||u||\,||v||}"
    )

    st.latex(
        r"Proj_v(u)=\frac{u\cdot v}{||v||^2}v"
    )
