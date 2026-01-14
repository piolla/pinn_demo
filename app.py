"""
PINN Interactive Demo - Streamlit App
Educational demonstration of Physics-Informed Neural Networks
"""

import streamlit as st
import numpy as np
import torch
from pathlib import Path
import sys

# Add utils and models to path
sys.path.append(str(Path(__file__).parent))

from utils.data_gen import DataGenerator
from utils.visualization import (
    plot_temperature_heatmap,
    plot_temperature_animation,
    plot_comparison_at_position,
    plot_loss_history,
    plot_pinn_loss_breakdown,
    plot_error_heatmap,
    plot_generalization_comparison,
    create_metrics_table,
)
from models.pure_ai import PureAIModel, PureAITrainer
from models.pinn import PINN, PINNTrainer


# Page config
st.set_page_config(
    page_title="PINN Demo - Heat Conduction",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("🔥 Physics-Informed Neural Networks (PINN) Interactive Demo")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Simulation Settings")

        with st.expander("🔧 Problem Configuration", expanded=True):
            length_mm = st.slider("Rod Length (mm)", 50, 200, 100, 10)
            t_max = st.slider("Simulation Time (s)", 50, 300, 100, 10)

            st.divider()
            st.subheader("Boundary Conditions")
            t_left = st.slider("Left Temperature (°C)", 50, 150, 100, 5)
            t_right = st.slider("Right Temperature (°C)", 10, 50, 20, 5)
            t_initial = st.slider("Initial Temperature (°C)", 10, 50, 20, 5)

            st.divider()
            st.subheader("Material Properties")
            alpha = st.select_slider(
                "Thermal Diffusivity (m²/s)",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                value=1e-4,
                format_func=lambda x: f"{x:.0e}"
            )

        with st.expander("🎓 Training Settings"):
            ai_epochs = st.number_input("Pure AI Epochs", 100, 5000, 1000, 100)
            pinn_epochs = st.number_input("PINN Epochs", 500, 10000, 3000, 500)

            st.info("💡 More epochs = Better accuracy but slower training")

        st.divider()
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

        if run_button:
            with st.spinner("🔄 Running simulation..."):
                results = run_simulation(
                    length_mm / 1000,
                    alpha,
                    t_left,
                    t_right,
                    t_initial,
                    t_max,
                    ai_epochs,
                    pinn_epochs,
                )
                st.session_state['results'] = results
                st.session_state['params'] = {
                    'length_mm': length_mm,
                    't_max': t_max,
                    't_left': t_left,
                    't_right': t_right,
                    't_initial': t_initial,
                    'alpha': alpha,
                }
                st.success("✅ Simulation completed!")

    # Main content area
    if 'results' not in st.session_state:
        show_landing_page()
    else:
        show_results_page(st.session_state['results'], st.session_state['params'])


def show_landing_page():
    """Landing page with overview and explanations"""

    st.markdown("""
    ## 📚 What is this demo about?

    This interactive demonstration compares **three different approaches** to solving
    the 1D heat conduction problem: predicting temperature distribution in a heated metal rod.
    """)

    st.divider()

    # Three columns for three methods
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🧮 CAE Method
        **Computer-Aided Engineering**

        **How it works:**
        - Solves heat equation directly
        - Uses Finite Difference Method
        - Divides rod into mesh points
        - Computes temperature step-by-step

        **Heat Equation:**
        ```
        ∂T/∂t = α ∂²T/∂x²
        ```

        **Pros:**
        - ✅ Very accurate
        - ✅ Well-established method
        - ✅ Physics-based

        **Cons:**
        - ❌ Requires mesh setup
        - ❌ New geometry = Re-mesh & Re-compute
        - ❌ Not flexible
        """)

    with col2:
        st.markdown("""
        ### 🤖 Pure AI Method
        **Data-Driven Learning**

        **How it works:**
        - Neural network learns (x,t) → T
        - Trained on sensor measurements
        - No physics knowledge
        - Pure pattern recognition

        **Loss Function:**
        ```
        Loss = MSE(T_pred, T_measured)
        ```

        **Pros:**
        - ✅ No physics needed
        - ✅ Works with data alone

        **Cons:**
        - ❌ Needs lots of data
        - ❌ Poor generalization
        - ❌ Black box
        - ❌ May violate physics
        """)

    with col3:
        st.markdown("""
        ### 🧠 PINN Method
        **Physics-Informed Neural Network**

        **How it works:**
        - Neural network learns physics law
        - Combines data + equations
        - **Learns to satisfy PDE**
        - Uses automatic differentiation

        **Loss Function:**
        ```
        Loss = Loss_BC + Loss_IC + Loss_PDE
             = Boundary + Initial + Physics
        ```

        **Pros:**
        - ✅ Minimal data needed
        - ✅ Excellent generalization
        - ✅ Physics-consistent
        - ✅ Flexible geometry

        **Cons:**
        - ⚠️ Requires knowing PDE
        - ⚠️ More complex training
        """)

    st.divider()

    # Problem setup visualization
    st.markdown("## 🔬 The Problem")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **Scenario:** A 100mm steel rod is heated on the left end to 100°C.

        **Question:** What is the temperature at the right end after 100 seconds?

        **Physics:** Heat flows from hot to cold according to the heat equation:

        $$\\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2}$$

        where:
        - $T$ = Temperature (°C)
        - $t$ = Time (s)
        - $x$ = Position (m)
        - $\\alpha$ = Thermal diffusivity (m²/s)
        """)

    with col2:
        st.info("""
        **Boundary Conditions:**
        - Left end: 100°C (fixed)
        - Right end: 20°C (fixed)

        **Initial Condition:**
        - Entire rod: 20°C at t=0

        **Material:**
        - Steel (α = 1×10⁻⁴ m²/s)
        """)

    st.divider()

    # Key insights
    st.markdown("## 💡 Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **PINN's Secret Sauce:**

        Instead of just fitting data, PINN learns to **satisfy the physics equation**.

        The network's output must obey:
        - Boundary conditions (T at edges)
        - Initial conditions (T at t=0)
        - **The heat equation itself!**

        This is enforced through the **PDE loss term**.
        """)

    with col2:
        st.info("""
        **Why PINN Generalizes Better:**

        - **Pure AI:** Memorizes data patterns
          → Fails on unseen conditions

        - **PINN:** Learns physics law
          → Works on different geometries!

        Example: Train on 100mm rod
        → Can predict 200mm rod!
        """)

    st.divider()

    # Call to action
    st.markdown("## 🚀 Ready to Explore?")
    st.info("""
    👈 **Use the sidebar** to configure parameters and click **"Run Simulation"**

    Then explore:
    - **CAE Tab:** See how traditional simulation works
    - **AI Tab:** Watch data-driven learning in action
    - **PINN Tab:** Discover physics-informed learning
    - **Comparison Tab:** Compare all three methods
    """)


def show_results_page(results, params):
    """Main results page with tabs"""

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧮 CAE Method",
        "🤖 Pure AI Method",
        "🧠 PINN Method",
        "📊 Comparison & Results"
    ])

    with tab1:
        show_cae_tab(results, params)

    with tab2:
        show_ai_tab(results, params)

    with tab3:
        show_pinn_tab(results, params)

    with tab4:
        show_comparison_tab(results, params)


def show_cae_tab(results, params):
    """CAE Method explanation and results"""

    st.header("🧮 CAE Method: Traditional Numerical Simulation")

    # Explanation
    with st.expander("📖 What is CAE?", expanded=True):
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("""
            **Computer-Aided Engineering (CAE)** solves the heat equation directly using
            numerical methods like **Finite Difference Method (FDM)**.

            ### How It Works:

            1. **Discretize Space:** Divide rod into grid points
               ```
               x: [0, Δx, 2Δx, ..., L]
               ```

            2. **Discretize Time:** Divide time into steps
               ```
               t: [0, Δt, 2Δt, ..., T_max]
               ```

            3. **Approximate Derivatives:**
               ```
               ∂T/∂t ≈ (T_new - T_old) / Δt
               ∂²T/∂x² ≈ (T[i+1] - 2T[i] + T[i-1]) / Δx²
               ```

            4. **Update Formula:**
               ```python
               T_new[i] = T_old[i] + α*(Δt/Δx²)*(T[i+1] - 2T[i] + T[i-1])
               ```

            5. **March Forward in Time** until reaching final time
            """)

        with col2:
            st.info("""
            **Key Parameters:**

            - Grid points: 100
            - Time step: 0.005s
            - Stability: Fo < 0.5

            **Fourier Number:**
            ```
            Fo = α·Δt/Δx²
            ```
            Must be < 0.5 for stability!
            """)

    st.divider()

    # Results visualization
    st.subheader("📊 CAE Simulation Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Temperature Distribution (Space-Time Heatmap)**")
        fig = plot_temperature_heatmap(
            results['ground_truth']['x'],
            results['ground_truth']['t'],
            results['predictions']['CAE'],
            title="CAE Solution: Temperature Field",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        🔥 Hot (red) → Cold (dark). Heat diffuses from left (100°C) to right (20°C).
        Notice how temperature gradient smooths out over time.
        """)

    with col2:
        st.markdown("**Temperature Evolution Animation**")
        fig = plot_temperature_animation(
            results['ground_truth']['x'],
            results['ground_truth']['t'],
            results['predictions']['CAE'],
            title="CAE: Temperature Profile Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        ▶️ Play to see heat diffusion in action!
        Temperature at right end gradually increases as heat reaches it.
        """)

    # Key metrics
    st.divider()
    st.subheader("📈 Solution Characteristics")

    col1, col2, col3, col4 = st.columns(4)

    T_final_right = results['predictions']['CAE'][-1, -1]
    T_final_center = results['predictions']['CAE'][-1, len(results['ground_truth']['x'])//2]

    with col1:
        st.metric(
            "Initial Temp (Right)",
            f"{results['predictions']['CAE'][0, -1]:.1f}°C",
        )

    with col2:
        st.metric(
            "Final Temp (Right)",
            f"{T_final_right:.1f}°C",
            delta=f"+{T_final_right - 20:.1f}°C"
        )

    with col3:
        st.metric(
            "Final Temp (Center)",
            f"{T_final_center:.1f}°C",
        )

    with col4:
        st.metric(
            "Grid Points",
            f"{len(results['ground_truth']['x'])}",
        )

    st.success("""
    ✅ **CAE Advantages:**
    - Accurate and reliable
    - Well-established method
    - Physics-based solution

    ❌ **CAE Limitations:**
    - Requires mesh generation
    - Inflexible (new geometry = start over)
    - Computationally expensive for complex geometries
    """)


def show_ai_tab(results, params):
    """Pure AI Method explanation and training"""

    st.header("🤖 Pure AI Method: Data-Driven Learning")

    # Explanation
    with st.expander("📖 What is Pure AI?", expanded=True):
        st.markdown("""
        **Pure AI** uses a neural network to learn the mapping **(x, t) → T**
        directly from sensor measurements, **without any physics knowledge**.

        ### Architecture:

        ```
        Input: (x, t)
           ↓
        [Dense Layer 32] → Tanh
           ↓
        [Dense Layer 32] → Tanh
           ↓
        [Dense Layer 32] → Tanh
           ↓
        [Dense Layer 1]
           ↓
        Output: T (temperature)
        ```

        ### Loss Function:

        ```python
        Loss = MSE(T_predicted, T_measured)
             = Mean((T_pred - T_true)²)
        ```

        **That's it!** No physics, just data fitting.
        """)

    st.divider()

    # Training data visualization
    st.subheader("📊 Training Data")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**Sensor Measurements (Sparse & Noisy)**")

        # Show sensor data
        sensor_data = results['sensor_data']
        n_sensors = len(sensor_data['x'])
        n_measurements = len(sensor_data['t'])
        total_points = sensor_data['T'].size

        st.info(f"""
        **Available Data:**
        - Number of sensors: {n_sensors} (at both ends)
        - Measurement times: {n_measurements}
        - Total data points: {total_points}
        - Noise level: ±0.5°C

        **Data Reduction:**
        CAE computes {results['ground_truth']['T'].size:,} points
        Pure AI trains on only {total_points} points!
        → **{results['ground_truth']['T'].size // total_points}x less data**
        """)

        # Sample data table
        st.markdown("**Sample Measurements:**")
        sample_indices = np.linspace(0, n_measurements-1, 5, dtype=int)
        sample_data = {
            'Time (s)': sensor_data['t'][sample_indices],
            'Left (°C)': sensor_data['T'][sample_indices, 0],
            'Right (°C)': sensor_data['T'][sample_indices, 1],
        }
        st.dataframe(sample_data, use_container_width=True)

    with col2:
        st.markdown("**Sensor Locations**")
        st.code(f"""
Position 1: {sensor_data['x'][0]*1000:.0f} mm (Left)
Position 2: {sensor_data['x'][1]*1000:.0f} mm (Right)

Measurement Interval: Every 10 time steps
Total Measurements: {n_measurements}

Note: Real sensors have noise!
Measurements include ±0.5°C error
        """)

    st.divider()

    # Training process
    st.subheader("🎓 Training Process")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Loss History**")
        fig = plot_loss_history(
            {'Pure AI': results['losses']['Pure AI']},
            title="Pure AI Training Loss",
        )
        st.plotly_chart(fig, use_container_width=True)

        initial_loss = results['losses']['Pure AI'][0]
        final_loss = results['losses']['Pure AI'][-1]
        improvement = (1 - final_loss/initial_loss) * 100

        st.caption(f"""
        📉 Loss decreased from {initial_loss:.2f} to {final_loss:.2f}
        → {improvement:.1f}% improvement
        """)

    with col2:
        st.markdown("**What is the AI Learning?**")
        st.info("""
        The neural network learns to:

        1. **Interpolate** between sensor measurements
        2. **Recognize patterns** in temperature data
        3. **Predict** temperature at any (x,t)

        **Important:**
        - No understanding of heat flow
        - No knowledge of physics equations
        - Pure pattern matching from data

        **Problem:**
        - Struggles with extrapolation
        - Needs data everywhere
        - May violate physics laws
        """)

    st.divider()

    # Results
    st.subheader("📈 Pure AI Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Predicted Temperature Field**")
        fig = plot_temperature_heatmap(
            results['ground_truth']['x'],
            results['ground_truth']['t'],
            results['predictions']['Pure AI'],
            title="Pure AI Prediction",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Prediction Error**")
        fig = plot_error_heatmap(
            results['ground_truth']['x'],
            results['ground_truth']['t'],
            results['predictions']['Pure AI'],
            results['predictions']['CAE'],
            method="Pure AI",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Metrics
    st.markdown("**Accuracy Metrics**")
    metrics = results['metrics']['Pure AI']
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("MSE", f"{metrics['mse']:.4f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.4f}°C")
    with col3:
        st.metric("MAE", f"{metrics['mae']:.4f}°C")
    with col4:
        st.metric("Relative Error", f"{metrics['relative_error']:.2f}%")

    st.warning("""
    ⚠️ **Pure AI Limitations:**
    - Trained only on 100mm rod with specific conditions
    - Will fail on different lengths or boundary conditions
    - No guarantee of physical consistency
    - Requires lots of measurement data
    """)


def show_pinn_tab(results, params):
    """PINN Method explanation and training"""

    st.header("🧠 PINN: Physics-Informed Neural Network")

    # Core concept
    with st.expander("📖 What makes PINN special?", expanded=True):
        st.markdown("""
        ## The Big Idea

        Instead of just fitting data, **PINN learns to satisfy the physics equation itself!**

        ### The Heat Equation (PDE):

        $$\\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2}$$

        **Translation:** *"Temperature change over time = Heat spreading through space"*

        ### PINN's Innovation:

        Add the PDE as a **constraint** during training:

        ```python
        Loss_total = Loss_boundary + Loss_initial + Loss_PDE
        ```

        Where:
        - **Loss_boundary**: Temperature at edges (100°C left, 20°C right)
        - **Loss_initial**: Temperature at t=0 (20°C everywhere)
        - **Loss_PDE**: How much the network violates the heat equation

        The network learns to **satisfy physics**, not just memorize data!
        """)

    st.divider()

    # PDE explanation
    st.subheader("🔬 Understanding the PDE")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### Heat Equation Breakdown:

        $$\\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2}$$

        **Left side:** $\\frac{\\partial T}{\\partial t}$
        - How fast temperature changes over time
        - Positive = heating up, Negative = cooling down

        **Right side:** $\\alpha \\frac{\\partial^2 T}{\\partial x^2}$
        - How curved the temperature profile is
        - Sharp curve = fast heat flow
        - Flat = slow heat flow

        **Physical Meaning:**
        > *"Heat flows from hot to cold. The sharper the temperature gradient,
        > the faster heat flows."*

        ### Example:

        If temperature profile looks like: 🔥━━━━━━━❄️
        - Hot on left, cold on right
        - Steep gradient → Fast heat flow
        - Temperature in middle will rise quickly
        """)

    with col2:
        st.info(f"""
        **In This Problem:**

        **Thermal Diffusivity:**
        α = {params['alpha']:.0e} m²/s

        Higher α = Faster heat spreading

        **Boundary Conditions:**
        - T(x=0, t) = {params['t_left']}°C
        - T(x=L, t) = {params['t_right']}°C

        **Initial Condition:**
        - T(x, t=0) = {params['t_initial']}°C

        These are **constraints** that
        PINN must satisfy!
        """)

    st.divider()

    # How PINN enforces PDE
    st.subheader("⚙️ How PINN Enforces the PDE")

    with st.expander("🔍 Technical Details: Automatic Differentiation", expanded=False):
        st.markdown("""
        ### Computing PDE Residual

        PINN uses **automatic differentiation** to compute derivatives:

        ```python
        # 1. Network predicts temperature
        T = neural_network(x, t)

        # 2. Compute first derivatives (automatic!)
        dT_dx = autograd(T, x)    # ∂T/∂x
        dT_dt = autograd(T, t)    # ∂T/∂t

        # 3. Compute second derivative
        d2T_dx2 = autograd(dT_dx, x)  # ∂²T/∂x²

        # 4. PDE residual (should be zero!)
        residual = dT_dt - alpha * d2T_dx2

        # 5. PDE loss
        Loss_PDE = mean(residual²)
        ```

        **Goal:** Make residual → 0

        When residual = 0, the network satisfies the heat equation!
        """)

    st.markdown("""
    **The key insight:**

    PINN doesn't need temperature measurements everywhere. Instead:

    1. **Boundary points:** Enforce T = 100°C (left), T = 20°C (right)
    2. **Initial points:** Enforce T = 20°C at t=0
    3. **Collocation points:** Enforce PDE at random (x,t) locations

    No temperature data needed at collocation points - just enforce physics!
    """)

    st.divider()

    # Training data
    st.subheader("📊 PINN Training Data")

    col1, col2, col3 = st.columns(3)

    pinn_data = results['pinn_data']

    with col1:
        st.markdown("**Boundary Conditions**")
        st.metric("Boundary Points", len(pinn_data['boundary']['x']))
        st.caption("""
        Points at x=0 (100°C) and x=L (20°C)
        for all times
        """)

    with col2:
        st.markdown("**Initial Conditions**")
        st.metric("Initial Points", len(pinn_data['initial']['x']))
        st.caption("""
        Points at t=0 (20°C)
        for all positions
        """)

    with col3:
        st.markdown("**Collocation Points**")
        st.metric("PDE Points", len(pinn_data['collocation']['x']))
        st.caption("""
        Random interior points
        where PDE must be satisfied
        (NO temperature data!)
        """)

    total_pinn_points = (len(pinn_data['boundary']['x']) +
                         len(pinn_data['initial']['x']) +
                         len(pinn_data['collocation']['x']))

    st.success(f"""
    **Total PINN training points: {total_pinn_points}**

    But only {len(pinn_data['boundary']['x']) + len(pinn_data['initial']['x'])}
    have temperature labels!

    The rest enforce **physics law** through PDE residual.
    """)

    st.divider()

    # Training process - THE MOST IMPORTANT PART!
    st.subheader("🎓 PINN Training Process")

    st.markdown("""
    **This is where the magic happens!**

    Watch how PINN learns to satisfy the physics equation:
    """)

    # Loss breakdown
    fig = plot_pinn_loss_breakdown(results['losses']['PINN'])
    st.plotly_chart(fig, use_container_width=True)

    # Detailed loss explanation
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 Loss Components Explained")

        loss_history = results['losses']['PINN']

        st.markdown(f"""
        **Initial Losses (Epoch 1):**
        - Boundary: {loss_history['boundary'][0]:.4f}
        - Initial: {loss_history['initial'][0]:.4f}
        - **PDE: {loss_history['pde'][0]:.4f}**
        - Total: {loss_history['total'][0]:.4f}

        **Final Losses (Last Epoch):**
        - Boundary: {loss_history['boundary'][-1]:.4f}
        - Initial: {loss_history['initial'][-1]:.4f}
        - **PDE: {loss_history['pde'][-1]:.4f}** ← KEY!
        - Total: {loss_history['total'][-1]:.4f}

        **PDE Loss Reduction:**
        {loss_history['pde'][0]:.6f} → {loss_history['pde'][-1]:.6f}
        = {(1 - loss_history['pde'][-1]/loss_history['pde'][0])*100:.1f}% improvement
        """)

    with col2:
        st.markdown("### 💡 What This Means")

        st.success("""
        **Boundary Loss ↓**
        → Network learns correct temperatures at edges
                   (막대기의 양 끝 온도가 0도라고 정해져 있다면, 그 끝부분의 온도를 정확히 0으로 맞추는 과정)

        **Initial Loss ↓**
        → Network learns correct starting conditions
                   (t=0일 때, 즉 실험을 시작한 바로 그 순간의 전체 온도 분포를 맞추는 과정)

        **PDE Loss ↓** ← **CRITICAL!**
        → Network learns to **satisfy heat equation**  
        → Predictions obey physics laws(시간이 흐르면서 온도가 어떻게 변해야 하는지, 열방정식(물리법칙을 잘 지키고 있는지 검사)

        **As PDE loss → 0:**
        The network's predictions increasingly
        satisfy $\\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2}$

        **This is learning physics itself!**
       
        **보충설명:** 
                   
        → 일반적인 AI는 정답($y$)과 예측값($\hat{y}$)의 차이만 계산합니다. 하지만 PINN은 내부적으로 미분을 계산합니다.
                   
        → PINN은 $T$에 대해 자동미분(autodiff)을 사용하여 $\\frac{\partial T}{\partial t}$와 $\\frac{\partial^2 T}{\partial x^2}$를 계산합니다.
                   
          결과적으로: PDE Loss를 줄인다는 것은, AI가 내놓는 모든 예측값이 이 미분 방정식의 관계를 만족하도록 AI의 뇌(가중치)를 물리 법칙에 맞게 깎아 나가는 과정입니다.
        
        **물리학을 배우는 과정 그 자체:**
                   
        → 데이터 사이의 빈 공간을 물리로 채움: 데이터가 $t=1$과 $t=10$에만 있어도, 그 사이의 $t=5$ 지점에서 PDE Loss를 계산하면 물리 법칙에 맞는 값을 추론해냅니다.**"법칙"**이 가이드라인이 되기 때문입니다.
                   
        → 인과 관계의 학습: 일반 AI는 상관관계(Correlation)를 찾지만, PINN은 PDE를 통해 **원인과 결과(Causality)**의 물리적 메커니즘을 학습합니다.
                   
        → 자기 교정(Self-Correction): AI가 물리적으로 불가능한 답(예: 열이 차가운 곳에서 뜨거운 곳으로 이동)을 내놓으면 PDE Loss가 폭발합니다. AI는 이 "벌점"을 피하기 위해 스스로 자연의 섭리를 따르게 됩니다.                    
        """)

    # Training log interpretation
    st.divider()
    st.subheader("📋 Training Log Interpretation")

    with st.expander("🔍 How to Read PINN Training Logs", expanded=True):
        st.code(f"""
Epoch    50 | Total: {loss_history['total'][49]:.4f} | BC: {loss_history['boundary'][49]:.4f} | IC: {loss_history['initial'][49]:.4f} | PDE: {loss_history['pde'][49]:.4f}
Epoch   500 | Total: {loss_history['total'][499]:.4f} | BC: {loss_history['boundary'][499]:.4f} | IC: {loss_history['initial'][499]:.4f} | PDE: {loss_history['pde'][499]:.4f}
Epoch  1000 | Total: {loss_history['total'][999]:.4f} | BC: {loss_history['boundary'][999]:.4f} | IC: {loss_history['initial'][999]:.4f} | PDE: {loss_history['pde'][999]:.4f}
        """)

        st.markdown("""
        **Reading the log:**

        - **Total:** Sum of all losses (overall error)
        - **BC (Boundary Condition):** How well edges match 100°C and 20°C
        - **IC (Initial Condition):** How well t=0 matches 20°C
        - **PDE:** How much the solution violates heat equation

        **Good training:**
        - All losses decrease over time
        - PDE loss approaching zero ← **Most important!**
        - Smooth convergence (no wild jumps)

        **If PDE loss stays high:**
        - Network hasn't learned physics
        - Need more epochs or better learning rate
        - May need more collocation points
        """)

    st.divider()

    # Results
    st.subheader("📈 PINN Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Predicted Temperature Field**")
        fig = plot_temperature_heatmap(
            results['ground_truth']['x'],
            results['ground_truth']['t'],
            results['predictions']['PINN'],
            title="PINN Prediction",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Prediction Error**")
        fig = plot_error_heatmap(
            results['ground_truth']['x'],
            results['ground_truth']['t'],
            results['predictions']['PINN'],
            results['predictions']['CAE'],
            method="PINN",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Metrics
    st.markdown("**Accuracy Metrics**")
    metrics = results['metrics']['PINN']
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("MSE", f"{metrics['mse']:.4f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.4f}°C")
    with col3:
        st.metric("MAE", f"{metrics['mae']:.4f}°C")
    with col4:
        st.metric("Relative Error", f"{metrics['relative_error']:.2f}%")

    st.success("""
    ✅ **PINN Advantages:**
    - Learned physics law, not just data
    - Minimal temperature measurements needed
    - Predictions are physically consistent
    - **Can generalize to new conditions!**
    """)


def show_comparison_tab(results, params):
    """Comparison of all three methods"""

    st.header("📊 Method Comparison & Generalization Test")

    # Quick comparison
    st.subheader("⚡ Quick Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🧮 CAE")
        st.metric("Data Points", f"{results['ground_truth']['T'].size:,}")
        st.metric("Method", "Numerical")
        st.caption("Direct PDE solver")

    with col2:
        st.markdown("### 🤖 Pure AI")
        st.metric("Data Points", f"{results['sensor_data']['T'].size}")
        st.metric("RMSE", f"{results['metrics']['Pure AI']['rmse']:.4f}°C")
        st.caption("Data-driven learning")

    with col3:
        st.markdown("### 🧠 PINN")
        pinn_data = results['pinn_data']
        total = (len(pinn_data['boundary']['x']) +
                len(pinn_data['initial']['x']) +
                len(pinn_data['collocation']['x']))
        st.metric("Training Points", f"{total}")
        st.metric("RMSE", f"{results['metrics']['PINN']['rmse']:.4f}°C")
        st.caption("Physics-informed")

    st.divider()

    # Accuracy comparison
    st.subheader("🎯 Accuracy Comparison")

    st.markdown(create_metrics_table({
        'Pure AI': results['metrics']['Pure AI'],
        'PINN': results['metrics']['PINN'],
    }))

    # Temperature comparison
    st.subheader("📈 Temperature Predictions at Right End")

    temps = {
        'CAE (Ground Truth)': results['predictions']['CAE'][:, -1],
        'Pure AI': results['predictions']['Pure AI'][:, -1],
        'PINN': results['predictions']['PINN'][:, -1],
    }
    fig = plot_comparison_at_position(
        results['ground_truth']['t'],
        temps,
        params['length_mm'],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # GENERALIZATION TEST - THE KILLER FEATURE!
    st.subheader("🔬 Generalization Test: The PINN Advantage")

    st.markdown("""
    ## 🎯 The Ultimate Test

    **Question:** Can models trained on 100mm rod predict temperature in 200mm rod?

    This tests **true understanding** vs **memorization**.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🧮 CAE")
        st.error("""
        **Cannot generalize**

        Would need to:
        1. Re-mesh for 200mm
        2. Re-run simulation
        3. Re-compute everything

        Not flexible!
        """)

    with col2:
        st.markdown("### 🤖 Pure AI")
        ai_gen = results['generalization']['metrics']['Pure AI']
        ai_train = results['metrics']['Pure AI']
        degradation = (ai_gen['rmse'] / ai_train['rmse'] - 1) * 100

        st.warning(f"""
        **Poor generalization**

        Training RMSE: {ai_train['rmse']:.2f}°C
        200mm RMSE: {ai_gen['rmse']:.2f}°C

        Degradation: {degradation:+.1f}%

        Only learned data patterns,
        not physics!
        """)

    with col3:
        st.markdown("### 🧠 PINN")
        pinn_gen = results['generalization']['metrics']['PINN']
        pinn_train = results['metrics']['PINN']
        degradation = (pinn_gen['rmse'] / pinn_train['rmse'] - 1) * 100

        st.success(f"""
        **Excellent generalization!**

        Training RMSE: {pinn_train['rmse']:.2f}°C
        200mm RMSE: {pinn_gen['rmse']:.2f}°C

        Degradation: {degradation:+.1f}%

        Learned physics law -
        works on any length!
        """)

    # Visual comparison
    st.markdown("### 📊 200mm Rod Prediction (Final Time)")

    time_idx = -1
    fig = plot_generalization_comparison(
        results['generalization']['data']['x'],
        results['generalization']['data']['T'],
        results['generalization']['predictions'],
        time_index=time_idx,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Final insights
    st.subheader("💡 Key Takeaways")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        ### Why PINN Generalizes Better

        **Pure AI:**
        - Learned: "At x=0.1m, t=50s, T≈40°C"
        - Memorized specific data points
        - Fails when x>0.1m (outside training range)

        **PINN:**
        - Learned: "$\\frac{\\partial T}{\\partial t} = \\alpha \\frac{\\partial^2 T}{\\partial x^2}$"
        - Understands heat flow physics
        - Works for any length!

        **The difference:**
        Data patterns vs Physical laws
        """)

    with col2:
        st.success("""
        ### When to Use Each Method

        **CAE:**
        - Need high accuracy
        - One-time analysis
        - Have computational resources

        **Pure AI:**
        - Have lots of data
        - Don't know governing equations
        - Fast inference needed

        **PINN:**
        - Limited data available
        - Know governing PDE
        - Need generalization
        - Multiple similar problems
        """)

    st.divider()

    st.markdown("""
    ## 🎓 Final Thoughts

    **PINN represents a paradigm shift in scientific computing:**

    - **Traditional:** Humans solve equations → Computer executes
    - **Pure AI:** Computer learns from data → Black box
    - **PINN:** Computer learns physics laws → Transparent & generalizable

    **The future:** Combining data, physics, and AI for robust predictions!
    """)


def run_simulation(length, alpha, t_left, t_right, t_initial, t_max, ai_epochs, pinn_epochs):
    """Run complete simulation"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: CAE
    status_text.text("Step 1/5: Running CAE simulation...")
    progress_bar.progress(10)

    gen = DataGenerator(length=length, alpha=alpha, nx=100, dt=0.005)
    ground_truth = gen.generate_ground_truth(
        t_left=t_left,
        t_right=t_right,
        t_initial=t_initial,
        t_max=t_max,
    )

    # Step 2: Sensor data
    status_text.text("Step 2/5: Generating sensor measurements...")
    progress_bar.progress(25)

    sensor_data = gen.generate_sensor_data(
        ground_truth,
        sensor_positions=np.array([0.0, length]),
        measurement_interval=10,
        noise_std=0.5,
        seed=42,
    )

    # Step 3: Train Pure AI
    status_text.text("Step 3/5: Training Pure AI...")
    progress_bar.progress(40)

    pure_ai = PureAIModel(hidden_layers=[32, 32, 32])
    ai_trainer = PureAITrainer(pure_ai, learning_rate=1e-3)
    ai_loss = ai_trainer.train(sensor_data, epochs=ai_epochs, verbose=False)

    # Step 4: PINN data
    status_text.text("Step 4/5: Preparing PINN training data...")
    progress_bar.progress(55)

    pinn_data = gen.generate_pinn_training_data(
        t_left=t_left,
        t_right=t_right,
        t_initial=t_initial,
        t_max=t_max,
    )

    # Step 5: Train PINN
    status_text.text("Step 5/5: Training PINN (learning physics)...")
    progress_bar.progress(70)

    pinn = PINN(hidden_layers=[32, 32, 32], alpha=alpha)
    pinn_trainer = PINNTrainer(pinn, learning_rate=1e-3)
    pinn_loss = pinn_trainer.train(
        pinn_data,
        epochs=pinn_epochs,
        verbose=False,
    )

    progress_bar.progress(85)
    status_text.text("Making predictions...")

    # Predictions
    x_test = ground_truth['x']
    t_test = ground_truth['t']
    X_grid, T_grid = np.meshgrid(x_test, t_test)
    x_flat = X_grid.flatten()
    t_flat = T_grid.flatten()

    T_ai = pure_ai.predict(x_flat, t_flat).reshape(len(t_test), len(x_test))
    T_pinn = pinn.predict(x_flat, t_flat).reshape(len(t_test), len(x_test))

    T_true_flat = ground_truth['T'].flatten()
    ai_metrics = ai_trainer.evaluate(x_flat, t_flat, T_true_flat)
    pinn_metrics = pinn_trainer.evaluate(x_flat, t_flat, T_true_flat)

    # Generalization
    progress_bar.progress(95)
    status_text.text("Testing generalization...")

    gen_test = gen.create_generalization_test(new_length=length * 2)
    x_gen = gen_test['x']
    t_gen = gen_test['t']
    X_gen, T_gen = np.meshgrid(x_gen, t_gen)

    T_ai_gen = pure_ai.predict(X_gen.flatten(), T_gen.flatten()).reshape(len(t_gen), len(x_gen))
    T_pinn_gen = pinn.predict(X_gen.flatten(), T_gen.flatten()).reshape(len(t_gen), len(x_gen))

    ai_gen_metrics = ai_trainer.evaluate(X_gen.flatten(), T_gen.flatten(), gen_test['T'].flatten())
    pinn_gen_metrics = pinn_trainer.evaluate(X_gen.flatten(), T_gen.flatten(), gen_test['T'].flatten())

    progress_bar.progress(100)
    status_text.text("Complete!")

    return {
        'ground_truth': ground_truth,
        'sensor_data': sensor_data,
        'pinn_data': pinn_data,
        'predictions': {
            'CAE': ground_truth['T'],
            'Pure AI': T_ai,
            'PINN': T_pinn,
        },
        'losses': {
            'Pure AI': ai_loss,
            'PINN': pinn_loss,
        },
        'metrics': {
            'Pure AI': ai_metrics,
            'PINN': pinn_metrics,
        },
        'generalization': {
            'data': gen_test,
            'predictions': {
                'Pure AI': T_ai_gen,
                'PINN': T_pinn_gen,
            },
            'metrics': {
                'Pure AI': ai_gen_metrics,
                'PINN': pinn_gen_metrics,
            },
        },
    }


if __name__ == "__main__":
    main()
