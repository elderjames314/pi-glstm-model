import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from utils import create_sequences, load_artifacts

# Page Configuration
st.set_page_config(
    page_title="PI-GLSTM Battery Prognostics",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Styling
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔋 Physics-Informed Battery Prognostics Dashboard")
st.caption("Real-time State-of-Health (SoH) & Remaining Useful Life (RUL) Prediction System")


def is_fitted(scaler):
    return hasattr(scaler, "scale_") or hasattr(scaler, "data_min_")


def normalize_to_soh_percentage(raw_array, nominal_capacity=2.0):
    """Converts raw model outputs (capacity in Ah, fraction, or standard scores) to 0-100% SOH."""
    if len(raw_array) == 0:
        return raw_array

    array_max = np.max(raw_array)
    array_min = np.min(raw_array)

    if 1.0 < array_max <= 2.5:
        return (raw_array / nominal_capacity) * 100.0
    elif 0.2 < array_max <= 1.0 and array_min >= 0.0:
        return raw_array * 100.0
    elif array_max <= 0.2 and array_min >= 0.0:
        return raw_array * 1000.0
    elif array_min < 0.0:
        return np.clip((raw_array * 20.0) + 80.0, 0.0, 100.0)

    return raw_array


def calculate_rul(soh_array, eol_threshold):
    """Calculates sequence index where SOH first crosses EOL threshold."""
    below_eol = np.where(soh_array <= eol_threshold)[0]
    if len(below_eol) > 0:
        first_eol_idx = below_eol[0]
        remaining_from_end = max(0, first_eol_idx - (len(soh_array) - 1))
        return first_eol_idx, remaining_from_end
    else:
        decay_rate = max(0.01, (soh_array[0] - soh_array[-1]) / max(1, len(soh_array)))
        est_rul = max(0, int((soh_array[-1] - eol_threshold) / decay_rate))
        return None, est_rul


# Sidebar Configuration
st.sidebar.header("⚙️ Configuration & Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Battery Cycling CSV", type=["csv"])

seq_length = st.sidebar.slider("Sequence Window Length", 5, 30, 10)
eol_threshold = st.sidebar.slider("EOL Threshold (%)", 70.0, 85.0, 80.0, 0.5)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    with st.expander("📊 Uploaded Dataset Preview", expanded=True):
        st.dataframe(df.head(6), use_container_width=True)

    if st.button("🚀 Run Comparative Inference Pipeline", type="primary"):
        start_time = time.time()

        with st.spinner("⚡ Processing & Running Neural Models..."):
            base_model, pi_model, feature_scaler, soh_scaler, errors = load_artifacts()

            if errors:
                st.error("⚠️ Artifact Loading Issue(s):")
                for err in errors:
                    st.write(f"- `{err}`")
                st.stop()

            # 1. Extract Physical Features
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_exclude = ["cycle", "rul", "soh"]
            feature_cols = [c for c in num_cols if c.lower() not in cols_to_exclude][:feature_scaler.n_features_in_]

            raw_3_features = df[feature_cols].values
            scaled_3_features = feature_scaler.transform(raw_3_features)

            # 2. Cycle Feature & Group Identification for Concatenated CSVs
            cycle_col = [c for c in df.columns if c.lower() == "cycle"]
            cell_col = [c for c in df.columns if "battery" in c.lower() or "cell" in c.lower() or "id" in c.lower()]

            if cell_col:
                groups = df[cell_col[0]].values
            else:
                if cycle_col:
                    c_vals = df[cycle_col[0]].values
                    resets = np.where(np.diff(c_vals) < 0)[0] + 1
                    groups = np.zeros(len(df), dtype=int)
                    for r in resets:
                        groups[r:] += 1
                else:
                    groups = np.zeros(len(df), dtype=int)

            cycles_raw = df[cycle_col[0]].values.astype(np.float32) if cycle_col else np.arange(1, len(df) + 1, dtype=np.float32)
            max_c = max(np.max(cycles_raw), 100.0)
            normalized_cycle = (cycles_raw / max_c).reshape(-1, 1)

            # 3. Build Input Matrix
            feature_matrix_4d = np.hstack([scaled_3_features, normalized_cycle])

            if feature_matrix_4d.shape[0] <= seq_length:
                st.error(f"Data length ({feature_matrix_4d.shape[0]}) must be greater than sequence length ({seq_length}).")
                st.stop()

            # 4. Neural Network Inference
            X_seq = create_sequences(feature_matrix_4d, seq_length=seq_length)
            X_tensor = torch.tensor(X_seq, dtype=torch.float32)

            with torch.no_grad():
                base_raw = base_model(X_tensor).numpy().flatten()
                pi_raw = pi_model(X_tensor).numpy().flatten()

            # 5. Inverse Target Transform
            if soh_scaler is not None and is_fitted(soh_scaler):
                base_soh = soh_scaler.inverse_transform(base_raw.reshape(-1, 1)).flatten()
                pi_soh = soh_scaler.inverse_transform(pi_raw.reshape(-1, 1)).flatten()
            else:
                base_soh = base_raw
                pi_soh = pi_raw

            # 6. Normalize to Percentage (0-100%)
            base_soh = normalize_to_soh_percentage(base_soh, nominal_capacity=2.0)
            pi_soh = normalize_to_soh_percentage(pi_soh, nominal_capacity=2.0)

            # 7. Monotonic Guarding Per Battery Cell
            seq_groups = groups[seq_length:]
            guarded_pi_soh = np.copy(pi_soh)
            for g in np.unique(seq_groups):
                idx = np.where(seq_groups == g)[0]
                guarded_pi_soh[idx] = np.minimum.accumulate(pi_soh[idx])

            plot_x = np.arange(1, len(guarded_pi_soh) + 1)
            inf_time = (time.time() - start_time) * 1000

            # Calculate EOL crossing points
            base_eol_idx, base_rul = calculate_rul(base_soh, eol_threshold)
            pi_eol_idx, pi_rul = calculate_rul(guarded_pi_soh, eol_threshold)

            # Count Monotonicity Violations (positive jumps in SoH)
            base_violations = int(np.sum(np.diff(base_soh) > 0))
            raw_pi_violations = int(np.sum(np.diff(pi_soh) > 0))

        st.success(f"✅ Inference Completed in {inf_time:.2f} ms")

        # Key Comparative Metrics Banner
        # Key Comparative Metrics Banner
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Processed Sequences", len(plot_x))
        m2.metric("Baseline SoH", f"{float(base_soh[-1]):.2f} %")
        m3.metric(
            "PI-GLSTM SoH (Guarded)",
            f"{float(guarded_pi_soh[-1]):.2f} %",
            delta=f"{float(guarded_pi_soh[-1] - base_soh[-1]):.2f}% vs Base",
        )
        m4.metric(
            "Baseline EOL Cycle", 
            f"Seq #{base_eol_idx}" if base_eol_idx is not None and base_eol_idx > 0 else ("At Start" if base_eol_idx == 0 else "N/A")
        )
        m5.metric(
            "PI-GLSTM EOL Cycle", 
            f"Seq #{pi_eol_idx}" if pi_eol_idx is not None and pi_eol_idx > 0 else ("At Start" if pi_eol_idx == 0 else "N/A")
        )
        # Comparative Visualization
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1e222b")

        # 1. Baseline LSTM
        ax.plot(plot_x, base_soh, "--", color="#ff5555", alpha=0.7, label="Baseline LSTM (Unconstrained)")

        # 2. Raw PI-GLSTM Output (Shows underlying model predictions)
        ax.plot(plot_x, pi_soh, ":", color="#8be9fd", alpha=0.6, label="Raw PI-GLSTM (Unguarded)")

        # 3. Guarded PI-GLSTM Output (Monotonicity Applied)
        ax.plot(plot_x, guarded_pi_soh, color="#50fa7b", linewidth=2, label="PI-GLSTM (Monotonically Guarded)")

        # 4. EOL Threshold Line
        ax.axhline(eol_threshold, color="#f1fa8c", linestyle="--", alpha=0.8, label=f"EOL Threshold ({eol_threshold}%)")

        ax.set_xlabel("Continuous Sequence Index", color="#f8f8f2")
        ax.set_ylabel("State of Health (%)", color="#f8f8f2")
        ax.tick_params(colors="#f8f8f2")
        ax.legend(facecolor="#1e222b", edgecolor="#44475a", labelcolor="white", loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.2)

        st.pyplot(fig)

        # Presentation Context Insight
        st.info(
            "💡 **Presentation Analysis:**\n"
            "* **Monotonicity as a One-Way Ratchet:** The cyan dotted line represents the raw PI-GLSTM predictions. "
            "Because physical degradation is non-reversible ($\partial\text{SoH}/\partial t \le 0$), post-hoc monotonic guarding (green line) "
            "locks the prediction at every lowest predicted noise spike.\n"
            "* **Why Baseline Appears Higher:** Unconstrained Baseline LSTM (red) oscillates freely around a steady average, "
            "whereas any transient low noise spike in PI-GLSTM permanently depresses the guarded curve."
        )

        # Detailed Comparative Analysis Table
        st.subheader("📋 Model Comparison Analysis")

        comparison_data = {
            "Metric Parameter": [
                "Final State of Health (SoH %)",
                "Minimum Predicted SoH (%)",
                "Mean Predicted SoH (%)",
                "EOL Crossing Point (Sequence Index)",
                "Unconstrained Monotonicity Violations (Upward Spikes)",
            ],
            "Baseline LSTM": [
                f"{base_soh[-1]:.2f} %",
                f"{np.min(base_soh):.2f} %",
                f"{np.mean(base_soh):.2f} %",
                f"Seq #{base_eol_idx}" if base_eol_idx is not None else "Did not cross EOL",
                f"{base_violations} upward jumps",
            ],
            "Raw PI-GLSTM (Unguarded)": [
                f"{pi_soh[-1]:.2f} %",
                f"{np.min(pi_soh):.2f} %",
                f"{np.mean(pi_soh):.2f} %",
                f"Seq #{pi_eol_idx}" if pi_eol_idx is not None else "Did not cross EOL",
                f"{raw_pi_violations} upward jumps",
            ],
            "PI-GLSTM (Monotonically Guarded)": [
                f"{guarded_pi_soh[-1]:.2f} %",
                f"{np.min(guarded_pi_soh):.2f} %",
                f"{np.mean(guarded_pi_soh):.2f} %",
                f"Seq #{pi_eol_idx}" if pi_eol_idx is not None else "Did not cross EOL",
                "0 (Physically Enforced)",
            ],
        }

        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

else:
    st.info("👋 Welcome! Upload your battery dataset `.csv` using the sidebar menu to begin analysis.")