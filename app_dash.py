
# ============================================================
# PI-GLSTM BATTERY PROGNOSTICS DASHBOARD
# Baseline LSTM / Objective 1 PI-GLSTM / Objective 2 Transfer
# ============================================================

from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "PI-GLSTM Battery Prognostics Dashboard"
ARTIFACT_DIR = Path("model_artifacts")

SEQ_LEN = 10
INPUT_SIZE = 4
HIDDEN_SIZE = 64
DROPOUT = 0.2

FEATURES = ["cycle", "voltage", "temperature", "capacity"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILES = {
    "Baseline LSTM": {
        "model": ARTIFACT_DIR / "standard_lstm_transfer_model.pt",
        "x_scaler": ARTIFACT_DIR / "x_scaler_standard.pkl",
        "rul_scaler": ARTIFACT_DIR / "rul_scaler_standard.pkl",
        "output_scaling": ARTIFACT_DIR / "baseline_output_scaling.pkl",
    },
    "Objective 1 PI-GLSTM": {
        "model": ARTIFACT_DIR / "pi_glstm_obj1_model.pt",
        "x_scaler": ARTIFACT_DIR / "x_scaler_obj1.pkl",
        "rul_scaler": ARTIFACT_DIR / "rul_scaler_obj1.pkl",
    },
    "Objective 2 Transfer PI-GLSTM": {
        "model": ARTIFACT_DIR / "pi_glstm_obj2_transfer_state_dict.pt",
        "x_scaler": ARTIFACT_DIR / "x_scaler_obj2.pkl",
        "rul_scaler": ARTIFACT_DIR / "rul_scaler_obj2.pkl",
    },
}

MODEL_ORDER = [
    "Baseline LSTM",
    "Objective 1 PI-GLSTM",
    "Objective 2 Transfer PI-GLSTM",
]

MODEL_COLORS = {
    "Baseline LSTM": "#64748b",
    "Objective 1 PI-GLSTM": "#2563eb",
    "Objective 2 Transfer PI-GLSTM": "#db2777",
}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔋",
    layout="wide",
)


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class BaselineLSTM(nn.Module):
    """
    Reconstructed from the baseline checkpoint parameter shapes:
        LSTM(input=4, hidden=64)
        Linear(64, 32)
        ReLU
        Linear(32, 2)

    Output columns:
        0: transformed/scaled SoH
        1: scaled RUL

    Confirm this forward pass against the original training class
    if that source code becomes available.
    """

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        fc_size=32,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, fc_size)
        self.out = nn.Linear(fc_size, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_h = h_n[-1]
        return self.out(F.relu(self.fc(last_h)))


class GuardedLSTMCell(nn.Module):
    """Guarded LSTM cell used by both PI-GLSTM models."""

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.x_gates = nn.Linear(input_size, 4 * hidden_size)
        self.h_gates = nn.Linear(
            hidden_size,
            4 * hidden_size,
            bias=False,
        )
        self.decay = nn.Linear(input_size + hidden_size, 1)

    def forward(self, x_t, state):
        h_prev, c_prev = state

        gates = self.x_gates(x_t) + self.h_gates(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_standard = f * c_prev + i * g
        h_standard = o * torch.tanh(c_standard)

        decay = F.softplus(
            self.decay(torch.cat([x_t, h_prev], dim=-1))
        )

        c_guard = c_prev[:, :1] - decay
        h_guard = torch.tanh(c_guard)

        c_new = torch.cat(
            [c_guard, c_standard[:, 1:]],
            dim=-1,
        )
        h_new = torch.cat(
            [h_guard, h_standard[:, 1:]],
            dim=-1,
        )

        return h_new, c_new


class BasePIGLSTM(nn.Module):
    """Objective 1: non-negative degradation prediction and scaled RUL."""

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden=HIDDEN_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.hidden_size = hidden
        self.cell = GuardedLSTMCell(input_size, hidden)
        self.drop = nn.Dropout(dropout)

        self.delta_head = nn.Linear(hidden, 1)
        self.rul_fc = nn.Linear(hidden, hidden // 2)
        self.rul_head = nn.Linear(hidden // 2, 1)

        self.last_guarded_cell = None
        self.last_guarded_hidden = None

    def forward(self, x):
        batch_size, steps, _ = x.shape

        h = x.new_zeros(batch_size, self.hidden_size)
        c = x.new_zeros(batch_size, self.hidden_size)

        guarded_h = []
        guarded_c = []

        for t in range(steps):
            h, c = self.cell(x[:, t, :], (h, c))
            guarded_h.append(h[:, 0])
            guarded_c.append(c[:, 0])

        self.last_guarded_hidden = torch.stack(guarded_h, dim=1)
        self.last_guarded_cell = torch.stack(guarded_c, dim=1)

        z = self.drop(h)

        delta = F.softplus(self.delta_head(z)).squeeze(-1)
        rul_scaled = self.rul_head(
            torch.relu(self.rul_fc(z))
        ).squeeze(-1)

        return delta, rul_scaled


class TransferPIGLSTM(nn.Module):
    """Objective 2: anchor-conditioned SoH and scaled RUL prediction."""

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden=HIDDEN_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.hidden_size = hidden
        self.cell = GuardedLSTMCell(input_size, hidden)
        self.drop = nn.Dropout(dropout)

        self.soh_head = nn.Linear(hidden, 1)
        self.rul_fc = nn.Linear(hidden, hidden // 2)
        self.rul_head = nn.Linear(hidden // 2, 1)

        self.last_guarded_cell = None
        self.last_guarded_hidden = None

    def forward(self, x, anchor_soh):
        batch_size, steps, _ = x.shape

        h = x.new_zeros(batch_size, self.hidden_size)
        c = x.new_zeros(batch_size, self.hidden_size)

        guarded_h = []
        guarded_c = []

        for t in range(steps):
            h, c = self.cell(x[:, t, :], (h, c))
            guarded_h.append(h[:, 0])
            guarded_c.append(c[:, 0])

        self.last_guarded_hidden = torch.stack(guarded_h, dim=1)
        self.last_guarded_cell = torch.stack(guarded_c, dim=1)

        last_h = self.drop(h)

        relative_health = torch.sigmoid(
            self.soh_head(last_h)
        ).squeeze(-1)

        soh = anchor_soh.reshape(-1).clamp(0.0, 1.0)
        soh = soh * relative_health

        rul_scaled = self.rul_head(
            torch.relu(self.rul_fc(last_h))
        ).squeeze(-1)

        return soh, rul_scaled


# ============================================================
# COLUMN STANDARDIZATION AND DATA PREPARATION
# ============================================================

COLUMN_ALIASES = {
    "cycle": ["cycle", "cycles", "cycle_number", "cycle_index"],
    "battery_id": ["battery_id", "battery", "batteryid", "unit_id"],
    "voltage": [
        "voltage", "voltage_measured", "voltage_load",
        "mean_voltage", "avg_voltage",
    ],
    "temperature": [
        "temperature", "temperature_measured", "temp",
        "mean_temperature", "avg_temperature",
    ],
    "capacity": [
        "capacity", "discharge_capacity", "measured_capacity",
        "capacity_ah",
    ],
    "soh": ["soh", "state_of_health", "actual_soh"],
    "rul": ["rul", "remaining_useful_life", "actual_rul"],
}


def standardize_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]

    rename = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = canonical
                break

    df = df.rename(columns=rename)

    if "battery_id" not in df.columns:
        df["battery_id"] = "B0005"

    return df


def prepare_cycle_data(uploaded_df):
    df = standardize_columns(uploaded_df)

    required = ["cycle", "voltage", "temperature", "capacity"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Required: "
            "cycle, voltage, temperature, capacity. Aggregate raw "
            "telemetry into cycle-level rows first."
        )

    soh_was_supplied = "soh" in df.columns
    rul_was_supplied = "rul" in df.columns

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if soh_was_supplied:
        df["soh"] = pd.to_numeric(df["soh"], errors="coerce")

    if rul_was_supplied:
        df["rul"] = pd.to_numeric(df["rul"], errors="coerce")

    df["battery_id"] = df["battery_id"].astype(str)

    df = df.dropna(subset=required)
    df = df.sort_values(
        ["battery_id", "cycle"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid cycle-level rows remain after cleaning.")

    if not soh_was_supplied:
        initial_capacity = (
            df.groupby("battery_id")["capacity"]
            .transform("first")
            .replace(0, np.nan)
        )
        df["soh"] = df["capacity"] / initial_capacity
    else:
        valid_soh = df["soh"].dropna()
        if not valid_soh.empty and valid_soh.max() > 1.5:
            df["soh"] = df["soh"] / 100.0

    if not rul_was_supplied:
        df["rul"] = (
            df.groupby("battery_id")["cycle"].transform("max")
            - df["cycle"]
        )

    return df, soh_was_supplied, rul_was_supplied


# ============================================================
# MODEL LOADING
# ============================================================

def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dictionary.")

    for key in ("model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate

    if checkpoint and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        return checkpoint

    raise ValueError(
        "Could not identify a raw state_dict or model_state_dict."
    )


def inverse_rul(scaler, scaled_value):
    value = np.asarray([[scaled_value]], dtype=np.float64)
    return float(scaler.inverse_transform(value)[0, 0])


@st.cache_resource
def load_available_models():
    """
    Load each model independently. A failure for one model does not
    prevent other valid models from loading.
    """

    models = {}
    errors = {}

    for name, paths in MODEL_FILES.items():
        required_paths = [
            paths["model"],
            paths["x_scaler"],
            paths["rul_scaler"],
        ]

        if name == "Baseline LSTM":
            required_paths.append(paths["output_scaling"])

        missing = [str(path) for path in required_paths if not path.is_file()]

        if missing:
            errors[name] = "Missing artifacts: " + ", ".join(missing)
            continue

        try:
            if name == "Baseline LSTM":
                model = BaselineLSTM(
                    input_size=INPUT_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    fc_size=32,
                )
            elif name == "Objective 1 PI-GLSTM":
                model = BasePIGLSTM(
                    input_size=INPUT_SIZE,
                    hidden=HIDDEN_SIZE,
                    dropout=DROPOUT,
                )
            else:
                model = TransferPIGLSTM(
                    input_size=INPUT_SIZE,
                    hidden=HIDDEN_SIZE,
                    dropout=DROPOUT,
                )

            checkpoint = torch.load(
                paths["model"],
                map_location="cpu",
                weights_only=True,
            )

            model.load_state_dict(
                extract_state_dict(checkpoint),
                strict=True,
            )

            model.to(DEVICE)
            model.eval()

            x_scaler = joblib.load(paths["x_scaler"])
            rul_scaler = joblib.load(paths["rul_scaler"])

            if not hasattr(x_scaler, "transform"):
                raise ValueError("Feature scaler has no transform() method.")

            if not hasattr(rul_scaler, "inverse_transform"):
                raise ValueError("RUL scaler has no inverse_transform().")

            entry = {
                "model": model,
                "x_scaler": x_scaler,
                "rul_scaler": rul_scaler,
            }

            if name == "Baseline LSTM":
                scaling = joblib.load(paths["output_scaling"])

                if not isinstance(scaling, dict):
                    raise ValueError(
                        "Baseline output scaling must be a dictionary."
                    )

                if not {"soh_scale", "soh_mean"}.issubset(scaling):
                    raise ValueError(
                        "Baseline output scaling must contain "
                        "'soh_scale' and 'soh_mean'."
                    )

                entry["soh_scale"] = float(scaling["soh_scale"])
                entry["soh_mean"] = float(scaling["soh_mean"])

                if (
                    not np.isfinite(entry["soh_scale"])
                    or entry["soh_scale"] == 0
                    or not np.isfinite(entry["soh_mean"])
                ):
                    raise ValueError(
                        "Invalid baseline SoH scaling values."
                    )

            models[name] = entry

        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    return models, errors


# ============================================================
# PREDICTIONS
# ============================================================

def run_predictions(df, artifacts, selected_battery, max_cycles=None):
    battery_df = (
        df[df["battery_id"].astype(str) == str(selected_battery)]
        .sort_values("cycle")
        .reset_index(drop=True)
        .copy()
    )

    if len(battery_df) <= SEQ_LEN:
        raise ValueError(
            f"Battery {selected_battery} has {len(battery_df)} rows. "
            f"At least {SEQ_LEN + 1} are required."
        )

    if max_cycles and max_cycles > 0:
        battery_df = battery_df.iloc[:SEQ_LEN + max_cycles].copy()

    scaled_features = {}

    for name, entry in artifacts.items():
        transformed = entry["x_scaler"].transform(
            battery_df[FEATURES]
        )

        if transformed.ndim != 2 or transformed.shape[1] != INPUT_SIZE:
            raise ValueError(
                f"{name}: scaler must output {INPUT_SIZE} features; "
                f"received shape {transformed.shape}."
            )

        scaled_features[name] = transformed.astype(np.float32)

    initial_capacity = float(battery_df.iloc[0]["capacity"])

    if not np.isfinite(initial_capacity) or initial_capacity <= 0:
        raise ValueError("Initial capacity must be positive.")

    # Objective 1 starts from SoH at the last observed input cycle.
    running_obj1_soh = float(np.clip(
        battery_df.loc[SEQ_LEN - 1, "capacity"] / initial_capacity,
        0.0,
        1.0,
    ))

    rows = []

    for target_idx in range(SEQ_LEN, len(battery_df)):
        row = {
            "battery_id": str(selected_battery),
            "cycle": float(battery_df.loc[target_idx, "cycle"]),
            "actual_soh": battery_df.loc[target_idx, "soh"],
            "actual_rul": battery_df.loc[target_idx, "rul"],
        }

        for name, entry in artifacts.items():
            x = scaled_features[name][target_idx - SEQ_LEN:target_idx]

            x_tensor = torch.as_tensor(
                x[None, :, :],
                dtype=torch.float32,
                device=DEVICE,
            )

            model = entry["model"]

            with torch.inference_mode():

                if name == "Baseline LSTM":
                    prediction = model(x_tensor)

                    if prediction.ndim != 2 or prediction.shape[1] != 2:
                        raise ValueError(
                            "Baseline model must return shape (batch, 2); "
                            f"received {tuple(prediction.shape)}."
                        )

                    soh_transformed = float(prediction[0, 0].item())
                    soh = (
                        soh_transformed * entry["soh_scale"]
                        + entry["soh_mean"]
                    )

                    rul = inverse_rul(
                        entry["rul_scaler"],
                        float(prediction[0, 1].item()),
                    )

                    row["Baseline LSTM SoH"] = float(
                        np.clip(soh, 0.0, 1.05)
                    )
                    row["Baseline LSTM RUL"] = rul

                elif name == "Objective 1 PI-GLSTM":
                    delta, rul_scaled = model(x_tensor)
                    delta_value = max(0.0, float(delta.item()))

                    running_obj1_soh = max(
                        0.0,
                        running_obj1_soh - delta_value,
                    )

                    row["Objective 1 PI-GLSTM SoH"] = running_obj1_soh
                    row["Objective 1 PI-GLSTM Delta"] = delta_value
                    row["Objective 1 PI-GLSTM RUL"] = inverse_rul(
                        entry["rul_scaler"],
                        float(rul_scaled.item()),
                    )

                elif name == "Objective 2 Transfer PI-GLSTM":
                    anchor_value = float(np.clip(
                        battery_df.loc[target_idx - 1, "capacity"]
                        / initial_capacity,
                        0.0,
                        1.0,
                    ))

                    anchor = torch.tensor(
                        [anchor_value],
                        dtype=torch.float32,
                        device=DEVICE,
                    )

                    soh, rul_scaled = model(x_tensor, anchor)

                    row["Objective 2 Transfer PI-GLSTM SoH"] = float(
                        soh.item()
                    )
                    row["Objective 2 Transfer PI-GLSTM RUL"] = inverse_rul(
                        entry["rul_scaler"],
                        float(rul_scaled.item()),
                    )

        rows.append(row)

    results = pd.DataFrame(rows)

    for col in results.columns:
        if col.endswith(" SoH"):
            results[col] = results[col].clip(0.0, 1.05)

    return results


# ============================================================
# METRICS
# ============================================================

def compute_metrics(model_name, target_name, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "Model": model_name,
            "Target": target_name,
            "Samples": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R²": np.nan,
            "MAPE (%)": np.nan,
        }

    nonzero = np.abs(y_true) > 1e-8

    mape = (
        np.mean(
            np.abs(
                (y_true[nonzero] - y_pred[nonzero])
                / np.abs(y_true[nonzero])
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    return {
        "Model": model_name,
        "Target": target_name,
        "Samples": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "MAPE (%)": mape,
    }


def build_metrics_table(results, use_soh_reference, use_rul_reference):
    metrics = []

    for model_name in MODEL_ORDER:
        for target, pred_col, actual_col, has_reference in [
            (
                "SoH",
                f"{model_name} SoH",
                "actual_soh",
                use_soh_reference,
            ),
            (
                "RUL",
                f"{model_name} RUL",
                "actual_rul",
                use_rul_reference,
            ),
        ]:
            if has_reference and pred_col in results.columns:
                metrics.append(
                    compute_metrics(
                        model_name,
                        target,
                        results[actual_col],
                        results[pred_col],
                    )
                )

    return pd.DataFrame(metrics)


# ============================================================
# CHARTS AND SUMMARY CARDS
# ============================================================

def make_prediction_chart(
    results,
    prediction_columns,
    title,
    y_label,
    percentage=False,
    show_reference=False,
    target="soh",
):
    fig, ax = plt.subplots(figsize=(12, 5.5))

    if show_reference:
        actual_col = "actual_soh" if target == "soh" else "actual_rul"

        if actual_col in results.columns:
            actual = results[actual_col].to_numpy(dtype=float)

            if percentage:
                actual = actual * 100.0

            if np.isfinite(actual).any():
                ax.plot(
                    results["cycle"],
                    actual,
                    label="Reference",
                    color="#111827",
                    linestyle="--",
                    linewidth=2.5,
                )

    for column, label in prediction_columns:
        if column not in results.columns:
            continue

        values = results[column].to_numpy(dtype=float)

        if percentage:
            values = values * 100.0

        ax.plot(
            results["cycle"],
            values,
            label=label,
            linewidth=2.2,
            color=MODEL_COLORS.get(label),
        )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Battery Cycle")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    return fig


def add_metric_cards(results):
    latest = results.iloc[-1]
    columns = st.columns(5)
 
    cards = [
        ("Current cycle", f"{latest['cycle']:.0f}"),
        (
            "Baseline SoH",
            f"{latest['Baseline LSTM SoH'] * 100:.2f}%"
            if pd.notna(latest.get("Baseline LSTM SoH", np.nan))
            else "N/A",
        ),
        (
            "Objective 1 SoH",
            f"{latest['Objective 1 PI-GLSTM SoH'] * 100:.2f}%"
            if pd.notna(latest.get("Objective 1 PI-GLSTM SoH", np.nan))
            else "N/A",
        ),
        (
            "Objective 1 RUL",
            f"{latest['Objective 1 PI-GLSTM RUL']:.1f} cycles"
            if pd.notna(latest.get("Objective 1 PI-GLSTM RUL", np.nan))
            else "N/A",
        ),
        (
            "Objective 2 SoH",
            f"{latest['Objective 2 Transfer PI-GLSTM SoH'] * 100:.2f}%"
            if pd.notna(latest.get("Objective 2 Transfer PI-GLSTM SoH", np.nan))
            else "N/A",
        ),
        (
            "Objective 2 RUL",
            f"{latest['Objective 2 Transfer PI-GLSTM RUL']:.1f} cycles"
            if pd.notna(latest.get("Objective 2 Transfer PI-GLSTM RUL", np.nan))
            else "N/A",
        ),
    ]

    
    for column, (label, value) in zip(columns, cards):
        with column:
            st.metric(label, value)


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("🔋 PI-GLSTM Battery Prognostics Dashboard")
st.caption(
    "Cycle-by-cycle SoH and RUL prediction, model comparison, "
    "live simulation, and physical-consistency evaluation."
)

with st.sidebar:
    st.header("Dashboard Settings")
    st.write(f"**Device:** `{DEVICE}`")
    st.write(f"**Sequence length:** `{SEQ_LEN}` cycles")

    uploaded_file = st.file_uploader(
        "Upload cycle-level battery CSV",
        type=["csv"],
        help=(
            "Required columns: cycle, voltage, temperature, capacity. "
            "Optional columns: battery_id, soh, rul."
        ),
    )

    st.divider()
    st.subheader("Live simulation")

    live_mode = st.checkbox(
        "Animate predictions cycle by cycle",
        value=False,
    )

    playback_delay = st.slider(
        "Delay per cycle (seconds)",
        min_value=0.05,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )

    max_display_cycles = st.number_input(
        "Maximum prediction cycles (0 = all)",
        min_value=0,
        max_value=10000,
        value=0,
        step=10,
    )


# ============================================================
# ARTIFACT STATUS
# ============================================================

st.subheader("Model artifact status")

status_cols = st.columns(3)

for i, (name, paths) in enumerate(MODEL_FILES.items()):
    with status_cols[i]:
        st.markdown(f"**{name}**")

        required = [
            paths["model"],
            paths["x_scaler"],
            paths["rul_scaler"],
        ]

        if name == "Baseline LSTM":
            required.append(paths["output_scaling"])

        missing = [p.name for p in required if not p.is_file()]

        if missing:
            st.warning("Missing: " + ", ".join(missing))
        else:
            st.success("Required artifacts found")


if uploaded_file is None:
    st.info(
        "Upload a cycle-level CSV to begin. At least 11 rows are needed "
        "per battery for the first next-cycle prediction."
    )

    st.markdown("### Expected CSV structure")

    st.dataframe(
        pd.DataFrame({
            "battery_id": ["B0005", "B0005", "B0005"],
            "cycle": [1, 2, 3],
            "voltage": [4.18, 4.15, 4.12],
            "temperature": [24.5, 24.8, 25.1],
            "capacity": [1.85, 1.83, 1.81],
            "soh": [1.00, 0.989, 0.978],
            "rul": [100, 99, 98],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.stop()


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

try:
    uploaded_df = pd.read_csv(uploaded_file)
    cycle_df, soh_reference, rul_reference = prepare_cycle_data(uploaded_df)

except Exception as exc:
    st.error(f"Could not prepare the uploaded CSV: {exc}")
    st.stop()

artifacts, model_errors = load_available_models()

with st.expander(
    "Model loading diagnostics",
    expanded=bool(model_errors),
):
    for name in artifacts:
        st.success(f"{name}: loaded successfully.")

    for name, error in model_errors.items():
        st.warning(f"{name}: {error}")

if not artifacts:
    st.error(
        "No supported model could be loaded. Check the diagnostics, "
        "checkpoint files, and scalers."
    )
    st.stop()

batteries = sorted(cycle_df["battery_id"].astype(str).unique())

selected_battery = st.sidebar.selectbox(
    "Select battery",
    batteries,
)

battery_rows = cycle_df[
    cycle_df["battery_id"].astype(str) == str(selected_battery)
]

if len(battery_rows) <= SEQ_LEN:
    st.error(
        f"Battery {selected_battery} has {len(battery_rows)} cycle rows. "
        f"At least {SEQ_LEN + 1} are required."
    )
    st.stop()

try:
    all_results = run_predictions(
        cycle_df,
        artifacts,
        selected_battery,
        max_cycles=(
            int(max_display_cycles)
            if max_display_cycles > 0
            else None
        ),
    )

except Exception as exc:
    st.error(f"Inference failed: {type(exc).__name__}: {exc}")
    st.exception(exc)
    st.stop()

if all_results.empty:
    st.warning("No predictions were generated.")
    st.stop()


# ============================================================
# SELECT AVAILABLE PREDICTION COLUMNS
# ============================================================

soh_columns = [
    (f"{name} SoH", name)
    for name in MODEL_ORDER
    if f"{name} SoH" in all_results.columns
]

rul_columns = [
    (f"{name} RUL", name)
    for name in MODEL_ORDER
    if f"{name} RUL" in all_results.columns
]


# ============================================================
# LIVE OR STATIC CHARTS
# ============================================================

if live_mode:
    st.subheader("Live prediction stream")

    status = st.empty()
    progress = st.progress(0)
    metrics_placeholder = st.empty()
    soh_chart_placeholder = st.empty()
    rul_chart_placeholder = st.empty()
    table_placeholder = st.empty()

    for i in range(1, len(all_results) + 1):
        current = all_results.iloc[:i].copy()

        with metrics_placeholder.container():
            add_metric_cards(current)

        soh_fig = make_prediction_chart(
            current,
            soh_columns,
            "Live State of Health Comparison",
            "State of Health (%)",
            percentage=True,
            show_reference=soh_reference,
            target="soh",
        )

        soh_chart_placeholder.pyplot(
            soh_fig,
            use_container_width=True,
        )
        plt.close(soh_fig)

        rul_fig = make_prediction_chart(
            current,
            rul_columns,
            "Live Remaining Useful Life Comparison",
            "Remaining Useful Life (cycles)",
            show_reference=rul_reference,
            target="rul",
        )

        rul_chart_placeholder.pyplot(
            rul_fig,
            use_container_width=True,
        )
        plt.close(rul_fig)

        table_placeholder.dataframe(
            current.tail(10).round(5),
            use_container_width=True,
            hide_index=True,
        )

        progress.progress(i / len(all_results))
        status.info(
            f"Streaming cycle {current.iloc[-1]['cycle']:.0f} "
            f"({i}/{len(all_results)})"
        )

        time.sleep(playback_delay)

    status.success("Live simulation completed.")

else:
    st.subheader("Current battery status")
    add_metric_cards(all_results)

    st.subheader("State of Health (SoH)")

    soh_fig = make_prediction_chart(
        all_results,
        soh_columns,
        "State of Health Across Battery Cycles",
        "State of Health (%)",
        percentage=True,
        show_reference=soh_reference,
        target="soh",
    )

    st.pyplot(soh_fig, use_container_width=True)
    plt.close(soh_fig)

    st.subheader("Remaining Useful Life (RUL)")

    rul_fig = make_prediction_chart(
        all_results,
        rul_columns,
        "Remaining Useful Life Across Battery Cycles",
        "Remaining Useful Life (cycles)",
        show_reference=rul_reference,
        target="rul",
    )

    st.pyplot(rul_fig, use_container_width=True)
    plt.close(rul_fig)


# ============================================================
# RESULTS TABLE AND CSV EXPORT
# ============================================================

st.subheader("Prediction results")

st.dataframe(
    all_results.round(5),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download prediction results (CSV)",
    data=all_results.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_battery}_prognostics_predictions.csv",
    mime="text/csv",
)


# ============================================================
# EVALUATION METRICS
# ============================================================

st.subheader("Model performance metrics")

if soh_reference or rul_reference:
    metric_table = build_metrics_table(
        all_results,
        use_soh_reference=soh_reference,
        use_rul_reference=rul_reference,
    )

    if not metric_table.empty:
        st.dataframe(
            metric_table.round(5),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No metrics are available for the supplied reference labels "
            "and successfully loaded models."
        )

else:
    st.info(
        "Independent SoH and RUL reference labels were not supplied. "
        "Labels derived from capacity or maximum cycle are proxies, "
        "not independent ground truth."
    )


# ============================================================
# PHYSICAL CONSISTENCY AUDIT
# ============================================================

st.subheader("SoH monotonicity audit")


def count_upward_violations(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return 0, 0

    return int((np.diff(values) > 1e-8).sum()), len(values) - 1


audit_rows = []

for name in MODEL_ORDER:
    col = f"{name} SoH"

    if col not in all_results.columns:
        continue

    violations, transitions = count_upward_violations(
        all_results[col]
    )

    audit_rows.append({
        "Model": name,
        "Upward SoH Transitions": violations,
        "Transitions Checked": transitions,
        "Strictly Non-Increasing": violations == 0,
    })

if audit_rows:
    st.dataframe(
        pd.DataFrame(audit_rows),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Objective 1 uses a recursive non-negative degradation rollout. "
    "Objective 2 predicts a bounded fraction of the preceding-cycle "
    "anchor SoH. Baseline LSTM has no enforced monotonicity constraint."
)


# ============================================================
# MODEL AND INPUT DETAILS
# ============================================================

with st.expander("Model and input details"):
    st.write("**Model input features**")
    st.code(", ".join(FEATURES))

    st.write("**Sequence length**")
    st.write(f"{SEQ_LEN} consecutive cycle-level observations")

    st.write("**Selected battery**")
    st.write(str(selected_battery))

    st.write("**Input rows for this battery**")
    st.write(len(battery_rows))

    st.write("**Prediction rows**")
    st.write(len(all_results))

    st.write("**Inference device**")
    st.write(str(DEVICE))

    for model_name, paths in MODEL_FILES.items():
        st.markdown(f"**{model_name}**")
        st.write(f"Checkpoint: `{paths['model']}`")
        st.write(f"Feature scaler: `{paths['x_scaler']}`")
        st.write(f"RUL scaler: `{paths['rul_scaler']}`")

        if model_name == "Baseline LSTM":
            st.write(
                f"SoH output scaling: `{paths['output_scaling']}`"
            )

st.success("Battery prognostics analysis completed.")
