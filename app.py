
from __future__ import annotations

import pickle
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Battery Prognostics | Streaming",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "model_artifacts"

SEQ_LEN = 10
INPUT_SIZE = 4
HIDDEN_SIZE = 64

FEATURES = [
    "cycle",
    "voltage",
    "temperature",
    "capacity",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
        "initial_capacity": ARTIFACT_DIR / "initial_capacity_obj2.pkl",
    },
}

MODEL_COLORS = {
    "Baseline LSTM": "#2563EB",
    "Objective 1 PI-GLSTM": "#16A34A",
    "Objective 2 Transfer PI-GLSTM": "#9333EA",
}

SOH_COLUMNS = {
    "Baseline LSTM": "Baseline LSTM SoH",
    "Objective 1 PI-GLSTM": "Objective 1 PI-GLSTM SoH",
    "Objective 2 Transfer PI-GLSTM": "Objective 2 Transfer PI-GLSTM SoH",
}

RUL_COLUMNS = {
    "Baseline LSTM": "Baseline LSTM RUL",
    "Objective 1 PI-GLSTM": "Objective 1 PI-GLSTM RUL",
    "Objective 2 Transfer PI-GLSTM": "Objective 2 Transfer PI-GLSTM RUL",
}


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class BaselineLSTM(nn.Module):
    """
    Baseline architecture inferred from the exported state-dict
    tensor dimensions:

        LSTM(input=4, hidden=64)
        Linear(64, 32)
        ReLU
        Linear(32, 2)

    Output:
        [:, 0] = normalized SoH
        [:, 1] = scaled RUL
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 32)
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        z = F.relu(self.fc(last_hidden))
        return self.out(z)


class GuardedLSTMCell(nn.Module):
    """
    PI-GLSTM cell with a guarded degradation state.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.x_gates = nn.Linear(
            input_size,
            4 * hidden_size,
        )
        self.h_gates = nn.Linear(
            hidden_size,
            4 * hidden_size,
            bias=False,
        )
        self.decay = nn.Linear(
            input_size + hidden_size,
            1,
        )

    def forward(self, x_t, state):
        h_prev, c_prev = state

        gates = (
            self.x_gates(x_t)
            + self.h_gates(h_prev)
        )

        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_standard = f * c_prev + i * g
        h_standard = o * torch.tanh(c_standard)

        decay = F.softplus(
            self.decay(
                torch.cat([x_t, h_prev], dim=-1)
            )
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
    """
    Objective 1:
    Non-negative degradation increment and scaled RUL output.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.cell = GuardedLSTMCell(
            input_size,
            hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

        self.delta_head = nn.Linear(
            hidden_size,
            1,
        )

        self.rul_fc = nn.Linear(
            hidden_size,
            hidden_size // 2,
        )

        self.rul_head = nn.Linear(
            hidden_size // 2,
            1,
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        h = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device,
        )
        c = torch.zeros_like(h)

        for t in range(seq_len):
            h, c = self.cell(
                x[:, t, :],
                (h, c),
            )

        h = self.dropout(h)

        delta = F.softplus(
            self.delta_head(h)
        ).squeeze(-1)

        rul_scaled = self.rul_head(
            F.relu(self.rul_fc(h))
        ).squeeze(-1)

        return delta, rul_scaled


class TransferPIGLSTM(nn.Module):
    """
    Objective 2:
    Transfer PI-GLSTM with a relative-health head and RUL head.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.cell = GuardedLSTMCell(
            input_size,
            hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

        self.soh_head = nn.Linear(
            hidden_size,
            1,
        )

        self.rul_fc = nn.Linear(
            hidden_size,
            hidden_size // 2,
        )

        self.rul_head = nn.Linear(
            hidden_size // 2,
            1,
        )

    def forward(self, x, anchor_soh):
        batch_size, seq_len, _ = x.shape

        h = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device,
        )
        c = torch.zeros_like(h)

        for t in range(seq_len):
            h, c = self.cell(
                x[:, t, :],
                (h, c),
            )

        h = self.dropout(h)

        relative_health = torch.sigmoid(
            self.soh_head(h)
        ).squeeze(-1)

        soh = (
            torch.clamp(anchor_soh, 0.0, 1.0)
            * relative_health
        )

        rul_scaled = self.rul_head(
            F.relu(self.rul_fc(h))
        ).squeeze(-1)

        return soh, rul_scaled


# ============================================================
# ARTIFACT LOADING
# ============================================================

def load_pickle(path: Path) -> Any:
    """
    Load an exported scaler or pickle artifact.
    """
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
        )

    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
        ):
            if key in checkpoint and isinstance(
                checkpoint[key], dict
            ):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported checkpoint format: {path.name}"
        )

    # Remove DataParallel prefixes if present.
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def extract_scalar(value: Any) -> float:
    """
    Read a scalar or scalar-containing exported artifact.
    """
    if isinstance(value, dict):
        for key in (
            "initial_capacity",
            "capacity",
            "value",
        ):
            if key in value:
                return extract_scalar(value[key])

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if len(arr):
            return float(arr[0])

    return float(value)


def load_baseline_output_scaling(path: Path) -> tuple[float, float]:
    """
    Expected export:
        {"soh_scale": ..., "soh_mean": ...}

    Also accepts a two-element sequence:
        (soh_scale, soh_mean)
    """
    obj = load_pickle(path)

    if isinstance(obj, dict):
        if "soh_scale" in obj and "soh_mean" in obj:
            return (
                float(obj["soh_scale"]),
                float(obj["soh_mean"]),
            )

        if "scale" in obj and "mean" in obj:
            return (
                float(obj["scale"]),
                float(obj["mean"]),
            )

    if isinstance(obj, (tuple, list, np.ndarray)):
        values = np.asarray(obj, dtype=float).reshape(-1)

        if len(values) >= 2:
            return float(values[0]), float(values[1])

    raise ValueError(
        "Baseline output scaling must contain "
        "'soh_scale' and 'soh_mean', or a two-value sequence."
    )


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    """
    Models are loaded once and reused between Streamlit reruns.

    A model with missing artifacts is marked unavailable while
    the other models can still run.
    """
    models = {}
    errors = {}

    architectures = {
        "Baseline LSTM": BaselineLSTM,
        "Objective 1 PI-GLSTM": BasePIGLSTM,
        "Objective 2 Transfer PI-GLSTM": TransferPIGLSTM,
    }

    for name, paths in MODEL_FILES.items():
        try:
            required = [
                paths["model"],
                paths["x_scaler"],
                paths["rul_scaler"],
            ]

            if name == "Baseline LSTM":
                required.append(paths["output_scaling"])

            missing = [
                p.name for p in required if not p.exists()
            ]

            if missing:
                raise FileNotFoundError(
                    "Missing artifact(s): " + ", ".join(missing)
                )

            model = architectures[name]()

            state_dict = load_checkpoint(paths["model"])

            model.load_state_dict(
                state_dict,
                strict=True,
            )

            model.to(DEVICE)
            model.eval()

            x_scaler = load_pickle(paths["x_scaler"])
            rul_scaler = load_pickle(paths["rul_scaler"])

            item = {
                "model": model,
                "x_scaler": x_scaler,
                "rul_scaler": rul_scaler,
            }

            if name == "Baseline LSTM":
                (
                    item["soh_scale"],
                    item["soh_mean"],
                ) = load_baseline_output_scaling(
                    paths["output_scaling"]
                )

            if name == "Objective 2 Transfer PI-GLSTM":
                capacity_path = paths.get("initial_capacity")

                if capacity_path and capacity_path.exists():
                    item["initial_capacity"] = extract_scalar(
                        load_pickle(capacity_path)
                    )
                else:
                    item["initial_capacity"] = None

            models[name] = item

        except Exception as exc:
            errors[name] = str(exc)

    return models, errors


# ============================================================
# DATASET PREPARATION
# ============================================================

def normalise_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
    )


def prepare_battery_data(raw: pd.DataFrame):
    """
    Normalise common column aliases, sort cycles, and create
    optional reference values.

    Explicitly supplied SoH/RUL values are distinguished from
    values derived as proxies.
    """
    df = raw.copy()
    df.columns = [
        normalise_column_name(c)
        for c in df.columns
    ]

    aliases = {
        "cycle": [
            "cycle",
            "cycles",
            "cycle_number",
            "cycle_index",
        ],
        "voltage": [
            "voltage",
            "voltage_measured",
            "terminal_voltage",
            "voltage_v",
        ],
        "temperature": [
            "temperature",
            "temperature_measured",
            "temp",
            "temperature_c",
        ],
        "capacity": [
            "capacity",
            "capacity_measured",
            "discharge_capacity",
            "battery_capacity",
        ],
        "soh": [
            "soh",
            "state_of_health",
            "state_health",
            "soh_percent",
            "capacity_retention",
        ],
        "rul": [
            "rul",
            "remaining_useful_life",
            "remaining_cycles",
            "remaining_life",
        ],
    }

    rename = {}

    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = canonical
                break

    df = df.rename(columns=rename)

    missing_features = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The uploaded CSV is missing required columns: "
            + ", ".join(missing_features)
            + ". Expected cycle, voltage, temperature, capacity."
        )

    df = df.copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    has_soh_reference = "soh" in df.columns
    has_rul_reference = "rul" in df.columns

    if has_soh_reference:
        df["soh"] = pd.to_numeric(
            df["soh"],
            errors="coerce",
        )

    if has_rul_reference:
        df["rul"] = pd.to_numeric(
            df["rul"],
            errors="coerce",
        )

    df = (
        df.dropna(subset=FEATURES)
        .sort_values("cycle")
        .drop_duplicates(subset=["cycle"], keep="last")
        .reset_index(drop=True)
    )

    if len(df) <= SEQ_LEN:
        raise ValueError(
            f"At least {SEQ_LEN + 1} valid cycles are required."
        )

    if (df["capacity"] <= 0).all():
        raise ValueError(
            "Capacity values must include positive measurements."
        )

    initial_capacity = float(df["capacity"].iloc[0])

    if initial_capacity <= 0:
        raise ValueError(
            "The first valid capacity must be positive."
        )

    # Measured SoH is preferred when supplied.
    if has_soh_reference:
        df["soh"] = df["soh"].where(
            df["soh"].notna(),
            np.nan,
        )

        valid_soh = df["soh"].dropna()

        if not valid_soh.empty and valid_soh.max() > 1.5:
            df["soh"] = df["soh"] / 100.0

        df["soh"] = df["soh"].clip(0.0, 1.2)
        df["_soh_reference"] = df["soh"]
        df["_soh_reference_label"] = "Supplied SoH reference"
    else:
        df["_soh_reference"] = (
            df["capacity"] / initial_capacity
        ).clip(0.0, 1.2)
        df["_soh_reference_label"] = (
            "Capacity-derived SoH proxy"
        )

    # Supplied RUL is preferred when available.
    if has_rul_reference:
        df["_rul_reference"] = df["rul"]
        df["_rul_reference_label"] = "Supplied RUL reference"
    else:
        max_cycle = float(df["cycle"].max())
        df["_rul_reference"] = (
            max_cycle - df["cycle"]
        ).clip(lower=0)
        df["_rul_reference_label"] = (
            "Cycle-derived RUL proxy"
        )

    return df, initial_capacity, has_soh_reference, has_rul_reference


def inverse_rul(scaler, value: float) -> float:
    arr = np.asarray([[value]], dtype=float)
    return float(scaler.inverse_transform(arr)[0, 0])


def scale_input(scaler, raw_window: pd.DataFrame) -> torch.Tensor:
    values = raw_window[FEATURES].to_numpy(dtype=np.float32)
    scaled = scaler.transform(values)

    return torch.as_tensor(
        scaled,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)


# ============================================================
# SINGLE-CYCLE INFERENCE
# ============================================================

@torch.inference_mode()
def predict_one_cycle(
    models: dict,
    df: pd.DataFrame,
    step_index: int,
    previous_obj1_soh: float,
    dataset_initial_capacity: float,
):
    """
    Process one sliding window and return this cycle's outputs.

    step_index 0 corresponds to the first complete 10-step window.
    """
    end_idx = SEQ_LEN - 1 + step_index
    start_idx = end_idx - SEQ_LEN + 1

    raw_window = df.iloc[start_idx:end_idx + 1]

    if len(raw_window) != SEQ_LEN:
        raise ValueError("Invalid sliding-window length.")

    row = {
        "cycle": float(df.iloc[end_idx]["cycle"]),
        "soh_reference": float(
            df.iloc[end_idx]["_soh_reference"]
        ) if pd.notna(
            df.iloc[end_idx]["_soh_reference"]
        ) else np.nan,
        "rul_reference": float(
            df.iloc[end_idx]["_rul_reference"]
        ) if pd.notna(
            df.iloc[end_idx]["_rul_reference"]
        ) else np.nan,
    }

    next_obj1_soh = previous_obj1_soh

    # --------------------------------------------------------
    # Baseline LSTM
    # --------------------------------------------------------
    if "Baseline LSTM" in models:
        try:
            item = models["Baseline LSTM"]
            x = scale_input(item["x_scaler"], raw_window)

            pred = item["model"](x)[0].detach().cpu().numpy()

            soh = (
                float(pred[0]) * item["soh_scale"]
                + item["soh_mean"]
            )

            rul = inverse_rul(
                item["rul_scaler"],
                float(pred[1]),
            )

            row[SOH_COLUMNS["Baseline LSTM"]] = float(
                np.clip(soh, 0.0, 1.05)
            )
            row[RUL_COLUMNS["Baseline LSTM"]] = max(
                0.0, rul
            )

        except Exception as exc:
            row[SOH_COLUMNS["Baseline LSTM"]] = np.nan
            row[RUL_COLUMNS["Baseline LSTM"]] = np.nan
            row["Baseline LSTM error"] = str(exc)

    # --------------------------------------------------------
    # Objective 1 PI-GLSTM
    # --------------------------------------------------------
    if "Objective 1 PI-GLSTM" in models:
        try:
            item = models["Objective 1 PI-GLSTM"]
            x = scale_input(item["x_scaler"], raw_window)

            delta, rul_scaled = item["model"](x)

            delta_value = float(
                delta[0].detach().cpu().item()
            )

            # Apply the non-negative degradation increment
            # progressively to the previous SoH estimate.
            next_obj1_soh = max(
                0.0,
                previous_obj1_soh - delta_value,
            )

            rul = inverse_rul(
                item["rul_scaler"],
                float(rul_scaled[0].detach().cpu().item()),
            )

            row[SOH_COLUMNS["Objective 1 PI-GLSTM"]] = float(
                np.clip(next_obj1_soh, 0.0, 1.05)
            )
            row[RUL_COLUMNS["Objective 1 PI-GLSTM"]] = max(
                0.0, rul
            )

        except Exception as exc:
            row[SOH_COLUMNS["Objective 1 PI-GLSTM"]] = np.nan
            row[RUL_COLUMNS["Objective 1 PI-GLSTM"]] = np.nan
            row["Objective 1 PI-GLSTM error"] = str(exc)

    # --------------------------------------------------------
    # Objective 2 Transfer PI-GLSTM
    # --------------------------------------------------------
    if "Objective 2 Transfer PI-GLSTM" in models:
        try:
            item = models["Objective 2 Transfer PI-GLSTM"]
            x = scale_input(item["x_scaler"], raw_window)

            model_initial_capacity = item.get(
                "initial_capacity"
            )

            if (
                model_initial_capacity is None
                or not np.isfinite(model_initial_capacity)
                or model_initial_capacity <= 0
            ):
                model_initial_capacity = dataset_initial_capacity

            # Use the last observed capacity in the window as
            # the current health anchor.
            anchor_capacity = float(
                raw_window["capacity"].iloc[-1]
            )

            anchor_soh = np.clip(
                anchor_capacity / model_initial_capacity,
                0.0,
                1.0,
            )

            anchor_tensor = torch.tensor(
                [anchor_soh],
                dtype=torch.float32,
                device=DEVICE,
            )

            soh, rul_scaled = item["model"](
                x,
                anchor_tensor,
            )

            soh_value = float(
                soh[0].detach().cpu().item()
            )

            rul = inverse_rul(
                item["rul_scaler"],
                float(rul_scaled[0].detach().cpu().item()),
            )

            row[SOH_COLUMNS["Objective 2 Transfer PI-GLSTM"]] = float(
                np.clip(soh_value, 0.0, 1.05)
            )
            row[RUL_COLUMNS["Objective 2 Transfer PI-GLSTM"]] = max(
                0.0, rul
            )

        except Exception as exc:
            row[SOH_COLUMNS["Objective 2 Transfer PI-GLSTM"]] = np.nan
            row[RUL_COLUMNS["Objective 2 Transfer PI-GLSTM"]] = np.nan
            row["Objective 2 Transfer PI-GLSTM error"] = str(exc)

    return row, next_obj1_soh


# ============================================================
# STREAMING GRAPH RENDERING
# ============================================================

def build_soh_figure(
    history: pd.DataFrame,
    title: str,
    show_reference: bool,
):
    fig = go.Figure()

    if history.empty:
        fig.update_layout(
            title=title,
            xaxis_title="Battery Cycle",
            yaxis_title="State of Health",
            template="plotly_white",
            height=440,
        )
        return fig

    if show_reference:
        fig.add_trace(
            go.Scatter(
                x=history["cycle"],
                y=history["soh_reference"] * 100,
                mode="lines",
                name="Reference SoH",
                line=dict(
                    color="#64748B",
                    width=2,
                    dash="dash",
                ),
                connectgaps=False,
            )
        )

    for name, column in SOH_COLUMNS.items():
        if column not in history.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=history["cycle"],
                y=history[column] * 100,
                mode="lines+markers",
                name=name,
                line=dict(
                    color=MODEL_COLORS[name],
                    width=2.5,
                ),
                marker=dict(size=4),
                connectgaps=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Battery Cycle",
        yaxis_title="SoH (%)",
        yaxis=dict(
            rangemode="tozero",
            range=[0, 105],
        ),
        hovermode="x unified",
        template="plotly_white",
        height=440,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=20, r=20, t=80, b=30),
    )

    return fig


def build_rul_figure(
    history: pd.DataFrame,
    title: str,
    show_reference: bool,
):
    fig = go.Figure()

    if history.empty:
        fig.update_layout(
            title=title,
            xaxis_title="Battery Cycle",
            yaxis_title="Remaining Useful Life (cycles)",
            template="plotly_white",
            height=440,
        )
        return fig

    if show_reference:
        fig.add_trace(
            go.Scatter(
                x=history["cycle"],
                y=history["rul_reference"],
                mode="lines",
                name="Reference RUL",
                line=dict(
                    color="#64748B",
                    width=2,
                    dash="dash",
                ),
                connectgaps=False,
            )
        )

    for name, column in RUL_COLUMNS.items():
        if column not in history.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=history["cycle"],
                y=history[column],
                mode="lines+markers",
                name=name,
                line=dict(
                    color=MODEL_COLORS[name],
                    width=2.5,
                ),
                marker=dict(size=4),
                connectgaps=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Battery Cycle",
        yaxis_title="RUL (cycles)",
        yaxis=dict(rangemode="tozero"),
        hovermode="x unified",
        template="plotly_white",
        height=440,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=20, r=20, t=80, b=30),
    )

    return fig


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "dataset_hash": None,
    "stream_running": False,
    "stream_index": 0,
    "stream_history": [],
    "stream_tick": 0,
    "obj1_previous_soh": None,
    "last_error": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_stream(initial_soh: float | None = None):
    st.session_state.stream_running = False
    st.session_state.stream_index = 0
    st.session_state.stream_history = []
    st.session_state.stream_tick = 0
    st.session_state.obj1_previous_soh = initial_soh
    st.session_state.last_error = None


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🔋 Battery Prognostics")
st.caption(
    "Progressive SoH and RUL inference across Baseline LSTM, "
    "Objective 1 PI-GLSTM, and Objective 2 Transfer PI-GLSTM."
)

with st.sidebar:
    st.header("Simulation settings")

    uploaded_file = st.file_uploader(
        "Upload battery-cycle CSV",
        type=["csv"],
        help=(
            "Required columns: cycle, voltage, temperature, "
            "capacity. SoH and RUL reference columns are optional."
        ),
    )

    st.divider()

    playback_label = st.selectbox(
        "Playback speed",
        options=[
            "Slow — 1.0 second/cycle",
            "Medium — 0.3 seconds/cycle",
            "Fast — 0.05 seconds/cycle",
        ],
        index=1,
    )

    playback_intervals = {
        "Slow — 1.0 second/cycle": 1.0,
        "Medium — 0.3 seconds/cycle": 0.3,
        "Fast — 0.05 seconds/cycle": 0.05,
    }

    playback_seconds = playback_intervals[playback_label]

    show_reference = st.checkbox(
        "Show reference curves",
        value=True,
        help=(
            "If the CSV lacks measured targets, the app labels "
            "capacity-derived SoH and cycle-derived RUL as proxies."
        ),
    )

    st.divider()

    st.caption(f"Inference device: `{DEVICE}`")
    st.caption(f"Sequence length: `{SEQ_LEN}` cycles")


# Load models once.
models, model_errors = load_models()

with st.expander("Model availability", expanded=bool(model_errors)):
    for name in MODEL_FILES:
        if name in models:
            st.success(f"{name}: loaded")
        else:
            st.error(
                f"{name}: unavailable — "
                f"{model_errors.get(name, 'Unknown loading error')}"
            )

if not models:
    st.error(
        "No model could be loaded. Check the checkpoint and scaler "
        "paths under the model_artifacts directory."
    )
    st.stop()

if uploaded_file is None:
    st.info(
        "Upload a battery-cycle CSV to begin streaming model predictions."
    )

    st.markdown(
        """
        **Expected input features**

        - `cycle`
        - `voltage`
        - `temperature`
        - `capacity`

        Optional reference columns: `soh` and `rul`.
        """
    )

    st.stop()


# ------------------------------------------------------------
# Read and prepare the uploaded dataset
# ------------------------------------------------------------

file_bytes = uploaded_file.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()

try:
    raw_df = pd.read_csv(uploaded_file)

    (
        battery_df,
        dataset_initial_capacity,
        has_measured_soh,
        has_measured_rul,
    ) = prepare_battery_data(raw_df)

except Exception as exc:
    st.error(f"Unable to prepare dataset: {exc}")
    st.stop()


# Reset stream whenever the uploaded dataset changes.
if st.session_state.dataset_hash != file_hash:
    initial_soh = float(
        battery_df["capacity"].iloc[SEQ_LEN - 1]
        / dataset_initial_capacity
    )

    reset_stream(initial_soh)
    st.session_state.dataset_hash = file_hash


total_steps = len(battery_df) - SEQ_LEN + 1

reference_description = []

if show_reference:
    if has_measured_soh:
        reference_description.append("SoH: supplied reference")
    else:
        reference_description.append("SoH: capacity-derived proxy")

    if has_measured_rul:
        reference_description.append("RUL: supplied reference")
    else:
        reference_description.append("RUL: cycle-derived proxy")

st.caption(
    f"Dataset: {len(battery_df):,} valid cycles · "
    f"{total_steps:,} prediction windows · "
    f"{' | '.join(reference_description) if reference_description else 'Reference curves hidden'}"
)


# ------------------------------------------------------------
# Streaming controls
# ------------------------------------------------------------

control_start, control_pause, control_reset, control_spacer = st.columns(
    [1.1, 1.2, 0.9, 3]
)

with control_start:
    if st.button(
        "▶ Start" if not st.session_state.stream_running else "▶ Running",
        disabled=(
            st.session_state.stream_running
            or st.session_state.stream_index >= total_steps
        ),
        use_container_width=True,
    ):
        st.session_state.stream_running = True
        st.rerun()

with control_pause:
    if st.button(
        "⏸ Pause" if st.session_state.stream_running else "▶ Resume",
        disabled=(
            st.session_state.stream_index >= total_steps
        ),
        use_container_width=True,
    ):
        st.session_state.stream_running = (
            not st.session_state.stream_running
        )
        st.rerun()

with control_reset:
    if st.button("↺ Reset", use_container_width=True):
        initial_soh = float(
            battery_df["capacity"].iloc[SEQ_LEN - 1]
            / dataset_initial_capacity
        )
        reset_stream(initial_soh)
        st.rerun()


# ------------------------------------------------------------
# Progress and live prediction indicators
# ------------------------------------------------------------

current_step = st.session_state.stream_index
progress = min(current_step / total_steps, 1.0)

st.progress(progress, text=f"Streaming progress: {progress:.1%}")

history_df = pd.DataFrame(
    st.session_state.stream_history
)

metric_cols = st.columns(5)

latest = (
    history_df.iloc[-1].to_dict()
    if not history_df.empty
    else {}
)

with metric_cols[0]:
    if latest:
        st.metric(
            "Current cycle",
            f"{latest['cycle']:.0f}",
        )
    else:
        st.metric("Current cycle", "Waiting")

with metric_cols[1]:
    value = latest.get(
        SOH_COLUMNS["Baseline LSTM"],
        np.nan,
    )
    st.metric(
        "Baseline SoH",
        f"{value * 100:.2f}%" if pd.notna(value) else "—",
    )

with metric_cols[2]:
    value = latest.get(
        SOH_COLUMNS["Objective 1 PI-GLSTM"],
        np.nan,
    )
    st.metric(
        "Objective 1 SoH",
        f"{value * 100:.2f}%" if pd.notna(value) else "—",
    )

with metric_cols[3]:
    value = latest.get(
        SOH_COLUMNS["Objective 2 Transfer PI-GLSTM"],
        np.nan,
    )
    st.metric(
        "Objective 2 SoH",
        f"{value * 100:.2f}%" if pd.notna(value) else "—",
    )

with metric_cols[4]:
    value = latest.get(
        RUL_COLUMNS["Objective 2 Transfer PI-GLSTM"],
        np.nan,
    )
    st.metric(
        "Objective 2 RUL",
        f"{value:.1f} cycles" if pd.notna(value) else "—",
    )


# ------------------------------------------------------------
# Live graph fragment
#
# The fragment refreshes every 0.05 seconds. A tick counter
# throttles the actual inference to the selected playback speed.
# ------------------------------------------------------------

@st.fragment(run_every=0.05)
def streaming_dashboard():
    if (
        st.session_state.stream_running
        and st.session_state.stream_index < total_steps
    ):
        st.session_state.stream_tick += 1

        required_ticks = max(
            1,
            round(playback_seconds / 0.05),
        )

        if st.session_state.stream_tick >= required_ticks:
            st.session_state.stream_tick = 0

            step_index = st.session_state.stream_index
            previous_soh = st.session_state.obj1_previous_soh

            if previous_soh is None:
                previous_soh = float(
                    battery_df["capacity"].iloc[SEQ_LEN - 1]
                    / dataset_initial_capacity
                )

            try:
                prediction, next_obj1_soh = predict_one_cycle(
                    models=models,
                    df=battery_df,
                    step_index=step_index,
                    previous_obj1_soh=previous_soh,
                    dataset_initial_capacity=dataset_initial_capacity,
                )

                st.session_state.stream_history.append(
                    prediction
                )

                st.session_state.obj1_previous_soh = (
                    next_obj1_soh
                )

                st.session_state.stream_index += 1
                st.session_state.last_error = None

            except Exception as exc:
                st.session_state.last_error = str(exc)
                st.session_state.stream_running = False

    current_history = pd.DataFrame(
        st.session_state.stream_history
    )

    if st.session_state.last_error:
        st.error(
            "Streaming stopped because of an inference error: "
            + st.session_state.last_error
        )

    if (
        st.session_state.stream_index >= total_steps
        and total_steps > 0
    ):
        st.success(
            "Streaming complete. Reset the simulation to replay "
            "the battery-cycle sequence."
        )
    elif st.session_state.stream_running:
        st.info(
            f"Streaming cycle "
            f"{st.session_state.stream_index + SEQ_LEN - 1:,} "
            f"of {len(battery_df):,}."
        )
    elif st.session_state.stream_index == 0:
        st.caption("Ready. Press Start to begin the simulation.")
    else:
        st.caption("Streaming paused. Resume to continue.")

    soh_fig = build_soh_figure(
        current_history,
        "Streaming State of Health (SoH)",
        show_reference,
    )

    rul_fig = build_rul_figure(
        current_history,
        "Streaming Remaining Useful Life (RUL)",
        show_reference,
    )

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        st.plotly_chart(
            soh_fig,
            use_container_width=True,
            key="streaming_soh_chart",
        )

    with graph_col2:
        st.plotly_chart(
            rul_fig,
            use_container_width=True,
            key="streaming_rul_chart",
        )


streaming_dashboard()


# ------------------------------------------------------------
# Optional export of the points revealed during the simulation
# ------------------------------------------------------------

if not history_df.empty:
    st.divider()

    export_col, status_col = st.columns([1, 3])

    with export_col:
        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download streamed predictions",
            data=csv_data,
            file_name="battery_streaming_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with status_col:
        st.caption(
            "The export contains only the prediction points that "
            "have been streamed so far. It does not expose future "
            "predictions before they are revealed."
        )
