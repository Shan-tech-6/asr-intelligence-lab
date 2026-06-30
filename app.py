"""
====================================================================================
 ASR INTELLIGENCE LAB  (app_v2.py)
====================================================================================
A production-ready Streamlit application for benchmarking multiple Automatic Speech
Recognition (ASR) engines side-by-side, across ALL languages:

    1. OpenAI Whisper            (local inference, 90+ languages)
    2. Faster-Whisper            (CTranslate2 optimized local inference, 90+ languages)
    3. Sarvam AI                 (cloud API - Indic languages, skipped for audio > 30s)
    4. Shrutam                   (cloud API - multilingual, safe mode if unavailable)
    5. Gemini                    (Google multimodal LLM - 100+ languages, cloud API)

Features
--------
- Upload a WAV file OR record live audio from the microphone
- Pick a target/spoken language (used as a hint for engines that support it)
- Run all available engines and measure:
    * Latency (seconds)
    * RTF (Real-Time Factor = processing_time / audio_duration)
    * WER  (Word Error Rate)      -- requires a reference transcript
    * CER  (Character Error Rate) -- requires a reference transcript
    * Accuracy (1 - WER, clipped to [0,1])
- Automatically detect the "Best Model" (highest accuracy) and "Fastest Model"
  (lowest latency)
- Professional dashboard with comparison bar charts (Latency / Accuracy / WER / CER)
- Side-by-side transcript comparison
- Download transcripts (.txt) for each engine
- Automatic logging of every run to an Excel workbook (asr_lab_logs.xlsx)
- Defensive error handling everywhere so a single failing engine never crashes the app

Run with:
    streamlit run app_v2.py
====================================================================================
"""

import os
import io
import time
import wave
import json
import base64
import tempfile
import traceback
import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import ctranslate2
from streamlit_mic_recorder import mic_recorder
# ------------------------------------------------------------------------------------
# OPTIONAL / SOFT DEPENDENCIES
# ------------------------------------------------------------------------------------
# Every external/heavy dependency is imported defensively so that the app degrades
# gracefully (instead of crashing) if a package or API is not installed/configured.
# ------------------------------------------------------------------------------------

# --- Whisper (local) ----------------------------------------------------------------
try:
    import whisper as openai_whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# --- Faster-Whisper (local, CTranslate2) ---------------------------------------------
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    FASTER_WHISPER_AVAILABLE = False

# --- HTTP client, shared by all cloud engines (Sarvam, Shrutam) ----------------------
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# --- Gemini (Google Generative AI, multimodal) ----------------------------------------
try:
    import google.generativeai as genai
    GEMINI_SDK_AVAILABLE = True
except Exception:
    GEMINI_SDK_AVAILABLE = False

# --- Audio I/O for live microphone recording --------------------------------------------
try:
    import sounddevice as sd
    MIC_AVAILABLE = True
except Exception:
    MIC_AVAILABLE = False

# --- WER / CER metrics -----------------------------------------------------------------
try:
    import jiwer
    JIWER_AVAILABLE = True
except Exception:
    JIWER_AVAILABLE = False

# --- Plotting ----------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


# ======================================================================================
# CONSTANTS
# ======================================================================================
LOG_FILE = "asr_lab_logs.xlsx"
SARVAM_MAX_DURATION_SEC = 30.0   # Sarvam AI is skipped beyond this duration
SAMPLE_RATE = 16000              # standard ASR sample rate
DEFAULT_RECORD_SECONDS = 5

SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"
SHRUTAM_API_URL = "https://api.shrutam.ai/v1/transcribe"  # placeholder endpoint
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# Languages offered in the UI. "Auto Detect" lets each engine infer the language itself.
# (code, label) -- codes follow ISO 639-1 / BCP-47 where applicable.
LANGUAGE_OPTIONS = [
    ("auto", "Auto Detect"),
    ("en", "English"),
    ("hi", "Hindi"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("kn", "Kannada"),
    ("ml", "Malayalam"),
    ("mr", "Marathi"),
    ("bn", "Bengali"),
    ("gu", "Gujarati"),
    ("pa", "Punjabi"),
    ("ur", "Urdu"),
    ("or", "Odia"),
    ("as", "Assamese"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("zh", "Chinese (Mandarin)"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("ar", "Arabic"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
]
LANGUAGE_CODE_TO_LABEL = dict(LANGUAGE_OPTIONS)


# ======================================================================================
# DATA STRUCTURES
# ======================================================================================
@dataclass
class EngineResult:
    """Container for a single ASR engine's run result."""
    engine: str
    transcript: str = ""
    detected_language: str = ""
    latency_sec: float = 0.0
    rtf: float = 0.0
    wer: Optional[float] = None
    cer: Optional[float] = None
    accuracy: Optional[float] = None
    status: str = "pending"     # pending | success | skipped | error
    error_message: str = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


# ======================================================================================
# UTILITY / HELPER FUNCTIONS
# ======================================================================================

def safe_run(fn, *args, **kwargs):
    """Execute a function and capture any exception, returning (result, error_str)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def get_wav_duration_seconds(file_path: str) -> float:
    """Return the duration (in seconds) of a WAV file. Returns 0.0 on failure."""
    try:
        with wave.open(file_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception:
        return 0.0


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, Optional[float]]:
    """
    Compute WER, CER and Accuracy between a reference transcript and a hypothesis.
    Returns None values if jiwer is unavailable or reference text is empty.
    """
    if not reference or not reference.strip():
        return {"wer": None, "cer": None, "accuracy": None}

    if not JIWER_AVAILABLE:
        return {"wer": None, "cer": None, "accuracy": None}

    try:
        wer_score = jiwer.wer(reference, hypothesis)
        cer_score = jiwer.cer(reference, hypothesis)
        accuracy = max(0.0, min(1.0, 1.0 - wer_score))
        return {"wer": round(wer_score, 4), "cer": round(cer_score, 4), "accuracy": round(accuracy, 4)}
    except Exception:
        return {"wer": None, "cer": None, "accuracy": None}


def record_microphone_audio(seconds: int, sample_rate: int = SAMPLE_RATE) -> Optional[str]:
    """
    Record `seconds` of audio from the default microphone and save it to a temp WAV file.
    Returns the file path, or None on failure.
    """
    if not MIC_AVAILABLE:
        return None
    try:
        recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 -> 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())
        return tmp_path
    except Exception:
        return None


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Persist an uploaded Streamlit file object to a temp WAV file on disk."""
    try:
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return tmp_path
    except Exception:
        return None


# ======================================================================================
# ENGINE WRAPPERS
# Each function below: takes a wav file path (+ language hint), returns an EngineResult.
# All exceptions are caught internally so one failing engine never crashes the app.
# ======================================================================================

@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = "base"):
    """Load (and cache) the local OpenAI Whisper model."""
    return openai_whisper.load_model(model_size)


def run_whisper(file_path: str, model_size: str, language_code: str) -> EngineResult:
    result = EngineResult(engine="Whisper")
    if not WHISPER_AVAILABLE:
        result.status = "error"
        result.error_message = "openai-whisper package not installed."
        return result
    try:
        model = load_whisper_model(model_size)
        kwargs = {}
        if language_code != "auto":
            kwargs["language"] = language_code

        start = time.perf_counter()
        output = model.transcribe(file_path, **kwargs)
        latency = time.perf_counter() - start

        result.transcript = output.get("text", "").strip()
        result.detected_language = output.get("language", language_code)
        result.latency_sec = round(latency, 3)
        result.status = "success"
    except Exception as exc:
        result.status = "error"
        result.error_message = f"{type(exc).__name__}: {exc}"
    return result


@st.cache_resource(show_spinner=False)
def load_faster_whisper_model(model_size: str = "base"):
    """Load (and cache) the Faster-Whisper model (CPU, int8 for speed/portability)."""
    return FasterWhisperModel(model_size, device="cpu", compute_type="int8")


def run_faster_whisper(file_path: str, model_size: str, language_code: str) -> EngineResult:
    result = EngineResult(engine="Faster-Whisper")
    if not FASTER_WHISPER_AVAILABLE:
        result.status = "error"
        result.error_message = "faster-whisper package not installed."
        return result
    try:
        model = load_faster_whisper_model(model_size)
        kwargs = {}
        if language_code != "auto":
            kwargs["language"] = language_code

        start = time.perf_counter()
        segments, info = model.transcribe(file_path, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments)
        latency = time.perf_counter() - start

        result.transcript = text.strip()
        result.detected_language = getattr(info, "language", language_code) or language_code
        result.latency_sec = round(latency, 3)
        result.status = "success"
    except Exception as exc:
        result.status = "error"
        result.error_message = f"{type(exc).__name__}: {exc}"
    return result


def run_sarvam(file_path: str, duration_sec: float, api_key: str, language_code: str) -> EngineResult:
    """
    Call the Sarvam AI speech-to-text API (strong for Indic languages).
    Per requirement: audio longer than SARVAM_MAX_DURATION_SEC is skipped entirely.
    """
    result = EngineResult(engine="Sarvam AI")

    if duration_sec > SARVAM_MAX_DURATION_SEC:
        result.status = "skipped"
        result.error_message = (
            f"Audio duration {duration_sec:.1f}s exceeds Sarvam AI's "
            f"{SARVAM_MAX_DURATION_SEC:.0f}s limit."
        )
        return result

    if not REQUESTS_AVAILABLE:
        result.status = "error"
        result.error_message = "requests package not available for Sarvam AI call."
        return result

    if not api_key:
        result.status = "error"
        result.error_message = "Sarvam AI API key not provided."
        return result

    # Sarvam uses BCP-47 style codes with a region suffix, e.g. "hi-IN". Fall back to
    # "unknown" (auto-detect) when the user picked Auto Detect or an unsupported code.
    sarvam_lang_map = {
        "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN", "ml": "ml-IN",
        "mr": "mr-IN", "bn": "bn-IN", "gu": "gu-IN", "pa": "pa-IN", "or": "od-IN",
        "en": "en-IN",
    }
    sarvam_lang = sarvam_lang_map.get(language_code, "unknown")

    try:
        start = time.perf_counter()
        with open(file_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            headers = {"api-subscription-key": api_key}
            data = {"model": "saarika:v2.5", "language_code": sarvam_lang}
            response = requests.post(
                SARVAM_API_URL, headers=headers, files=files, data=data, timeout=60
            )
        latency = time.perf_counter() - start

        if response.status_code == 200:
            payload = response.json()
            result.transcript = payload.get("transcript", "").strip()
            result.detected_language = payload.get("language_code", sarvam_lang)
            result.latency_sec = round(latency, 3)
            result.status = "success"
        else:
            result.status = "error"
            result.error_message = f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as exc:
        result.status = "error"
        result.error_message = f"{type(exc).__name__}: {exc}"
    return result


def run_shrutam(file_path: str, api_key: str, language_code: str) -> EngineResult:
    """
    Call the Shrutam speech-to-text API (multilingual).
    Runs in "safe mode" (graceful skip, no crash) if the service or API key is unavailable.
    """
    result = EngineResult(engine="Shrutam")

    if not REQUESTS_AVAILABLE:
        result.status = "skipped"
        result.error_message = "Shrutam safe mode: requests package unavailable."
        return result

    if not api_key:
        result.status = "skipped"
        result.error_message = "Shrutam safe mode: no API key configured."
        return result

    try:
        start = time.perf_counter()
        with open(file_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {}
            if language_code != "auto":
                data["language"] = language_code
            response = requests.post(
                SHRUTAM_API_URL, headers=headers, files=files, data=data, timeout=60
            )
        latency = time.perf_counter() - start

        if response.status_code == 200:
            payload = response.json()
            result.transcript = payload.get("text", payload.get("transcript", "")).strip()
            result.detected_language = payload.get("language", language_code)
            result.latency_sec = round(latency, 3)
            result.status = "success"
        else:
            # Service responded but with an error -> degrade safely, don't crash.
            result.status = "skipped"
            result.error_message = f"Shrutam safe mode: HTTP {response.status_code}."
    except Exception as exc:
        # Network/timeout/DNS errors -> degrade safely.
        result.status = "skipped"
        result.error_message = f"Shrutam safe mode: {type(exc).__name__}: {exc}"
    return result


def run_gemini(file_path: str, api_key: str, language_code: str) -> EngineResult:
    """
    Call Google Gemini (multimodal) to transcribe audio.
    Gemini natively understands 100+ languages and needs no language hint to work,
    but we pass one through the prompt to bias accuracy when known.
    """
    result = EngineResult(engine="Gemini")

    if not GEMINI_SDK_AVAILABLE:
        result.status = "error"
        result.error_message = "google-generativeai package not installed (`pip install google-generativeai`)."
        return result

    if not api_key:
        result.status = "skipped"
        result.error_message = "Gemini safe mode: no API key configured."
        return result

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        lang_label = LANGUAGE_CODE_TO_LABEL.get(language_code, "the spoken language")
        if language_code == "auto":
            prompt = (
                "Transcribe the spoken audio verbatim. Auto-detect the language. "
                "Return ONLY the transcript text, with no extra commentary, labels, or quotation marks."
            )
        else:
            prompt = (
                f"Transcribe the spoken audio verbatim. The audio is in {lang_label}. "
                "Return ONLY the transcript text, with no extra commentary, labels, or quotation marks."
            )

        start = time.perf_counter()
        uploaded = genai.upload_file(path=file_path, mime_type="audio/wav")
        response = model.generate_content([prompt, uploaded])
        latency = time.perf_counter() - start

        text = (getattr(response, "text", "") or "").strip()
        result.transcript = text
        result.detected_language = language_code if language_code != "auto" else "auto"
        result.latency_sec = round(latency, 3)
        result.status = "success" if text else "error"
        if not text:
            result.error_message = "Gemini returned an empty transcript."
    except Exception as exc:
        # Quota errors, invalid key, network issues etc. -> degrade safely, don't crash.
        result.status = "skipped"
        result.error_message = f"Gemini safe mode: {type(exc).__name__}: {exc}"
    return result


# ======================================================================================
# EXCEL LOGGING
# ======================================================================================

def log_run_to_excel(run_records: List[Dict[str, Any]], log_path: str = LOG_FILE):
    """
    Append the results of a benchmarking run to an Excel workbook.
    Creates the workbook if it does not already exist.
    """
    try:
        new_df = pd.DataFrame(run_records)
        if os.path.exists(log_path):
            existing_df = pd.read_excel(log_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        combined_df.to_excel(log_path, index=False)
        return True, os.path.abspath(log_path)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ======================================================================================
# DASHBOARD / CHART HELPERS
# ======================================================================================

ENGINE_COLORS = {
    "Whisper": "#4F46E5",
    "Faster-Whisper": "#06B6D4",
    "Sarvam AI": "#F59E0B",
    "Shrutam": "#EC4899",
    "Gemini": "#10B981",
}


def make_bar_chart(df: pd.DataFrame, metric_col: str, title: str, y_label: str):
    """Build a Plotly bar chart for a given metric across engines, colored per-engine."""
    if not PLOTLY_AVAILABLE:
        return None
    plot_df = df.dropna(subset=[metric_col])
    if plot_df.empty:
        return None
    colors = [ENGINE_COLORS.get(e, "#6B7280") for e in plot_df["engine"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["engine"],
                y=plot_df[metric_col],
                marker_color=colors,
                text=plot_df[metric_col],
                texttemplate="%{text}",
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        xaxis_title="Engine",
        template="plotly_white",
        height=380,
        margin=dict(t=60, b=40),
        showlegend=False,
    )
    return fig


# ======================================================================================
# STREAMLIT APP
# ======================================================================================

def configure_page():
    st.set_page_config(
        page_title="ASR Intelligence Lab",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #4F46E5, #06B6D4, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .sub-header {
            color: #6B7280;
            font-size: 1.0rem;
            margin-top: -8px;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .engine-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .status-success {color:#059669; font-weight:700;}
        .status-skipped {color:#D97706; font-weight:700;}
        .status-error {color:#DC2626; font-weight:700;}
        section[data-testid="stSidebar"] {
            border-right: 1px solid #E5E7EB;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">🎙️ ASR Intelligence Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Benchmark Whisper, Faster-Whisper, Sarvam AI, Shrutam & Gemini '
        'side-by-side across any language — latency, WER, CER, RTF & accuracy, in one dashboard.</div>',
        unsafe_allow_html=True,
    )


def sidebar_controls():
    """Render sidebar inputs and return a settings dict."""
    st.sidebar.header("⚙️ Configuration")

    audio_source = st.sidebar.radio("Audio Source", ["Upload WAV", "Record from Microphone"])

    uploaded_file = None
    record_seconds = DEFAULT_RECORD_SECONDS
    if audio_source == "Upload WAV":
        uploaded_file = st.sidebar.file_uploader("Upload a WAV file", type=["wav"])
    else:
        if not MIC_AVAILABLE:
            st.sidebar.warning("`sounddevice` not installed — microphone recording disabled.")
        record_seconds = st.sidebar.slider("Recording duration (sec)", 2, 30, DEFAULT_RECORD_SECONDS)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 Language")
    language_label = st.sidebar.selectbox(
        "Spoken language (used as a hint where supported)",
        [label for _, label in LANGUAGE_OPTIONS],
        index=0,
        help="Gemini, Whisper & Faster-Whisper can auto-detect language; Sarvam AI/Shrutam "
             "use this as an explicit hint for best accuracy.",
    )
    language_code = next(code for code, label in LANGUAGE_OPTIONS if label == language_label)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Local Models")
    model_size = st.sidebar.selectbox(
        "Whisper / Faster-Whisper model size",
        ["tiny", "base", "small", "medium"],
        index=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("☁️ Cloud API Keys")
    sarvam_key = st.sidebar.text_input(
        "Sarvam AI API Key", type="password",
        value=os.environ.get("SARVAM_API_KEY", ""),
    )
    shrutam_key = st.sidebar.text_input(
        "Shrutam API Key", type="password",
        value=os.environ.get("SHRUTAM_API_KEY", ""),
        help="If left blank or the service is unreachable, Shrutam runs in safe mode (auto-skipped).",
    )
    gemini_key = st.sidebar.text_input(
        "Gemini API Key", type="password",
        value=os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")),
        help="If left blank or the call fails (e.g. quota), Gemini runs in safe mode (auto-skipped).",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("✅ Engines to Run")
    engines_enabled = {
        "Whisper": st.sidebar.checkbox("Whisper", value=True),
        "Faster-Whisper": st.sidebar.checkbox("Faster-Whisper", value=True),
        "Sarvam AI": st.sidebar.checkbox("Sarvam AI", value=True),
        "Shrutam": st.sidebar.checkbox("Shrutam", value=True),
        "Gemini": st.sidebar.checkbox("Gemini", value=True),
    }

    st.sidebar.markdown("---")
    reference_text = st.sidebar.text_area(
        "Reference Transcript (optional, enables WER/CER/Accuracy)", height=110
    )

    return {
        "audio_source": audio_source,
        "uploaded_file": uploaded_file,
        "record_seconds": record_seconds,
        "model_size": model_size,
        "language_code": language_code,
        "language_label": language_label,
        "sarvam_key": sarvam_key.strip(),
        "shrutam_key": shrutam_key.strip(),
        "gemini_key": gemini_key.strip(),
        "engines_enabled": engines_enabled,
        "reference_text": reference_text.strip(),
    }


def render_engine_status_badges():
    """Show availability of each engine/dependency at the top of the app."""
    cols = st.columns(7)
    badges = [
        ("Whisper", WHISPER_AVAILABLE),
        ("Faster-Whisper", FASTER_WHISPER_AVAILABLE),
        ("Sarvam AI", REQUESTS_AVAILABLE),
        ("Shrutam", REQUESTS_AVAILABLE),
        ("Gemini", GEMINI_SDK_AVAILABLE),
        ("Microphone", MIC_AVAILABLE),
        ("WER/CER (jiwer)", JIWER_AVAILABLE),
    ]
    for col, (name, ok) in zip(cols, badges):
        with col:
            icon = "🟢" if ok else "🔴"
            st.markdown(f"**{icon} {name}**")


def run_all_engines(file_path: str, duration_sec: float, settings: Dict[str, Any]) -> List[EngineResult]:
    """Orchestrate running every enabled engine, with isolated error handling each."""
    results: List[EngineResult] = []
    enabled = settings["engines_enabled"]
    model_size = settings["model_size"]
    lang = settings["language_code"]

    steps = [name for name, on in enabled.items() if on]
    total = max(len(steps), 1)
    progress = st.progress(0.0, text="Starting benchmark run...")
    done = 0

    if enabled.get("Whisper"):
        progress.progress(done / total, text="Running Whisper...")
        results.append(run_whisper(file_path, model_size, lang))
        done += 1

    if enabled.get("Faster-Whisper"):
        progress.progress(done / total, text="Running Faster-Whisper...")
        results.append(run_faster_whisper(file_path, model_size, lang))
        done += 1

    if enabled.get("Sarvam AI"):
        progress.progress(done / total, text="Running Sarvam AI...")
        results.append(run_sarvam(file_path, duration_sec, settings["sarvam_key"], lang))
        done += 1

    if enabled.get("Shrutam"):
        progress.progress(done / total, text="Running Shrutam (safe mode if unavailable)...")
        results.append(run_shrutam(file_path, settings["shrutam_key"], lang))
        done += 1

    if enabled.get("Gemini"):
        progress.progress(done / total, text="Running Gemini (safe mode if unavailable)...")
        results.append(run_gemini(file_path, settings["gemini_key"], lang))
        done += 1

    progress.progress(1.0, text="Benchmark complete.")
    time.sleep(0.3)
    progress.empty()

    # Post-process: RTF + metrics
    for r in results:
        if r.status == "success" and duration_sec > 0:
            r.rtf = round(r.latency_sec / duration_sec, 4)
        if r.status == "success":
            metrics = compute_metrics(settings["reference_text"], r.transcript)
            r.wer = metrics["wer"]
            r.cer = metrics["cer"]
            r.accuracy = metrics["accuracy"]

    return results


def render_results_dashboard(results: List[EngineResult], duration_sec: float, settings: Dict[str, Any]):
    """Render the full results section: table, charts, transcripts, downloads, best/fastest."""

    df = pd.DataFrame([r.to_row() for r in results])

    st.markdown("## 📊 Benchmark Results")
    st.caption(
        f"Language: **{settings['language_label']}**  |  "
        f"Audio duration: **{duration_sec:.2f}s**  |  "
        f"Run time: **{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**"
    )

    # ---- Status table -------------------------------------------------------------
    display_df = df.copy()
    display_df["status"] = display_df["status"].map(
        {"success": "✅ Success", "skipped": "⏭️ Skipped", "error": "❌ Error", "pending": "⏳ Pending"}
    )
    st.dataframe(
        display_df[["engine", "status", "detected_language", "latency_sec", "rtf", "wer", "cer", "accuracy", "error_message"]]
        .rename(columns={
            "engine": "Engine", "status": "Status", "detected_language": "Language",
            "latency_sec": "Latency (s)", "rtf": "RTF", "wer": "WER", "cer": "CER",
            "accuracy": "Accuracy", "error_message": "Notes"
        }),
        use_container_width=True,
        hide_index=True,
    )

    successful = df[df["status"] == "success"].copy()

    # ---- Best / Fastest model callouts --------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        if not successful.empty and successful["accuracy"].notna().any():
            best_row = successful.loc[successful["accuracy"].idxmax()]
            st.markdown(
                f'<div class="metric-card">🏆 <b>Best Model (Accuracy)</b><br>'
                f'<span style="font-size:1.4rem;">{best_row["engine"]}</span><br>'
                f'Accuracy: {best_row["accuracy"]:.2%}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="metric-card">🏆 <b>Best Model</b><br>'
                'Provide a reference transcript to compute accuracy.</div>',
                unsafe_allow_html=True,
            )
    with col2:
        if not successful.empty:
            fastest_row = successful.loc[successful["latency_sec"].idxmin()]
            st.markdown(
                f'<div class="metric-card">⚡ <b>Fastest Model</b><br>'
                f'<span style="font-size:1.4rem;">{fastest_row["engine"]}</span><br>'
                f'Latency: {fastest_row["latency_sec"]:.2f}s</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="metric-card">⚡ <b>Fastest Model</b><br>No successful runs.</div>', unsafe_allow_html=True)
    with col3:
        n_success = (df["status"] == "success").sum()
        n_skipped = (df["status"] == "skipped").sum()
        n_error = (df["status"] == "error").sum()
        st.markdown(
            f'<div class="metric-card">📈 <b>Run Summary</b><br>'
            f'✅ {n_success} success &nbsp; ⏭️ {n_skipped} skipped &nbsp; ❌ {n_error} error</div>',
            unsafe_allow_html=True,
        )

    # ---- Charts ----------------------------------------------------------------------
    st.markdown("### 📉 Comparison Charts")
    if successful.empty:
        st.info("No successful engine runs to chart yet.")
    elif not PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed — charts unavailable. Run `pip install plotly`.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = make_bar_chart(successful, "latency_sec", "Latency by Engine", "Seconds")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = make_bar_chart(successful, "accuracy", "Accuracy by Engine", "Accuracy (0-1)")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Accuracy chart requires a reference transcript.")

        c3, c4 = st.columns(2)
        with c3:
            fig = make_bar_chart(successful, "wer", "WER by Engine", "Word Error Rate")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("WER chart requires a reference transcript.")
        with c4:
            fig = make_bar_chart(successful, "cer", "CER by Engine", "Character Error Rate")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("CER chart requires a reference transcript.")

    # ---- Transcript comparison + downloads --------------------------------------------
    st.markdown("### 📝 Transcript Comparison")
    n_cols = len(results) if results else 1
    cols = st.columns(n_cols)
    for col, r in zip(cols, results):
        with col:
            color = ENGINE_COLORS.get(r.engine, "#6B7280")
            st.markdown(
                f'<span class="engine-badge" style="background:{color}22; color:{color};">{r.engine}</span>',
                unsafe_allow_html=True,
            )
            status_class = {
                "success": "status-success", "skipped": "status-skipped", "error": "status-error"
            }.get(r.status, "")
            st.markdown(f'<span class="{status_class}">{r.status.upper()}</span>', unsafe_allow_html=True)
            if r.status == "success":
                st.text_area(f"{r.engine} transcript", r.transcript, height=160, key=f"ta_{r.engine}")
                st.download_button(
                    f"⬇️ Download",
                    data=r.transcript,
                    file_name=f"{r.engine.replace(' ', '_').lower()}_transcript.txt",
                    mime="text/plain",
                    key=f"dl_{r.engine}",
                )
            else:
                st.caption(r.error_message or "No transcript available.")

    return df


def main():
    configure_page()
    render_engine_status_badges()
    st.markdown("---")

    settings = sidebar_controls()

    # Session state holders
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
        st.session_state["last_duration"] = None

    # ---- Acquire audio file --------------------------------------------------------
    file_path = None
    duration_sec = 0.0

    if settings["audio_source"] == "Upload WAV":
        if settings["uploaded_file"] is not None:
            file_path = save_uploaded_file(settings["uploaded_file"])
            if file_path:
                duration_sec = get_wav_duration_seconds(file_path)
                st.audio(settings["uploaded_file"])
            else:
                st.error("Failed to save uploaded file.")
    else:
        st.info(f"Press the button below to record {settings['record_seconds']}s of audio from your microphone.")
        if st.button("🎤 Start Recording", disabled=not MIC_AVAILABLE):
            with st.spinner("Recording..."):
                file_path = record_microphone_audio(settings["record_seconds"])
            if file_path:
                duration_sec = get_wav_duration_seconds(file_path)
                st.session_state["recorded_path"] = file_path
                st.session_state["recorded_duration"] = duration_sec
                st.success("Recording complete!")
                st.audio(file_path)
            else:
                st.error("Microphone recording failed. Check that `sounddevice` is installed and a mic is connected.")
        elif st.session_state.get("recorded_path"):
            file_path = st.session_state["recorded_path"]
            duration_sec = st.session_state.get("recorded_duration", 0.0)
            st.audio(file_path)

    if duration_sec > SARVAM_MAX_DURATION_SEC and settings["engines_enabled"].get("Sarvam AI"):
        st.warning(
            f"⚠️ Audio is {duration_sec:.1f}s long — Sarvam AI will be **skipped** "
            f"(limit: {SARVAM_MAX_DURATION_SEC:.0f}s)."
        )

    st.markdown("---")
    run_clicked = st.button("🚀 Run Benchmark", type="primary", use_container_width=True, disabled=file_path is None)

    if run_clicked and file_path:
        if not any(settings["engines_enabled"].values()):
            st.error("Please enable at least one engine in the sidebar.")
        else:
            with st.spinner("Benchmarking ASR engines... this may take a moment."):
                results = run_all_engines(file_path, duration_sec, settings)
            st.session_state["last_results"] = results
            st.session_state["last_duration"] = duration_sec

            # ---- Auto-log to Excel -----------------------------------------------------
            log_rows = []
            timestamp = dt.datetime.now().isoformat(timespec="seconds")
            for r in results:
                row = r.to_row()
                row["timestamp"] = timestamp
                row["audio_duration_sec"] = round(duration_sec, 3)
                row["language"] = settings["language_label"]
                row["reference_text_provided"] = bool(settings["reference_text"])
                log_rows.append(row)
            ok, info = log_run_to_excel(log_rows)
            if ok:
                st.toast(f"📒 Run logged to {info}", icon="✅")
            else:
                st.warning(f"Could not write Excel log: {info}")

    # ---- Render last results (persists across reruns) -----------------------------------
    if st.session_state.get("last_results"):
        render_results_dashboard(
            st.session_state["last_results"],
            st.session_state["last_duration"] or 0.0,
            settings,
        )

        # Offer the full Excel log for download if it exists
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "rb") as f:
                st.download_button(
                    "⬇️ Download Full Excel Log",
                    data=f.read(),
                    file_name=LOG_FILE,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.caption(f"Log file path: `{os.path.abspath(LOG_FILE)}`")
    else:
        st.info("Upload or record audio, then click **Run Benchmark** to see results here.")

    st.markdown("---")
    st.caption(
        "ASR Intelligence Lab • Built with Streamlit • Whisper, Faster-Whisper, Sarvam AI, Shrutam & Gemini "
        "are benchmarked independently with isolated error handling so a single failure never breaks the run."
    )


# ======================================================================================
# ENTRYPOINT
# ================================a======================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as top_level_exc:
        st.error("An unexpected error occurred in the application.")
        st.exception(top_level_exc)
        st.code(traceback.format_exc())
