import streamlit as st
import os
import subprocess
import time
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf

# Optional: ffmpeg-python is convenient when installed, but we also support
# invoking local ffmpeg (e.g., macOS via `brew install ffmpeg`).
try:
    import ffmpeg  # type: ignore
except Exception:  # pragma: no cover
    ffmpeg = None


# -------------------------------
# Audio conversion helpers
# -------------------------------
SUPPORTED_AUDIO_EXTS = [
    "wav",
    "mp3",
    "m4a",
    "aac",
    "flac",
    "ogg",
    "opus",
    "webm",
    "mp4",
    "mov",
    "aiff",
    "aif",
    "caf",
    "wma",
]


def _which(cmd: str) -> str | None:
    """Return absolute path for cmd if it exists in PATH."""
    from shutil import which

    return which(cmd)


def _brew_prefix() -> str | None:
    """Return Homebrew prefix if available (macOS)."""
    try:
        res = subprocess.run(
            ["brew", "--prefix"],
            check=True,
            capture_output=True,
            text=True,
        )
        prefix = res.stdout.strip()
        return prefix or None
    except Exception:
        return None


def find_ffmpeg_binary() -> str:
    """Find an ffmpeg binary.

    Tries:
      1) ffmpeg in PATH
      2) Homebrew locations (macOS): $(brew --prefix)/bin/ffmpeg and common defaults

    Raises RuntimeError if not found.
    """
    ff = _which("ffmpeg")
    if ff:
        return ff

    prefix = _brew_prefix()
    if prefix:
        cand = os.path.join(prefix, "bin", "ffmpeg")
        if os.path.exists(cand):
            return cand

    # Common macOS/Homebrew defaults
    for cand in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]:
        if os.path.exists(cand):
            return cand

    raise RuntimeError(
        "ffmpeg was not found. Install it (macOS: `brew install ffmpeg`) or add it to PATH."
    )


def convert_to_wav(input_path: str, output_path: str, sr: int = 16000) -> None:
    """Convert arbitrary audio/video to mono WAV using ffmpeg.

    Uses ffmpeg-python if installed, else falls back to subprocess.
    """
    if ffmpeg is not None:
        (
            ffmpeg.input(input_path)
            .output(
                output_path,
                format="wav",
                acodec="pcm_s16le",
                ac=1,
                ar=sr,
                loglevel="error",
            )
            .overwrite_output()
            .run()
        )
        return

    ff_bin = find_ffmpeg_binary()
    cmd = [
        ff_bin,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-acodec",
        "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def load_audio_as_wav(uploaded_file, target_sr: int = 16000):
    """Load Streamlit UploadedFile by converting to WAV first, then reading via soundfile.

    Returns:
      y: np.ndarray float32 mono
      sr: int
    """
    suffix = Path(uploaded_file.name).suffix.lower() or ".dat"
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, f"input{suffix}")
        out_path = os.path.join(tmpdir, "converted.wav")
        with open(in_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        convert_to_wav(in_path, out_path, sr=target_sr)
        y, sr = sf.read(out_path, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y, sr


# -------------------------------
# Existing app logic below
# -------------------------------

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "intro"


def navigate(page_name: str):
    st.session_state.page = page_name


def page_intro():
    st.title("Voice Scope")
    st.write(
        "Welcome to Voice Scope. Upload audio files and analyze them through multiple result pages."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Result Page 1"):
            navigate("result1")
    with col2:
        if st.button("Go to Result Page 2"):
            navigate("result2")


def page_result1():
    st.header("Result Page 1")

    st.markdown(
        """
        <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; border:1px solid #e9ecef;">
            <strong>Legend:</strong><br>
            <span style="color:#FF4B4B;">■</span> Original Audio Waveform<br>
            <span style="color:#00CC96;">■</span> Processed Audio Waveform<br>
            <span style="color:#636EFA;">■</span> Feature/Metric Overlay (if applicable)
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=SUPPORTED_AUDIO_EXTS,
        key="uploader_result1",
    )

    if uploaded_file is not None:
        try:
            y, sr = load_audio_as_wav(uploaded_file, target_sr=16000)
        except Exception as e:
            st.error(f"Failed to load audio. {e}")
            return

        # Example: compute basic stats / waveform plot
        duration = len(y) / sr if sr else 0
        st.write(f"Sample rate: {sr} Hz")
        st.write(f"Duration: {duration:.2f} seconds")

        # Plot waveform
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y, mode="lines", name="Waveform"))
        fig.update_layout(title="Audio Waveform", xaxis_title="Samples", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)


def page_result2():
    st.header("Result Page 2")

    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=SUPPORTED_AUDIO_EXTS,
        key="uploader_result2",
    )

    if uploaded_file is not None:
        try:
            y, sr = load_audio_as_wav(uploaded_file, target_sr=16000)
        except Exception as e:
            st.error(f"Failed to load audio. {e}")
            return

        # Example feature extraction
        st.write(f"Loaded audio with {len(y)} samples at {sr} Hz")
        rms = librosa.feature.rms(y=y)[0]
        df = pd.DataFrame({"rms": rms})
        st.line_chart(df)


# Routing
if st.session_state.page == "intro":
    page_intro()
elif st.session_state.page == "result1":
    if st.button("Back"):
        navigate("intro")
    page_result1()
elif st.session_state.page == "result2":
    if st.button("Back"):
        navigate("intro")
    page_result2()
