import gradio as gr
from TTS.api import TTS
import soundfile as sf
import librosa
import os
import json
import tempfile
import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from openvoice_wrapper import synthesize_openvoice

# Allow loading checkpoints that include config objects
add_safe_globals([XttsConfig, XttsAudioConfig, VitsConfig])

# Default model and additional high quality TTS models
DEFAULT_MODEL = "tts_models/en/vctk/vits"
AVAILABLE_MODELS = [
    ("VITS VCTK (English)", "tts_models/en/vctk/vits"),
    ("Jenny (English)", "tts_models/en/jenny/jenny"),
    ("XTTS v2 (Multilingual)", "tts_models/multilingual/multi-dataset/xtts_v2"),
]
MODEL_NAME_MAP = dict(AVAILABLE_MODELS)
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

PROFILES_FILE = "profiles.json"
if not os.path.exists(PROFILES_FILE):
    with open(PROFILES_FILE, "w") as f:
        json.dump({}, f)

# Current TTS model instance
tts = TTS(model_name=DEFAULT_MODEL, progress_bar=False)
tts.to("cuda")
current_model = DEFAULT_MODEL

def load_tts_model(model_label):
    """Load selected TTS model and update speaker list."""
    global tts, current_model, speakers
    model_name = MODEL_NAME_MAP.get(model_label, model_label)
    if model_name == current_model:
        return gr.update()
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        import torch.serialization
        torch.serialization.add_safe_globals([XttsConfig])
    except Exception:
        pass
    tts = TTS(model_name=model_name, progress_bar=False)
    tts.to("cuda")
    current_model = model_name
    speakers = (
        tts.speakers if tts.is_multi_speaker else ["default"]
    ) + list(openvoice_speakers.keys())
    return gr.update(choices=speakers, value=speakers[0])

def on_model_change(model_label):
    """Handle model dropdown change by mapping label to model name."""
    model_name = MODEL_NAME_MAP.get(model_label, DEFAULT_MODEL)
    return load_tts_model(model_name)

with open(PROFILES_FILE, "r") as f:
    profiles = json.load(f)

openvoice_speakers = {
    "Clone: Laura": "clones/laura.wav",
    "Clone: Mark": "clones/mark.wav",
}

speakers = (tts.speakers if tts.is_multi_speaker else ["default"]) + list(openvoice_speakers.keys())

def synthesize(text, model_label, speaker_id, emotion, pitch, speed):
    # Load the requested model if different
    load_tts_model(model_label)
    if speaker_id in openvoice_speakers:
        speaker_wav = openvoice_speakers[speaker_id]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile_path = tmpfile.name
        synthesize_openvoice(text, speaker_wav, tmpfile_path)
        history_path = os.path.join(HISTORY_DIR, os.path.basename(tmpfile_path))
        os.rename(tmpfile_path, history_path)
        return history_path

    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile_path = tmpfile.name
            tts_kwargs = {"text": text, "file_path": tmpfile_path}
            if tts.is_multi_speaker:
                tts_kwargs["speaker"] = speaker_id
            if hasattr(tts, "is_multi_emotion") and tts.is_multi_emotion:
                tts_kwargs["emotion"] = emotion
            tts.tts_to_file(**tts_kwargs)

        y, sr = librosa.load(tmpfile_path, sr=None)
        y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch)
        y_stretched = librosa.effects.time_stretch(y_shifted, rate=speed)
        sf.write(tmpfile_path, y_stretched, sr)

        history_path = os.path.join(HISTORY_DIR, os.path.basename(tmpfile_path))
        os.rename(tmpfile_path, history_path)
        return history_path

def save_profile(profile_name, speaker_id, emotion, pitch, speed):
    profiles[profile_name] = {
        "speaker_id": speaker_id,
        "emotion": emotion,
        "pitch": pitch,
        "speed": speed,
    }
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=4)
    return gr.update(choices=list(profiles.keys()))

def load_profile(profile_name):
    p = profiles.get(profile_name, {})
    return p.get("speaker_id", ""), p.get("emotion", 0.5), p.get("pitch", 0), p.get("speed", 1.0)

with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Advanced Coqui TTS Studio V3 + OpenVoice support")

    with gr.Row():
        text_input = gr.Textbox(label="Enter text", lines=4, placeholder="Type something...")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=[label for label, _ in AVAILABLE_MODELS],
            value=AVAILABLE_MODELS[0][0],
            label="TTS Model",
        )
    
    with gr.Row():
        speaker_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label="Speaker")

    with gr.Row():
        emotion_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Emotion Intensity")
        pitch_slider = gr.Slider(-12, 12, value=0, step=1, label="Pitch Shift (semitones)")
        speed_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.01, label="Speed (rate)")

    synth_button = gr.Button("🎤 Synthesize Speech")
    audio_output = gr.Audio(label="Generated Speech")

    gr.Markdown("### 🎚️ Voice Profiles")

    with gr.Row():
        profile_name_input = gr.Textbox(label="Profile Name")
        save_profile_button = gr.Button("💾 Save Profile")
        profile_dropdown = gr.Dropdown(choices=list(profiles.keys()), label="Load Profile")
        load_profile_button = gr.Button("📂 Load Profile")

    synth_button.click(
        synthesize,
        inputs=[text_input, model_dropdown, speaker_dropdown, emotion_slider, pitch_slider, speed_slider],
        outputs=[audio_output],
    )

    model_dropdown.change(
        on_model_change,
        inputs=[model_dropdown],
        outputs=[speaker_dropdown],
    )

    save_profile_button.click(
        save_profile,
        inputs=[profile_name_input, speaker_dropdown, emotion_slider, pitch_slider, speed_slider],
        outputs=[profile_dropdown],
    )

    load_profile_button.click(
        load_profile,
        inputs=[profile_dropdown],
        outputs=[speaker_dropdown, emotion_slider, pitch_slider, speed_slider],
    )

if __name__ == "__main__":
    demo.launch(share=True)
