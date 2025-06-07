import gradio as gr
from TTS.api import TTS
import soundfile as sf
import librosa
import os
import json
import tempfile
from openvoice_wrapper import synthesize_openvoice

MODEL_NAME = "tts_models/en/vctk/vits"
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

PROFILES_FILE = "profiles.json"
if not os.path.exists(PROFILES_FILE):
    with open(PROFILES_FILE, "w") as f:
        json.dump({}, f)

tts = TTS(model_name=MODEL_NAME, progress_bar=False)
tts.to("cuda")

with open(PROFILES_FILE, "r") as f:
    profiles = json.load(f)

openvoice_speakers = {
    "Clone: Laura": "clones/laura.wav",
    "Clone: Mark": "clones/mark.wav",
}

speakers = (tts.speakers if tts.is_multi_speaker else ["default"]) + list(openvoice_speakers.keys())

def synthesize(text, speaker_id, emotion, pitch, speed):
    if speaker_id in openvoice_speakers:
        speaker_wav = openvoice_speakers[speaker_id]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            sf.write(tmpfile.name, y_shifted, sr)
            tmpfile_path = tmpfile.name
            return history_path
        
        os.rename(tmpfile_path, history_path)

    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tts.tts_to_file(
                text=text,
                file_path=tmpfile.name,
                speaker=speaker_id,
                emotion = emotion if hasattr(tts, "is_multi_emotion") and tts.is_multi_emotion else None,

            )
            y, sr = librosa.load(tmpfile.name, sr=None)
            y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch)
            y_stretched = librosa.effects.time_stretch(y_shifted, rate=speed)
            sf.write(tmpfile.name, y_stretched, sr)

            history_path = os.path.join(HISTORY_DIR, os.path.basename(tmpfile.name))
            os.rename(tmpfile.name, history_path)
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
    return gr.Dropdown.update(choices=list(profiles.keys()))

def load_profile(profile_name):
    p = profiles.get(profile_name, {})
    return p.get("speaker_id", ""), p.get("emotion", 0.5), p.get("pitch", 0), p.get("speed", 1.0)

with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Advanced Coqui TTS Studio V3 + OpenVoice support")

    with gr.Row():
        text_input = gr.Textbox(label="Enter text", lines=4, placeholder="Type something...")
    
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
        inputs=[text_input, speaker_dropdown, emotion_slider, pitch_slider, speed_slider],
        outputs=[audio_output],
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
