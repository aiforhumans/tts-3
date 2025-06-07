# Advanced Local TTS Studio

This repository hosts **Advanced Local TTS Studio V3.2**, a Gradio based interface around [Coqui TTS](https://github.com/coqui-ai/TTS) with a basic wrapper for OpenVoice. The app lets you pick from several high quality models, clone voices, adjust pitch and speed, and save reusable voice profiles.

## Installation

1. Install PyTorch nightly with CUDA 12.8:
   ```bash
   pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```
2. Install the remaining requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App
Start the Gradio UI with:
```bash
python app_v3.py
```
The console will print the local URL where the interface is available.

### Adding Cloned Voices
Create a folder named `clones` in the project root and drop `.wav` files inside. Each file will appear in the speaker dropdown as `Clone: <name>`. Example:
```
clones/
├── laura.wav
└── mark.wav
```

### Example Command
```bash
python app_v3.py
```

## Notes
- Generated audio files are stored in the `history` folder.
- Voice profiles are saved in `profiles.json` and can be managed from the UI.
- `openvoice_wrapper.py` currently contains a stub function; extend it to perform real OpenVoice synthesis.
- Gradio 5.33 is pinned for stability.
