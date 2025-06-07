import torch
import torchaudio


def synthesize_openvoice(text, speaker_wav_path, output_wav_path):
    print(f"[OpenVoice] Synthesizing: '{text}' with speaker: {speaker_wav_path}")
    sample_rate = 22050
    duration_seconds = 3
    silence = torch.zeros(int(sample_rate * duration_seconds))
    torchaudio.save(output_wav_path, silence.unsqueeze(0), sample_rate)
    print(f"[OpenVoice] Output saved: {output_wav_path}")
    return output_wav_path
