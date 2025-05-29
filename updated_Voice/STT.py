# HUGGINGFACE_TOKEN = "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"  # 🔒 REPLACE THIS


# updated_Voice/STT.py
import sounddevice as sd
import numpy as np
import torch
import wave
import tempfile
import os
import threading
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# === Settings ===
SAMPLE_RATE = 16000
WHISPER_MODEL_SIZE = "base.en"

# === Load environment variables ===
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("Missing Hugging Face token. Set HUGGINGFACE_TOKEN in a .env file.")

# === Load Whisper ===
print("[+] Loading Whisper model...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

# === Load diarization pipeline ===
print("[+] Loading speaker diarization pipeline...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HUGGINGFACE_TOKEN
)

# === Globals ===
recorded_audio = []

# === Callback for real-time audio recording ===
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_chunk = indata.copy()
    recorded_audio.append(audio_chunk)

# === Record from microphone ===
def record_audio(duration=None):
    print("[*] Press Ctrl+C to stop recording and get diarization.")
    recorded_audio.clear()
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            if duration:
                sd.sleep(int(duration * 1000))
            else:
                while True:
                    sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n[+] Transcription ended.")

# === Save audio to WAV file ===
def save_audio_to_wav():
    if not recorded_audio:
        print("[!] No audio recorded. Exiting.")
        return None
    audio_data = np.concatenate(recorded_audio, axis=0)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    return temp_file.name

# === Transcribe live (stream simulation) ===
def live_transcribe():
    print("[+] Starting live transcription...")
    while True:
        try:
            if len(recorded_audio) >= 10:
                chunk = np.concatenate(recorded_audio[:10], axis=0)
                del recorded_audio[:10]

                audio_float = chunk.astype(np.float32)
                segments, _ = whisper_model.transcribe(audio_float, language="en", beam_size=5, vad_filter=False)

                if segments:
                    for segment in segments:
                        print(f"[Live] {segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[!] Error in live transcription: {e}")

# === Run diarization ===
def run_diarization(audio_path):
    print("[*] Running speaker diarization...")
    diarization = diarization_pipeline(audio_path)

    print("\n[+] Diarization result:")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"{speaker}: {turn.start:.1f}s - {turn.end:.1f}s")

# === Main ===
def main():
    print("[*] Starting STT + diarization pipeline...")

    # Start live transcription in parallel
    transcribe_thread = threading.Thread(target=live_transcribe, daemon=True)
    transcribe_thread.start()

    # Record until Ctrl+C
    record_audio()

    # Save and analyze
    audio_path = save_audio_to_wav()
    if audio_path:
        run_diarization(audio_path)
        os.remove(audio_path)

if __name__ == "__main__":
    main()
