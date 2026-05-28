from transformers import AutoTokenizer, AutoModelForCausalLM
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from huggingface_hub import login
import torch
import time
import os
import simpleaudio as sa
import numpy as np
import subprocess
import tempfile
import wave
import io
from pydub import AudioSegment
from dotenv import load_dotenv
from groq import Groq
import pyttsx3

load_dotenv()
print("GROQ_API_KEY found:", os.environ.get("GROQ_API_KEY") is not None)

api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None


def generate_safety_alert_all_groq(distraction_output, warning_type: str = "moderate", play_audio=True):
    start_time = time.time()
    warning_text = None
    raw_audio_bytes = None

    # 1. Groq LLM (text only)
    if groq_client:
        try:
            llm_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a car safety AI. Output a VERY short, one sentence, authoritative warning directly to the driver, consider the type of distraction."},
                    {"role": "user", "content": f"The driver is {distraction_output['distraction_type']}. Give a {warning_type} warning."}
                ],
                stream=False
            )
            warning_text = llm_response.choices[0].message.content
            print(f"Assistant (Groq): {warning_text}")
        except Exception as e:
            print(f"[WARN] Groq LLM failed: {e}")

    # 2. pyttsx3 local TTS
    if warning_text:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False)
            tmp.close()
            aiff_path = tmp.name

            subprocess.run(["say", "-v", "Samantha", "-o", aiff_path, warning_text], check=True)

            if os.path.exists(aiff_path) and os.path.getsize(aiff_path) > 0:
                audio_segment = AudioSegment.from_file(aiff_path, format="aiff")
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                raw_audio_bytes = wav_io.getvalue()
                print("[OK] macOS say TTS generated successfully.")
            if os.path.exists(aiff_path):
                os.remove(aiff_path)
        except Exception as e:
            print(f"[WARN] macOS say TTS failed: {e}")

    # 3. Offline Fallback Logic
    if not raw_audio_bytes:
        dtype = distraction_output.get("distraction_type", "distraction").replace("_", " ").lower()
        warning_text = warning_text or f"Warning: please stop {dtype} and focus on the road."
        print(f"Assistant (Local): {warning_text}")

        # Windows-native Speech Synthesizer fallback
        if os.name == 'nt':
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            wav_path = tmp.name
            ps_cmd = f'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.SetOutputToWaveFile("{wav_path}"); $synth.Speak("{warning_text}"); $synth.Dispose()'
            try:
                subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, check=True)
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    with open(wav_path, "rb") as f:
                        raw_audio_bytes = f.read()
                    print("[OK] Offline Windows TTS generated successfully.")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception as ps_err:
                print(f"[ERROR] Offline Windows TTS failed: {ps_err}")

        # Last resort: beep
        if not raw_audio_bytes:
            print("[INFO] Generating fallback alarm beep.")
            try:
                fs = 44100
                seconds = 1.5
                t = np.linspace(0, seconds, int(seconds * fs), False)
                note = np.sin(880 * t * 2 * np.pi) * 0.5
                mid = len(note) // 2
                note[mid - 2000: mid + 2000] = 0.0
                audio = (note * (2**15 - 1)).astype(np.int16)

                wav_io = io.BytesIO()
                with wave.open(wav_io, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(fs)
                    wav_file.writeframes(audio.tobytes())
                raw_audio_bytes = wav_io.getvalue()
                print("[OK] Fallback alarm beep generated successfully.")
            except Exception as beep_err:
                print(f"[ERROR] Failed to generate alarm beep: {beep_err}")

    # 4. Playback and return
    if raw_audio_bytes:
        if play_audio:
            try:
                subprocess.run(["say", "-v", "Samantha", warning_text], check=True)
                print("Alert played.")
            except Exception as playback_err:
                print(f"Playback error: {playback_err}")

        total_time = time.time() - start_time
        print(f"Total Alert Latency: {total_time:.3f}s")
        return raw_audio_bytes, warning_text
    else:
        print("[ERROR] No alert audio generated.")
        return None, warning_text


#generate_safety_alert_all_groq({'distraction_type': 'phone', 'severity': 'critical'})
