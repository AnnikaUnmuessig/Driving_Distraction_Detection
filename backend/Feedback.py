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
from groq import Groq

api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None

def generate_safety_alert_all_groq(distraction_output, play_audio=True):
    start_time = time.time()
    warning_text = None
    raw_audio_bytes = None
    
    # 1. Try Groq (Online) LLM + TTS
    if groq_client:
        try:
            llm_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a car safety AI. Output a 1 SENTENCE, authoritative warning directly to the driver."},
                    {"role": "user", "content": f"The driver is {distraction_output['distraction_type']}. Give a {distraction_output['type of warning']} warning."}
                ],
                stream=False 
            )
            warning_text = llm_response.choices[0].message.content
            print(f"Assistant (Groq): {warning_text}")
            
            tts_response = groq_client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice="hannah", 
                input=f"[authoritative] {warning_text}",
                response_format="wav"
            )
            raw_audio_bytes = tts_response.read()
            print("Groq Online LLM + TTS completed successfully.")
        except Exception as e:
            print(f"[WARN] Groq API warning generation failed: {e}. Trying offline fallback.")

    # 2. Offline Fallback Logic (Windows System.Speech or Beep Synthesizer)
    if not raw_audio_bytes:
        dtype = distraction_output.get("distraction_type", "distraction").replace("_", " ").lower()
        if dtype == "hands off wheel":
            warning_text = "Please put your hands back on the steering wheel immediately."
        else:
            warning_text = f"Warning: please stop {dtype} and focus on the road."
        print(f"Assistant (Local): {warning_text}")

        # Windows-native Speech Synthesizer fallback
        if os.name == 'nt':
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            wav_path = tmp.name
            # Generate WAV using PowerShell & System.Speech
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

        # If offline TTS failed or not on Windows, generate a clean alarm beep sound
        if not raw_audio_bytes:
            print("[INFO] Generating fallback alarm beep.")
            try:
                fs = 44100
                seconds = 1.5
                t = np.linspace(0, seconds, int(seconds * fs), False)
                # Generate double beep sound (880 Hz)
                note = np.sin(880 * t * 2 * np.pi) * 0.5
                mid = len(note) // 2
                note[mid - 2000 : mid + 2000] = 0.0 # silence gap
                audio = note * (2**15 - 1)
                audio = audio.astype(np.int16)

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

    # 3. Playback and return
    if raw_audio_bytes:
        if play_audio:
            try:
                audio_data = io.BytesIO(raw_audio_bytes)
                audio_segment = AudioSegment.from_wav(audio_data)

                # Convert to simpleaudio-compatible format
                audio_segment = audio_segment.set_frame_rate(44100).set_channels(1).set_sample_width(2)
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)

                wave_obj = sa.WaveObject.from_wave_file(wav_io)
                wave_obj.play()
                print("Playing Alert in background...")
            except Exception as playback_err:
                print(f"Playback error: {playback_err}")
                
        total_time = time.time() - start_time
        print(f"Total Alert Latency: {total_time:.3f}s")
        return raw_audio_bytes
    else:
        print("[ERROR] No alert audio generated.")
        return None