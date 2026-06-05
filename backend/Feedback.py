"""
This module handles driving distraction safety feedback.
It generates safety warning text using Groq LLM and synthesizes it to speech.
"""

import os
import shutil
import time
import subprocess
import tempfile
import wave
import io
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import simpleaudio as sa
from pydub import AudioSegment

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, ".env"))

def _locate_and_add_ffmpeg_to_path():
    """Locates the FFmpeg executable and adds its directory to the system PATH."""
    if shutil.which("ffmpeg") is not None:
        return
    if os.name == 'nt':
        common_paths = [
            r"C:\Program Files\DownloadHelper CoApp",
            r"C:\Program Files (x86)\DownloadHelper CoApp",
        ]
        for p in common_paths:
            if os.path.exists(os.path.join(p, "ffmpeg.exe")):
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
                print(f"[INFO] Added {p} to PATH for FFmpeg.")
                return

_locate_and_add_ffmpeg_to_path()

print("GROQ_API_KEY found:", os.environ.get("GROQ_API_KEY") is not None)
api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None

def get_ffmpeg_cmd():
    """Returns the name of the FFmpeg command."""
    return "ffmpeg"

def generate_safety_alert_all_groq(distraction_output, warning_type: str = "moderate", play_audio=True, generate_audio=True):
    """Generates a text alert using Groq LLM and converts it to audio using TTS, optionally playing it."""
    start_time = time.time()
    warning_text = None
    raw_audio_bytes = None

    #Action mapping to increase LLMs understanding of distraction and response quality
    action_map = {
        "drinking": "drinking a beverage",
        "radio": "adjusting the radio",
        "hair_and_makeup": "fixing hair",
        "texting_right": "texting with their right hand",
        "texting_left": "texting with their left hand",
        "phonecall_right": "making a phone call with their right hand",
        "phonecall_left": "making a phone call with their left hand",
        "reach_side": "reaching to the side",
        "one hand off wheel": "driving with only one hand on the wheel",
        "both hands off wheel": "driving with no hands on the steering wheel",
    }
    dtype_raw = distraction_output.get("distraction_type", "distraction")
    dtype_for_llm = action_map.get(dtype_raw, dtype_raw)

    if groq_client:
        try:
            #Prompt engineering to get specific, content-rich warnings
            llm_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an in-car safety assistant. Output a VERY short, concise, single-sentence warning (maximum 12 words) speaking directly to the driver. "
                            "You MUST explicitly specify: "
                            "1. The exact action being performed (e.g., texting, phone call, drinking, adjusting the radio). "
                            "2. The specific hand involved if the input mentions left/right or hands-off status (e.g., 'right hand', 'left hand', 'one hand', 'both hands'). "
                            "3. The severity/danger level of the risk based on the warning type: "
                            "   - For 'light' warnings: state that it is a 'minor distraction' or 'low risk' (e.g., 'Adjusting the radio is a minor distraction, please focus'). "
                            "   - For 'light-mid' warnings: state that it is 'unsafe', 'risky', or 'moderate risk' (e.g., 'Drinking while driving is unsafe, please keep hands on the wheel'). "
                            "   - For 'heavy' warnings: state that it is 'highly dangerous', 'critical hazard', or 'extremely high risk' (e.g., 'Texting with your right hand is highly dangerous, stop immediately!'). "
                            "Be direct, specific, and authoritative. Do not make generic warnings that could apply to anything. Never assume alcohol or illegal substances."
                        )
                    },
                    {"role": "user", "content": f"The driver is {dtype_for_llm}. Give a {warning_type} warning."}
                ],
                stream=False
            )
            warning_text = llm_response.choices[0].message.content.strip().strip('"\'').replace('"', "'")
            print(f"Assistant (Groq): {warning_text}")
        except Exception as e:
            print(f"[WARN] Groq LLM failed: {e}")

    if warning_text and generate_audio:
        # MacOS TTS: concatenative TTS
        if os.name != 'nt':
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

        # Windows TTS fallback using PowerShell and System.Speech
        else:
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

        #Fallback beep alarm if TTS fails
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

    #Alert playback and latency logging
    if raw_audio_bytes:
        #print(f"[DEBUG] play_audio={play_audio}, audio_bytes size={len(raw_audio_bytes) if raw_audio_bytes else 0}")
        if play_audio:
            try:
                wave_obj = sa.WaveObject.from_wave_file(io.BytesIO(raw_audio_bytes))
                play_obj = wave_obj.play()
                play_obj.wait_done()
                print("Alert played successfully.")
            except Exception as playback_err:
                try:
                    subprocess.run(["say", "-v", "Samantha", warning_text], check=True)
                    print("Alert played via macOS say.")
                except Exception as say_err:
                    print(f"Playback error: {playback_err} | macOS say error: {say_err}")

        total_time = time.time() - start_time
        print(f"Total Alert Latency: {total_time:.3f}s")
        return raw_audio_bytes, warning_text
    else:
        print("[ERROR] No alert audio generated.")
        return None, warning_text
