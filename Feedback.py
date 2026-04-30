from transformers import AutoTokenizer, AutoModelForCausalLM
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from huggingface_hub import login
import torch
import time
import os
api_key = os.environ.get("GROQ_API_KEY")


#VERSION 2: GROQ
from groq import Groq
groq_client = Groq(api_key=api_key)

distraction_output = {"distracted": "yes", "distraction_type": "texting", "type of warning": "heavy"}

import time
import io
from groq import Groq
from pydub import AudioSegment
from pydub.playback import play


def generate_safety_alert_all_groq():
    start_time = time.time()
    
    # 1. LLM Generation (Non-streaming for TTS input)
    # We need the full string to pass to the TTS endpoint
    llm_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a car safety AI. Output a 1 SENTENCE, authoritative warning directly to the driver."},
            {"role": "user", "content": f"The driver is {distraction_output['distraction_type']}. Give a {distraction_output['type of warning']} warning."}
        ],
        stream=False 
    )
    
    warning_text = llm_response.choices[0].message.content
    print(f"Assistant: {warning_text}")
    llm_done = time.time()

    # 2. Groq Text-to-Speech (TTS)
    # Using the Orpheus model for low-latency speech
    tts_response = groq_client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="hannah", 
            input=f"[authoritative] {warning_text}",
            response_format="wav"
        )
    
    tts_done = time.time()

    # 3. Playback - FIX START
    # Use .read() to get the bytes from the BinaryAPIResponse
    raw_audio_bytes = tts_response.read() 
    
    audio_data = io.BytesIO(raw_audio_bytes)
    audio_segment = AudioSegment.from_wav(audio_data)
    
    print("Playing Alert...")
    play(audio_segment)
    
    total_time = time.time() - start_time
    print(f"\n--- Performance Breakdown ---")
    print(f"LLM Generation: {llm_done - start_time:.3f}s")
    print(f"TTS Synthesis:  {tts_done - llm_done:.3f}s")
    print(f"Total Latency:   {total_time:.3f}s")


generate_safety_alert_all_groq()