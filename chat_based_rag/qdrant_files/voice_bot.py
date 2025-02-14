############### Code If You Have paid version of Assembly AI / ElevenLabs #################################
## Answers For a Live Transcript


# import assemblyai as aai
# import os
# import requests
# from elevenlabs import generate, stream
# from dotenv import load_dotenv



# load_dotenv()
# SAMBANOVA_API_URL = os.getenv("SAMBANOVA_API_URL")
# SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

# class AI_Assistant:
#     def __init__(self):
#         aai.settings.api_key = os.getenv("aai.settings.api_key")
#         self.elevenlabs_api_key = os.getenv("elevenlabs_api_key")
#         self.transcriber = None

#         # Prompt to initialize the AI
#         self.full_transcript = [
#             {"role":"system", "content":"You are a helpful assistant."},
#         ]
        
#         self.start_transcription()  # Start the transcription process right away

#     ###### Step 2: Real-Time Transcription with AssemblyAI ######
#     def start_transcription(self):
#         self.transcriber = aai.RealtimeTranscriber(
#             sample_rate=16000,
#             on_data=self.on_data,
#             on_error=self.on_error,
#             on_open=self.on_open,
#             on_close=self.on_close,
#             end_utterance_silence_threshold=1000
#         )
#         self.transcriber.connect()

#         # Start microphone stream for real-time transcription
#         microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000)
#         self.transcriber.stream(microphone_stream)

#     def stop_transcription(self):
#         if self.transcriber:
#             self.transcriber.close()
#             self.transcriber = None

#     def on_open(self, session_opened: aai.RealtimeSessionOpened):
#         print("Session ID:", session_opened.session_id)

#     def on_data(self, transcript: aai.RealtimeTranscript):
#         if not transcript.text:
#             return

#         if isinstance(transcript, aai.RealtimeFinalTranscript):
#             self.generate_ai_response(transcript)
#         else:
#             print(transcript.text, end="\r")

#     def on_error(self, error: aai.RealtimeError):
#         print("An error occurred:", error)

#     def on_close(self):
#         print("Closing session")

#     ###### Step 3: Pass real-time transcript to SambaNova API ######
#     def query_sambanova(self, prompt):
#         headers = {"Authorization": f"Bearer {SAMBANOVA_API_KEY}"}
#         payload = {
#             "model": "Meta-Llama-3.1-8B-Instruct",
#             "messages": [{"role": "system", "content": "You are a helpful assistant."},
#                          {"role": "user", "content": prompt}],
#             "max_tokens": 500
#         }
#         response = requests.post(SAMBANOVA_API_URL, json=payload, headers=headers)
#         if response.status_code == 200:
#             return response.json().get("choices")[0].get("message").get("content")
#         else:
#             raise Exception(f"SambaNova API Error: {response.status_code} - {response.text}")

#     ###### Step 4: Generate AI Response and Convert to Audio with ElevenLabs ######
#     def generate_ai_response(self, transcript):
#         self.stop_transcription()
#         self.full_transcript.append({"role": "user", "content": transcript.text})
#         print(f"\nPatient: {transcript.text}")

#         # Query SambaNova to get the response
#         ai_response = self.query_sambanova(transcript.text)

#         self.generate_audio(ai_response)  # Convert the AI response to speech
#         self.start_transcription()  # Start transcribing again after the AI response

#     def generate_audio(self, text):
#         self.full_transcript.append({"role": "assistant", "content": text})
#         print(f"\nAI Receptionist: {text}")

#         # Generate the audio stream from ElevenLabs
#         audio_stream = generate(
#             api_key=self.elevenlabs_api_key,
#             text=text,
#             voice="Rachel",
#             stream=True
#         )

#         stream(audio_stream)  # Stream the audio to the user

# ###### Step 5: Initial Greeting and Start Transcription ######
# greeting = "Thank you for calling Sails Software. My name is Sambaa, how may I assist you?"
# ai_assistant = AI_Assistant()
# ai_assistant.generate_audio(greeting)
# ai_assistant.start_transcription()  # Start transcription for user input


################ Fully Functioning Bot With Both Recording and Upload Feature (Without Live Transcript) ##########################

import streamlit as st
import assemblyai as aai
import os
import tempfile
import logging
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
# from elevenlabs import generate
from elevenlabs import generate
from dotenv import load_dotenv
from qdrant_results_1024 import rag_pipeline
import subprocess

# ✅ Set page configuration (MUST be first Streamlit command)
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎤", layout="wide")

# ✅ Load environment variables
load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("aai.settings.api_key")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ✅ Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Configure AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY
transcriber = aai.Transcriber()

# ✅ Custom CSS for Dark Theme & Mic Button
st.markdown(
    """
    <style>
    .stApp { background-color: #1E1E1E; color: white; }
    .stButton > button {
        background-color: #4CAF50; color: white; border-radius: 20px;
        padding: 10px 20px; border: none; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stButton > button:hover { background-color: #45a049; }
    .mic-button {
        display: flex; align-items: center; justify-content: center;
        background: #3a3a3a; border-radius: 50%;
        width: 60px; height: 60px; cursor: pointer;
        box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.2);
        margin: auto;
    }
    .mic-button:hover { background: #575757; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎤 Pharmacy Voice Assistant")
st.markdown("---")

# ✅ Check if greeting has been played
if 'greeting_played' not in st.session_state:
    st.session_state.greeting_played = False

# ✅ Greeting message
greeting = "Thank you for calling Sails Pharma Assistant Saambaa. How can I assist you ??"
st.write(greeting)

if not st.session_state.greeting_played:
    audio = generate(api_key=ELEVENLABS_API_KEY, text=greeting, voice="Eric")

    # ✅ Generate Audio for Greeting
    with st.spinner("Generating greeting audio..."):
        # Save the generated audio to a file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            audio_file.write(audio)
            audio_file_path = audio_file.name

        # ✅ Play Greeting Audio using MPV
        subprocess.run(["mpv", audio_file_path, "--no-video"], check=True)

        # Mark greeting as played
        st.session_state.greeting_played = True

# ✅ Audio Recording Variables
recording = st.session_state.get("recording", False)

# ✅ Function to play audio using MPV
def play_audio_with_mpv(audio_file_path: str):
    """Play audio using MPV."""
    try:
        subprocess.run(["mpv", audio_file_path, "--no-video"], check=True)
    except Exception as e:
        st.error(f"Error playing audio with MPV: {e}")

# ✅ Start/Stop Recording
def record_audio(duration=5, samplerate=44100):
    """Records audio from the microphone and saves it as a temporary file."""
    try:
        st.info("🎙️ Recording... Speak now!")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()  # Wait until recording is finished
        
        # Save to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            write(temp_wav.name, samplerate, audio_data)
            return temp_wav.name  # Return path of temp file
    
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

# ✅ Mic Button (Toggle Start/Stop)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🎙️ Start Recording" if not recording else "⏹️ Stop Recording"):
        st.session_state.recording = not recording
        if st.session_state.recording:
            audio_path = record_audio()
            if audio_path:
                st.success("✅ Recording complete! Processing audio...")

                # ✅ Transcribe Audio
                transcript = transcriber.transcribe(audio_path).text
                st.write(f"**You said:** {transcript}")

                # ✅ Retrieve Context & Generate Response
                retrieved_context = rag_pipeline(transcript)
                # prompt = f"Context: {context}\nQuestion: {transcript}"
                # ai_response = query_sambanova(prompt)

                # ✅ Display Response
                st.write(f"**AI Response:** {retrieved_context['response']}")

                # ✅ Generate Audio Response
                with st.spinner("Generating audio response..."):
                    audio = generate(api_key=ELEVENLABS_API_KEY, text=retrieved_context['response'], voice="Eric")
                    
                    # Save the generated audio to a file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
                        audio_file.write(audio)
                        audio_file_path = audio_file.name

                    # ✅ Play Audio Response using MPV
                    play_audio_with_mpv(audio_file_path)

                # ✅ Cleanup: Delete Temporary File
                os.unlink(audio_path)
                os.unlink(audio_file_path)
            else:
                st.error("❌ Audio recording failed. Please try again.")

# ✅ File Upload Option (Alternative to Mic)
st.markdown("### 🎵 Or Upload an Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[1]}") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    st.success("✅ Audio uploaded successfully! Processing...")

    # ✅ Transcribe Audio
    transcript = transcriber.transcribe(temp_path).text
    st.write(f"**You said:** {transcript}")

    # ✅ Retrieve Context & Generate Response
    retrieved_context = rag_pipeline(transcript)
    # prompt = f"Context: {context}\nQuestion: {transcript}"
    # ai_response = query_sambanova(prompt)

    # ✅ Display AI Response
    st.write(f"**AI Response:** {retrieved_context['response']}")

    # ✅ Generate & Play Audio Response
    with st.spinner("Generating audio response..."):
        audio = generate(api_key=ELEVENLABS_API_KEY, text=retrieved_context['response'], voice="Eric")
        
        # Save the generated audio to a file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            audio_file.write(audio)
            audio_file_path = audio_file.name

        # ✅ Play Audio Response using MPV
        play_audio_with_mpv(audio_file_path)

    # ✅ Cleanup: Delete Temporary File
    os.unlink(temp_path)
    os.unlink(audio_file_path)
