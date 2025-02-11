import streamlit as st
import requests
import io

BACKEND_URL = "http://127.0.0.1:5000"

st.title("CoppeliaSim Voice Command Interface")

# Voice Recording Section
st.subheader("Upload or Record Voice Command")
audio_buffer = st.file_uploader("Upload an audio file (wav/mp3)", type=["wav", "mp3"])

if audio_buffer is not None:
    audio_bytes = audio_buffer.read()

    # Send audio to backend
    files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
    response = requests.post(f"{BACKEND_URL}/transcribe", files=files)

    if response.status_code == 200:
        transcribed_text = response.json().get("text", "")
        st.write("Transcription:", transcribed_text)

        if st.button("Confirm & Send Command"):
            res = requests.post(f"{BACKEND_URL}/execute", json={"command": transcribed_text})
            st.success(f"Command Sent: {transcribed_text}")
    else:
        st.error("Error in transcription. Try again.")

# Pre-defined paths
st.subheader("Pre-defined paths")
path_options = ["Circle", "Square", "Line"]
selected_path = st.selectbox("Choose a path:", path_options)

# Button to send selection
if st.button("Send Path Command"):
    response = requests.post(f"{BACKEND_URL}/send_path", json={"path": selected_path})
    st.success(f"Command Sent: {selected_path}")

# Clear Button
if st.button("Clear Selection"):
    selected_path = None
    st.experimental_rerun()
