import torch
import streamlit as st
from transformers import pipeline
import librosa
import io
from audio_recorder_streamlit import audio_recorder

st.markdown(
    """
    <h1 style="text-align: center;">
        Speech to Text Yoruba Language <span style="color: lightblue;">App</span>
    </h1>
    """,
    unsafe_allow_html=True
)
col1, col2= st.columns(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline(task='automatic-speech-recognition',
                model='Humphery7/asr-yoruba-best-checkpoint',
                chunk_length_s=30,
                stride_length_s=(15, 3),
                device=device)

samplerate = 16000
MAX_DURATION = 90


# Function to record audio from the microphone
# def record_audio(duration=10, samplerate=16000)
def transcribe_audio(audio):
    prediction = pipe(audio, batch_size=8)["text"]
    return prediction

with col1:

    st.header(':blue[Record Audio]')
    duration = st.number_input('Enter Recording duration', max_value=90,
                                   min_value=1, step=1,
                                   format = '%d', value=1)
    audio_bytes = audio_recorder()
    # print(audio_bytes)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        try:
            audio_stream = io.BytesIO(audio_bytes)
            audio, sample_rate = librosa.load(audio_stream, sr=16000, mono=True)

            # st.write(f"Audio length: {len(audio)} samples")
            # st.write(f"Sample rate: {sample_rate} Hz")

            result = transcribe_audio(audio)
            st.write(':blue[Transcription:]')
            st.write(result)

        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")

with col2:
    st.header(':blue[Upload Audio]')
    uploaded_audio = st.file_uploader('Upload an Audio File', type=['wav', 'mp3'])

    if uploaded_audio is not None:
        try:

            audio_bytes = uploaded_audio.read()
            audio, sr = librosa.load(io.BytesIO(audio_bytes),sr=None, mono=True)
            duration_in_seconds = librosa.get_duration(y=audio, sr=sr)

            if duration_in_seconds > MAX_DURATION:
                st.error(f"Uploaded audio is too long! Max allowed duration is {MAX_DURATION} seconds.")
            else:
                st.success("Audio file uploaded successfully!")
                st.audio(uploaded_audio)
                if st.button('Transcribe'):
                    result = transcribe_audio(audio)
                    st.write(":blue[Transcription:]")
                    st.write(result)
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")


