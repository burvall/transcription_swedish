import streamlit as st
from transformers import pipeline
from torch import cuda
import torchaudio
import torchaudio.functional as F
from pydub import AudioSegment
import logging
import io


class ASR:
    def __init__(self):
        self.model_name = "viktor-enzell/wav2vec2-large-voxrex-swedish-4gram"
        self.device = cuda.current_device() if cuda.is_available() else -1
        self.model = None

    def load_model(self):
        self.model = pipeline(model=self.model_name, device=self.device)

    def run_inference(self, file):
        audio = self.load_16khz_audio(file)
        return self.model(audio, chunk_length_s=10)["text"].lower()

    @staticmethod
    def load_16khz_audio(file):
        waveform, sample_rate = torchaudio.load(file)

        if sample_rate == 16_000:
            waveform = waveform[0]
        else:
            waveform = F.resample(waveform, sample_rate, 16_000)[0]

        return waveform.numpy()


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    asr = ASR()
    asr.load_model()
    return asr


@st.cache(allow_output_mutation=True, hash_funcs={ASR: lambda _: None}, show_spinner=False)
def run_inference(asr, file):
    return asr.run_inference(file)


def convert_uploaded_file_to_wav(file):
    try:
        media_type = file.type.split("/")[0]
        file_extension = file.name.split(".")[-1]

        if media_type != "audio" and media_type != "video":
            return None

        if file_extension == "wav":
            return file

        audio = AudioSegment.from_file(file, file_extension)
        in_memory_buffer = io.BytesIO()
        return audio.export(in_memory_buffer, format="wav")

    except Exception as e:
        logging.exception(e)
        return None


if __name__ == "__main__":
    st.set_page_config(
        page_title="Swedish Transcription",
        page_icon="🎙️"
    )
    st.image(
        "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/320/apple/325/studio-microphone_1f399-fe0f.png",
        width=100,
    )
    st.markdown("""
    # Swedish Speech-to-text

    Swedish transcripts for  audio and video files. The speech-to-text model is KBLab's wav2vec 2.0 large VoxRex Swedish (C) with a 4-gram language model. If you have any questions, mail joachim@burvall.com
    """)

    with st.spinner(text="Loading model..."):
        asr = load_model()

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        file = convert_uploaded_file_to_wav(uploaded_file)

        if file is None:
            st.error(
                "There was a problem handling the uploaded file. Try again using an audio or video file.")
        else:
            with st.spinner(text="Transcribing..."):
                transcript = run_inference(asr, file)
                st.download_button("Download transcript",
                                   transcript, "transcript.txt")

            with st.expander("Transcript", expanded=True):
                st.write(transcript)
