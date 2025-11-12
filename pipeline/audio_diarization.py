"""
This file is able to handle audio diarization quite fine
It loads in segmentation and embedding models in memory as class instance to avoid initializing the models again and again.
It helps with memory.
Also the diarization is quite accurate as segmentation is almost 100% accurate with embedding reuires moving the THRESHOLD up and down.
You can test it by running this file.
"""

import asyncio
from dataclasses import replace
from pathlib import Path
import numpy as np
from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core.annotation import Segment
from scipy.spatial.distance import cdist
import torch
from constant import HUGGINGFACE_TOKEN, AUDIO_DB_PATH, AUDIO_THRESHOLD

DB_PATH = AUDIO_DB_PATH
THRESHOLD = AUDIO_THRESHOLD


class AudioDiarization:
    def __init__(self) -> None:
        self.pipeline = self.get_model_pipeline()

    def get_model_pipeline(self):
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN,
        )

    def get_tracks(self, audio_path):
        diarization = self.pipeline(audio_path)
        for segment, _, person in diarization.itertracks(yield_label=True):
            yield segment, person


class AudioSegmentationFile:
    def __init__(self) -> None:
        self.model = self.get_model()

    def get_model(self):
        return Model.from_pretrained(
            "pyannote/segmentation-3.0", use_auth_token=HUGGINGFACE_TOKEN
        )

    def voice_detection(self, audio_path):
        pipeline = VoiceActivityDetection(segmentation=self.model)
        HYPER_PARAMETERS = {"min_duration_on": 0.3, "min_duration_off": 0.0}
        pipeline.instantiate(HYPER_PARAMETERS)
        vad = pipeline(audio_path)
        for segment, _, person in vad.itertracks(yield_label=True):
            yield segment, person


class AudioEmbedding:
    def __init__(self) -> None:
        self.model = self.get_model()
        self.embedding_matrix, self.names = None, None
        # self.names = [p.name for p in DB_PATH.iterdir() if p.is_dir()]
        # self.embedding_paths = {p.name: p / "embedding.npy" for p in DB_PATH.iterdir() if p.is_dir()}
        # self.embedding_matrix = None
        self.inference = None

    def get_model(self):
        return Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")

    def get_embeddings(self, audio_path, segment: Segment):
        if self.inference is None:
            self.inference = Inference(self.model, window="whole")
        return np.atleast_2d(self.inference.crop(audio_path, segment))

    def save_embeddings(self, embedding):
        new_name = f"person_{len(list(DB_PATH.iterdir())) + 1}"
        new_dir = DB_PATH / new_name
        new_dir.mkdir(parents=True, exist_ok=True)

        # Save the embedding as a .npy file
        np.save(new_dir / "embedding.npy", embedding)
        if self.embedding_matrix is None or self.names is None:
            self.embedding_matrix, self.names = self.load_all_embeddings()
        # Instead of loading all embeddings again just add it to memory array
        else:
            self.embedding_matrix = np.vstack((self.embedding_matrix, embedding))
            self.names.append(new_name)

        print(f"Saved embedding for {new_name} at {new_dir / 'embedding.npy'}")
        return new_name

    def load_embeddings(self, person):
        return np.load(str(DB_PATH / person / "embedding.npy"))

    def load_all_embeddings(self):
        embeddings = []
        names = []

        for person_dir in DB_PATH.iterdir():
            if person_dir.is_dir() and (person_dir / "embedding.npy").exists():
                emb = np.load(person_dir / "embedding.npy")
                embeddings.append(emb)
                names.append(person_dir.name)

        if not embeddings:
            return np.empty((1, 256)), []

        return np.vstack(embeddings), names

    def compare_embeddings(self, embedding1):
        if self.embedding_matrix is None or self.names is None:
            self.embedding_matrix, self.names = self.load_all_embeddings()
            if not self.names:
                self.save_embeddings(embedding1)
                self.embedding_matrix, self.names = self.load_all_embeddings()
        print("loaded all embeddings")
        distances = cdist(self.embedding_matrix, embedding1, metric="cosine")
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx, 0]
        print("compared min dist")
        if min_dist < THRESHOLD:
            return self.names[min_idx]
        else:
            return None


def simulated_diarization(audio_path):
    aseg = AudioSegmentationFile()
    aem = AudioEmbedding()
    for segment, _ in aseg.voice_detection(audio_path):
        print("detected audio voice")
        emb = aem.get_embeddings(audio_path, segment)
        name = aem.compare_embeddings(emb)
        if name is None:
            aem.save_embeddings(emb)
        yield segment, name


def simulated_diarization_list(audio_path):
    segments = []
    for segment, _ in simulated_diarization(audio_path=audio_path):
        segments.append(segment)
    return segments


def run_async_gen_in_thread(async_gen_func, *args, **kwargs):
    """
    Runs an async generator in a separate thread and collects all outputs as a list.
    """

    async def collect():
        results = []
        async for item in async_gen_func(*args, **kwargs):
            results.append(item)
        return results

    return asyncio.run(collect())


if __name__ == "__main__":
    for segment, name in simulated_diarization(
        "/home/shreyanshp/Projects/Dymensia/pipeline/2 people conversation.opus"
    ):
        print(segment, name)
#
# aseg = AudioSegmentation()
# for output in aseg.voice_detection("audio.wav"):
#     print(output)
#
# # run the pipeline on an audio file
# diarization = pipeline("audio.wav")
# print(help(diarization))
# # dump the diarization output to disk using RTTM format
# # with open("audio1.rttm", "w") as rttm:
# #     diarization.write_rttm(rttm)
#
# for segment, _, person in diarization.itertracks(yield_label=True):
#     print(f"{person}: {segment.start} - {segment.end}. Duration: {segment.duration}")
