"""
Audio diarization module for speaker identification and voice activity detection.

This module handles audio diarization by loading segmentation and embedding models
into memory as class instances to avoid re-initialization. The diarization accuracy
is high with segmentation being almost 100% accurate, while embedding accuracy can
be tuned by adjusting the THRESHOLD parameter.
"""

import asyncio
import numpy as np
from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core.annotation import Segment
from scipy.spatial.distance import cdist
from constant import HUGGINGFACE_TOKEN, AUDIO_DB_PATH, AUDIO_THRESHOLD

DB_PATH = AUDIO_DB_PATH
THRESHOLD = AUDIO_THRESHOLD


class AudioDiarization:
    """
    Speaker diarization using pyannote's pre-trained pipeline.

    Identifies and segments audio by different speakers using the
    pyannote/speaker-diarization-3.1 model.
    """

    def __init__(self) -> None:
        """Initialize the AudioDiarization with the speaker diarization pipeline."""
        self.pipeline = self.get_model_pipeline()

    def get_model_pipeline(self):
        """
        Load the speaker diarization pipeline from Hugging Face.

        :return: Loaded pyannote speaker diarization pipeline
        """
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN,
        )

    def get_tracks(self, audio_path):
        """
        Get speaker tracks from an audio file.

        :param audio_path: Path to the audio file to process
        :return: Generator yielding (segment, speaker_label) tuples
        """
        diarization = self.pipeline(audio_path)
        for segment, _, person in diarization.itertracks(yield_label=True):
            yield segment, person


class AudioSegmentationFile:
    """
    Voice activity detection using pyannote's segmentation model.

    Detects speech segments in audio files with high accuracy (~100%) using
    the pyannote/segmentation-3.0 model.
    """

    def __init__(self) -> None:
        """Initialize the AudioSegmentationFile with the segmentation model."""
        self.model = self.get_model()

    def get_model(self):
        """
        Load the segmentation model from Hugging Face.

        :return: Loaded pyannote segmentation model
        """
        return Model.from_pretrained(
            "pyannote/segmentation-3.0", use_auth_token=HUGGINGFACE_TOKEN
        )

    def voice_detection(self, audio_path):
        """
        Detect voice activity in an audio file.

        Uses voice activity detection to identify speech segments with
        configurable minimum durations for speech and silence.

        :param audio_path: Path to the audio file to process
        :return: Generator yielding (segment, label) tuples for detected speech
        """
        pipeline = VoiceActivityDetection(segmentation=self.model)
        HYPER_PARAMETERS = {"min_duration_on": 0.3, "min_duration_off": 0.0}
        pipeline.instantiate(HYPER_PARAMETERS)
        vad = pipeline(audio_path)
        for segment, _, person in vad.itertracks(yield_label=True):
            yield segment, person


class AudioEmbedding:
    """
    Speaker embedding extraction and comparison for speaker identification.

    Uses the wespeaker-voxceleb-resnet34-LM model to generate speaker embeddings
    and compare them against a database of known speakers. Matches are determined
    by cosine distance threshold.
    """

    def __init__(self) -> None:
        """
        Initialize the AudioEmbedding with the speaker embedding model.

        Sets up the model and initializes embedding storage for speaker comparison.
        """
        self.model = self.get_model()
        self.embedding_matrix, self.names = None, None
        self.inference = None

    def get_model(self):
        """
        Load the speaker embedding model from Hugging Face.

        :return: Loaded wespeaker-voxceleb-resnet34-LM model
        """
        return Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")

    def get_embeddings(self, audio_path, segment: Segment):
        """
        Extract speaker embeddings from an audio segment.

        :param audio_path: Path to the audio file
        :param segment: Audio segment to extract embedding from
        :return: 2D numpy array containing the speaker embedding vector
        """
        if self.inference is None:
            self.inference = Inference(self.model, window="whole")
        return np.atleast_2d(self.inference.crop(audio_path, segment))

    def save_embeddings(self, embedding):
        """
        Save a new speaker embedding to the database.

        Creates a new directory with an auto-generated name (person_N) and saves
        the embedding as a .npy file. Updates the in-memory embedding matrix.

        :param embedding: Speaker embedding vector to save
        :return: Auto-generated name for the new speaker (e.g., "person_1")
        """
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
        """
        Load a specific speaker's embedding from the database.

        :param person: Name/identifier of the speaker
        :return: Numpy array containing the speaker's embedding
        """
        return np.load(str(DB_PATH / person / "embedding.npy"))

    def load_all_embeddings(self):
        """
        Load all speaker embeddings from the database into memory.

        Scans the database directory and loads all existing speaker embeddings
        into a single matrix for efficient comparison.

        :return: Tuple of (embedding_matrix, speaker_names) where embedding_matrix
                 is a 2D array with all embeddings stacked, and speaker_names is
                 a list of corresponding speaker identifiers
        """
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
        """
        Compare an embedding against all known speakers in the database.

        Uses cosine distance to find the closest match. If the minimum distance
        is below the threshold, returns the matching speaker's name. Otherwise,
        saves the embedding as a new speaker.

        :param embedding1: Speaker embedding to compare
        :return: Speaker name if match found below threshold, None otherwise
        """
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
    """
    Perform complete speaker diarization on an audio file.

    Combines voice activity detection with speaker identification to segment
    audio and identify speakers. New speakers are automatically added to the
    database when encountered.

    :param audio_path: Path to the audio file to process
    :return: Generator yielding (segment, speaker_name) tuples. speaker_name
             may be None for newly encountered speakers on first detection
    """
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
    """
    Get all voice activity segments from an audio file as a list.

    Convenience function that collects all detected speech segments into a list.

    :param audio_path: Path to the audio file to process
    :return: List of all detected speech segments
    """
    segments = []
    for segment, _ in simulated_diarization(audio_path=audio_path):
        segments.append(segment)
    return segments


def run_async_gen_in_thread(async_gen_func, *args, **kwargs):
    """
    Run an async generator function in a separate thread and collect results.

    Helper function to execute async generators synchronously by running them
    in an event loop and collecting all yielded values into a list.

    :param async_gen_func: Async generator function to execute
    :param args: Positional arguments to pass to the async generator
    :param kwargs: Keyword arguments to pass to the async generator
    :return: List of all values yielded by the async generator
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
