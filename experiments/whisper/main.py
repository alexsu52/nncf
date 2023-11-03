import nncf

from nncf.torch import patch_torch_operators

patch_torch_operators()

import torch
import whisper
from torch.utils.tensorboard import SummaryWriter

from nncf.experimental.torch.quantization.quantize_model import wrap_model


def dump_torch_graph(model, input_to_model, logdir):
    writer = SummaryWriter(logdir)
    writer.add_graph(model, input_to_model=input_to_model)
    writer.close()


model = whisper.load_model("base")
model.eval()


def compress_encoder(model):
    # dump_torch_graph
    mel = torch.zeros((1, 80, 3000)).cuda()
    audio_features = model.encoder(mel)
    # dump_torch_graph(model.encoder, mel, "runs/whisper/encoder")

    # dump_nncf_graph
    wrapped_model = wrap_model(model.encoder, mel)
    wrapped_model.nncf.get_graph().visualize_graph("runs/whisper/encoder/original_graph.dot")

    # compress_weights
    # compressed_model = nncf.compress_weights(model.encoder)

    # dump_torch_graph
    # audio_features = compressed_model(mel)
    # dump_torch_graph(compressed_model, mel, "runs/whisper/encoder/compressed")

    # dump_nncf_graph
    # wrapped_compressed_model = create_nncf_network(compressed_model, mel)
    # wrapped_compressed_model.nncf.get_graph().visualize_graph("runs/whisper/encoder/compressed_graph.dot")


def compress_decoder(model):
    # dump_torch_graph
    tokens = torch.zeros((1, 3), dtype=torch.int64).cuda()
    audio_features = torch.zeros((1, 1500, 512)).cuda()
    mel = torch.zeros((1, 80, 3000)).cuda()
    # audio_features = model.encoder(mel)
    text = model.decoder(tokens, audio_features)
    # dump_torch_graph(model.decoder, (tokens, audio_features), "runs/whisper/decoder")

    # dump_nncf_graph
    wrapped_model = wrap_model(model.decoder, (tokens, audio_features))
    wrapped_model.nncf.get_graph().visualize_graph("runs/whisper/decoder/original_graph.dot")

    # compress_weights
    # compressed_model = nncf.compress_weights(model.decoder)

    # dump_torch_graph
    # text = compressed_model(tokens, audio_features)
    # dump_torch_graph(compressed_model, (tokens, audio_features), "runs/whisper/decoder/compressed")

    # # dump_nncf_graph
    # wrapped_compressed_model = create_nncf_network(compressed_model, (tokens, audio_features))
    # wrapped_compressed_model.nncf.get_graph().visualize_graph("runs/whisper/decoder/compressed_graph.dot")


compress_encoder(model)

compress_decoder(model)

# ---------------------------------

import ipywidgets as widgets

VIDEO_LINK = "https://youtu.be/kgL5LBM-hFI"
link = widgets.Text(value=VIDEO_LINK, placeholder="Type link for video", description="Video:", disabled=False)

# ---------------------------------

from pathlib import Path

from pytube import YouTube

print(f"Downloading video {link.value} started")

output_file = Path("downloaded_video.mp4")
yt = YouTube(link.value)
yt.streams.get_highest_resolution().download(filename=output_file)
print(f"Video saved to {output_file}")

# ---------------------------------

from experiments.whisper.utils import get_audio

audio = get_audio(output_file)

# ---------------------------------

task = widgets.Select(
    options=["transcribe", "translate"], value="translate", description="Select task:", disabled=False
)

# ---------------------------------

transcription = model.transcribe(audio, task=task.value)

# ---------------------------------

from experiments.whisper.utils import prepare_srt

srt_lines = prepare_srt(transcription)
print("".join(srt_lines))
