"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import multiprocessing
import sys
import tempfile
from multiprocessing import Process
from pathlib import Path
import time

import numpy as np
import openvino.runtime as ov
import torch
from sklearn.metrics import accuracy_score
from openvino.offline_transformations import compress_quantize_weights_transformation
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

FOOD101_CLASSES = 101
ROOT = Path(__file__).parent.resolve()
DATASET_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'dataset'
CHECKPOINT_URL = 'https://huggingface.co/AlexKoff88/mobilenet_v2_food101/resolve/main/pytorch_model.bin'


def fix_names(state_dict):
    state_dict = {key.replace('module.', ''): value for (key, value) in state_dict.items()}
    return state_dict


def load_checkpoint(model):  
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, progress=False)
    weights = fix_names(checkpoint['state_dict'])
    model.load_state_dict(weights)
    return model


def validate(model, val_loader):
    predictions = []
    references = []

    output_name = model.outputs[0]

    for images, target in tqdm(val_loader):
        pred = model(images)[output_name]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)  
    return accuracy_score(predictions, references)


def ov_convert(model, args):
    onnx_model_path = f'{tempfile.gettempdir()}/model.onnx'
    torch.onnx.export(model, args, onnx_model_path, verbose=False)

    ov_model = ov.Core().read_model(onnx_model_path)
    compress_quantize_weights_transformation(ov_model)
    return ov_model


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
val_dataset = datasets.Food101(
    root=DATASET_PATH,
    split = 'test', 
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),
    download = True
)
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, num_workers=4, shuffle=False)

model = models.mobilenet_v2(num_classes=FOOD101_CLASSES) 
model.eval()
model = load_checkpoint(model)

dummy_input = torch.randn(1, 3, 224, 224)
ov_model = ov_convert(model.cpu(), dummy_input)

ov_model.reshape([-1, 3, 224, 224])

def fun(ov_model, val_loader, num_threads):
    config = {}
    if num_threads > 1:
        config['AFFINITY'] = 'NUMA'
        config['INFERENCE_NUM_THREADS'] = multiprocessing.cpu_count() / num_threads
    compiled_model = ov.Core().compile_model(ov_model, device_name='CPU', config=config)
    validate(compiled_model, val_loader)

p0 = Process(target=fun, args=(ov_model, val_loader, 1))
start_time = time.time()
p0.start()
p0.join()
wall_time = time.time() - start_time
print(f'process time (1 validation): {wall_time:.3f}')

p1 = Process(target=fun, args=(ov_model, val_loader, 2))
p2 = Process(target=fun, args=(ov_model, val_loader, 2))
start_time = time.time()
p1.start()
p2.start()
p1.join()
p2.join()
wall_time = time.time() - start_time
print(f'process time (2 parallel validation): {wall_time:.3f}')