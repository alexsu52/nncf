import cv2
import openvino as ov
import torch
from openvino._offline_transformations import apply_pruning_transformation
from yolox.data.data_augment import preproc
from yoloxdetect import YoloxDetector

import nncf

# create a model instance
model = YoloxDetector(
    model_path="kadirnar/yolox_l-v0.1.1", config_path="yoloxdetect.configs.yolox_l", device="cuda:0", hf_model=True
)


# create an example input
device = "cuda:0"
image_size = 640
image = cv2.imread("/ssd/datasets/stanford_cars/cars_test/00001.jpg")
ratio = min(image_size / image.shape[0], image_size / image.shape[1])
img, _ = preproc(image, input_size=(image_size, image_size))
img = torch.from_numpy(img).to(device).unsqueeze(0).float()


model.model.eval()

# convert Torch model to OpenVINO model
ov_model = ov.convert_model(model.model, example_input=img)

# save model in FP32 precisoin
ov.save_model(ov_model, "/home/susloval/work/projects/nncf/experiments/yolox/fp32_yolox.xml", compress_to_fp16=False)


# NNCF pruning config with initial pruning rate 0.5 (It is just for example, to spped up in OpenVINO from prunoing algorithm)
nncf_config = nncf.NNCFConfig(
    {
        "input_info": {"sample_size": [1, 3, 640, 640]},
        "compression": [{"algorithm": "filter_pruning", "pruning_init": 0.5}],
    }
)


# create a prunable model with initial pruning rate 0.5.
# NNCF adds masks for each prunnable weights where value in mask is 0 then weight is pruned otherwise 1.
compression_ctrl, compressed_model = nncf.torch.create_compressed_model(model.model, nncf_config)
print(compression_ctrl.statistics().to_str())

# remove prune masks and zero pruned weights
stripped_model = nncf.strip(compressed_model)

# convert Torch model to OpenVINO model
ov_pruned_model = ov.convert_model(stripped_model, example_input=img)

# remove zero filters, the model size decreased
apply_pruning_transformation(ov_pruned_model)

# save pruned model
ov.save_model(
    ov_pruned_model,
    "/home/susloval/work/projects/nncf/experiments/yolox/fp32_pruned_yolox.xml",
    compress_to_fp16=False,
)

# benchmark FP32 model
# benchmark_app -m fp32_yolox.xml -shape [1,3,640,640]

# benchmark FP32 pruned model
# benchmark_app -m fp32_pruned_yolox.xml -shape [1,3,640,640]

# I got 1.86x speed up for pruned model (45.3% filters were pruned) on my CPU Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
