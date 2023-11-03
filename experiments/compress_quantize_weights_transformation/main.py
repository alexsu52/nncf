import time

import openvino.runtime as ov
from openvino._offline_transformations import compress_quantize_weights_transformation

model = ov.Core().read_model("/home/susloval/work/projects/nncf/dolly_full_quantized/openvino_model.xml")
print("model is loaded...")

tic = time.perf_counter()
compress_quantize_weights_transformation(model)
toc = time.perf_counter()
print(f"Compression wall time: {toc - tic:0.4f} seconds")
