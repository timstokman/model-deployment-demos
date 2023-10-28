import onnxruntime
import numpy as np

session = onnxruntime.InferenceSession('my_model.onnx')
output = session.run(None, {'input': np.random.randn(200).astype(np.float32)})
print(output[0])