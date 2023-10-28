# ONNX

## Step 1 - Install dependencies

```
python -m venv .venv
source .venv/
pip install -r requirements.txt
```

## Step 2 - Export model

`python onnx_pytorch_export.py`

## Step 3 - Inference

`python onnx_inference.py`

# EdgeTPU

## Step 1 - Create conda environment with python 3.9

```
apt update && apt install edgetpu-compiler libedgetpu1-std
conda create -n coral python=3.9
conda activate coral
conda install pip
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
pip install tensorflow
```

## Step 2 - Export quantized model

```
python tflite_tensorflow_export.py
edgetpu_compiler my_model.tflite
```

## Step 3 - Inference on EdgeTPU

`python tflite_edgetpu_inference.py`

# LLAMA