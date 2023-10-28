from pycoral.utils.edgetpu import make_interpreter 
import tensorflow as tf

# Simple quantize/dequantize routines
def quantize(x, scale, zero_point):
    return tf.cast(x * scale + zero_point, tf.int8)

def dequantize(x, scale, zero_point):
    return (x - zero_point) / scale

# Setup runtime
interpreter = make_interpreter('my_model_edgetpu.tflite')
input_meta = interpreter.get_input_details()[0]
output_meta = interpreter.get_output_details()[0]

# Generate some random input
input = tf.random.normal([200])
quantized_input = quantize(input, input_meta['quantization'][0], input_meta['quantization'][0])
reshaped_input = tf.reshape(quantized_input, input_meta['shape'])

# Run model, get output
interpreter.allocate_tensors()
interpreter.set_tensor(input_meta['index'], reshaped_input)
interpreter.invoke()
output = interpreter.get_tensor(output_meta['index'])

# Output
dequantized_output = dequantize(output, output_meta['quantization'][0], output_meta['quantization'][0])
print(dequantized_output)