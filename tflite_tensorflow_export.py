import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=100, input_shape=[200], activation='relu'),
    tf.keras.layers.Dense(units=50, input_shape=[100], activation='relu'),
    tf.keras.layers.Dense(units=1, input_shape=[50], activation='sigmoid')
])

def quantize_dataset():
    for i in range(100):
        yield [tf.random.normal([200])]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = quantize_dataset
tflite_model = converter.convert()

with open('my_model.tflite', 'wb') as f:
    f.write(tflite_model)