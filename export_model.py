import tensorflow as tf

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
model = tf.keras.models.load_model('/Users/xueyingwen/Downloads/model.h5', custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform})
export_path = '/Users/xueyingwen/Downloads/House-Number-Recognition/house_number_recognition_model/v1'


with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})

