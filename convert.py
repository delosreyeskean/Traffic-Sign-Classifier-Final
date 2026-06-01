import tensorflow as tf
# Load the old format
model = tf.keras.models.load_model('traffic_classifier.h5', compile=False)
# Save it as the new format
model.save('traffic_classifier.keras')
print("Conversion complete!")
