import tensorflow as tf
import numpy as np
import keras
import spatial_transformer_custom
import data

phone_video_data = data.getFramesFromVideo("./Video1/Video1Phone008.mp4")
desktop_video_data = data.getFramesFromVideo("./Video1/Video1Desktop008.mp4")
manual_video_data = data.getFramesFromVideo("./Video1Phone008_manual.mp4")[:30,...]

# trans_matrices = tf.tile(tf.expand_dims(tf.constant([[0.7,0.0,0],[0.0,0.7,0.0],[0.0,0.0,1.0]],), axis=0), [30,1,1])
# print(tf.shape(trans_matrices))

# manual_video_data = spatial_transformer_custom.spatial_transformer_network(phone_video_data, trans_matrices)

loss_fn = keras.losses.MeanSquaredError()

nothing_loss = loss_fn(phone_video_data, desktop_video_data)
manual_loss = loss_fn(manual_video_data, desktop_video_data)

print(nothing_loss)
print(manual_loss)