import tensorflow as tf
import numpy as np
import data
import keras
import model


def train(video_names):
    # Video names is an array of arrays of names
    # [["Video1.mp4", "Video1Real.mp4"], ["Video2.mp4", "Video2Real.mp4"]]

    # tf.experimental.numpy.experimental_enable_numpy_behavior()

    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    # Instantiate a loss function.
    loss_fn = keras.losses.BinaryCrossentropy()
    video_model = model.VideoTransformer()
    # video_model.build((-1,720,1280,3))
    # video_model.load_weights("./test.weights.h5")

    video_model.compile(
        loss=loss_fn,
        optimizer=optimizer
    )

    # Prepare the training dataset.
    batch_size = 2
    x_data = data.getFramesFromVideo(video_names[0][0]).astype("float32")
    y_data = data.getFramesFromVideo(video_names[0][1]).astype("float32")

    print(video_names)

    for video_name in video_names[1:]:
        x_data = np.concatenate((x_data, data.getFramesFromVideo(video_name[0]).astype("float32")))
        y_data = np.concatenate((y_data, data.getFramesFromVideo(video_name[1]).astype("float32")))

    
    data_size = x_data.shape[0]
    split_pos = data_size//5

    print(data_size, x_data.shape)
    print(x_data.shape, y_data.shape)

    x_train = x_data[split_pos:]
    x_test = x_data[:split_pos]
    y_train = y_data[split_pos:]
    y_test = y_data[:split_pos]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1028).batch(batch_size)

    data_augmentation = tf.keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
    ])

    # video_model.fit(x_train, y_train, batch_size=16, epochs=2, verbose=2)

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            first_frame = None
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                x_batch_train = data_augmentation(x_batch_train)
                frame_out = video_model(x_batch_train, training=True)  # Frame output for this minibatch
                if first_frame is None:
                    first_frame = frame_out
                # print(frame_out)
                
                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, frame_out)
                print(loss_value)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, video_model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, video_model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
                data.getVideoFromFrames(first_frame.numpy(), 1280, 720, vid_name="z_test_video_{}{}.mp4".format(epoch,step))

    video_model.evaluate(x_test, y_test, batch_size=2)

    video_model.summary()

    video_model.save_weights('./test.weights.h5')