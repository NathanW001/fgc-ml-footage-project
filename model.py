import tensorflow as tf
import keras
import numpy as np
import spatial_transformer_custom

class VideoTransformer(keras.Model): 
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(4, (3,3), activation='leaky_relu')
        self.maxp1 = keras.layers.MaxPool2D(4,4)
        self.conv2 = keras.layers.Conv2D(32, (3,3), activation='leaky_relu')
        self.maxp2 = keras.layers.MaxPool2D(2,2)
        self.conv3 = keras.layers.Conv2D(64, (3,3), activation='leaky_relu')
        self.maxp3 = keras.layers.MaxPool2D(6,6)
        self.flat1 = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(360, activation='relu')
        self.dense2 = keras.layers.Dense(64)
        self.dense3 = keras.layers.Dense(9)

        self.dropout = keras.layers.Dropout(0.5)

    def call(self, inputs):
        # size of inputs should be 1280 x 720 x 3, for w * h * rgb single frame.
        x = self.conv1(inputs)
        x = self.dropout(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.maxp3(x)
        x = self.flat1(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = self.dropout(x)
        # x = tf.add(x, tf.constant([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]]))
        norm_matrix = tf.tile(tf.transpose(tf.expand_dims(x[:,8], axis=0)), [1,9])
        x = tf.divide(x, tf.multiply(tf.ones_like(x), norm_matrix))
        # print(x)
        # x = tf.add(tf.sigmoid(tf.add(x, tf.constant([[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]]))), tf.constant([[0.5,0,-0.5,0,0.5,-0.5,0,0,0.5]]))
        x = tf.reshape(x,[-1,3,3])

        # print(x)
        # print(x)
        return spatial_transformer_custom.spatial_transformer_network(inputs, x)
        # return spatial_transformer_custom.spatial_transformer_network(inputs, tf.constant([[0.7,0.2,0],[0.5,0.7,0],[-0.3,0.1,1]]))


        # return self.vidtrans(inputs, x)

        # transformation_vector = x.reshape((-1, 3, 3))
        
        # print(transformation_vector)


        
        

        # print(grid, grid.shape)
        
        # sample_grid = transformation_vector @ grid

        # normalized_sample_grid = np.transpose(np.array([((sample_grid[:,0,:]//sample_grid[:,2,:])%1280), ((sample_grid[:,1,:]//sample_grid[:,2,:])%720)], dtype='int64'), (0, 2, 1))#.reshape(-1, 1280, 720, 2)
        # print(normalized_sample_grid, normalized_sample_grid.shape)
        # normalized_sample_grid = np.array(sample_grid[:,0:2,:]//sample_grid[:,[2,2],:], dtype="int64")
        # np.clip(normalized_sample_grid[:,0,:], 0, 719, out=normalized_sample_grid[:,0,:])
        # np.clip(normalized_sample_grid[:,1,:], 0, 1279, out=normalized_sample_grid[:,1,:])
        # normalized_sample_grid = np.transpose(normalized_sample_grid, axes=(0,2,1)).reshape(-1,720,1280,2)

        # print(normalized_sample_grid)
        # print(normalized_sample_grid.shape)
        # inputs = inputs.numpy()
        # inputs = inputs[:,np.newaxis,np.newaxis,...]

        # print(normalized_sample_grid[:,:,:,0].shape, normalized_sample_grid[:,:,:,1].shape)
        # ret = np.zeros((inputs.shape[0], 1280, 720, 3))
        # for batch_index in range(inputs.shape[0]):
        #     x_coords = normalized_sample_grid[batch_index,:,:,0].flatten()
        #     y_coords = normalized_sample_grid[batch_index,:,:,1].flatten()
        #     print(x_coords, y_coords, np.ones(x_coords.shape[0])*batch_index)

        #     ret[batch_index] = inputs[np.ones(x_coords.shape[0], dtype='int64')*batch_index, (x_coords, y_coords)]
        #     print(ret[batch_index])
        # print(ret_samples, ret_samples.shape)

        # np.zeros(1)[2]        

        # print(normalized_sample_grid.shape[0])
        # batches, _ = np.mgrid[0:normalized_sample_grid.shape[0], 0:(normalized_sample_grid.shape[1]*normalized_sample_grid.shape[2])]
        # batches = batches.reshape(-1, 720, 1280)

        # ret = inputs[batches,normalized_sample_grid[:,:,:,0],normalized_sample_grid[:,:,:,1]]

        # print(xx, yy, xx.shape, yy.shape)
        # print(xx.ravel(), xx.ravel().shape)
        # print(grid.shape, grid)
    
        # for proj_width in range(1280):
        #     # print(proj_width/1280*100, "% done")
        #     for proj_height in range(720):
        #         sample_coords = np.matmul(transformation_vector, np.array([proj_width, proj_height, 1]))
        #         # print(sample_coords)
        #         real_sample_coords = np.array([sample_coords[:,0]//sample_coords[:,2], sample_coords[:,1]//sample_coords[:,2]]).astype(int)
        #         real_sample_coords[:,0] %= 1280
        #         real_sample_coords[:,1] %= 720
        #         # print(real_sample_coords)
        #         # print(inputs.shape)
        #         # print(inputs.numpy()[np.array([0,1]), np.array([0,1279]),np.array([0,719])], inputs.numpy()[np.array([0,1]), np.array([0,1279]),np.array([0,719])].shape)
        #         # Modulo to stabalize the sampling from the array
                
        #         ret[:,proj_width,proj_height] = inputs.numpy()[np.arange(len(real_sample_coords[0])),real_sample_coords[:,0], real_sample_coords[:,1]]
        
        # return ret

class VideoTransformLayer(tf.keras.layers.Layer):
    # Code referenced from https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py

    def __init__(self):
        super(VideoTransformLayer, self).__init__()

    def call(self, feature_map, matrix):
        batch_size = tf.shape(feature_map)[0]

        matrix = tf.constant([[1,0,0],[0,1,0],[0,0,1]])

        xx, yy = tf.meshgrid(tf.range(720), tf.range(1280))
        xx_flat = tf.reshape(xx, [-1])
        yy_flat = tf.reshape(yy, [-1])
        homo_coord = tf.ones_like(xx_flat)
        grid = tf.stack([xx_flat, yy_flat, homo_coord])
        
        transformed_grid = tf.matmul(tf.cast(tf.reshape(matrix, [-1,3,3]), 'float32'), tf.cast(grid, 'float32'))
        transformed_grid = tf.reshape(transformed_grid, [batch_size, 3, 720, 1280])

        x_samples = tf.divide(transformed_grid[:,0,:,:], (tf.multiply(tf.ones_like(transformed_grid[:,0,:,:]),transformed_grid[:,2,:,:])))
        y_samples = tf.divide(transformed_grid[:,1,:,:], (tf.multiply(tf.ones_like(transformed_grid[:,1,:,:]),transformed_grid[:,2,:,:])))

        x_samples = tf.cast(tf.floor(x_samples), 'int32')
        y_samples = tf.cast(tf.floor(y_samples), 'int32')

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, 720, 1280))

        indices = tf.stack([b, x_samples, y_samples], 3)
        print(tf.shape(indices))
        return tf.gather_nd(feature_map, indices)

        
        