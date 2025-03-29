import data
import train
import numpy
import model
import numpy as np
import tensorflow as tf

def processVideo(video_name):
    video_info = data.getVideoInfo(video_name)
    width = video_info['width']
    height = video_info['height']
    video_data = data.getFramesFromVideo(video_name)


    mymodel = model.VideoTransformer()
    mymodel.build((-1,720,1280,3))
    mymodel.load_weights("./test.weights.h5")

    new_video_data = []
    total_frames = video_data.shape[0]
    n = 0
    for video_frame in video_data:
        new_video_data.append(mymodel(video_frame.astype("float32")[np.newaxis,...]).numpy())
        print("done frame {} out of {}. ({}%)".format(n+1, total_frames, 100*(n+1)/total_frames))
        n+=1

    mymodel.summary()
    data.getVideoFromFrames(new_video_data, width, height)

def trainModel(video_names):
    train.train(video_names)


tf.experimental.numpy.experimental_enable_numpy_behavior()

vid_names = []
for i in range(116,122):
    vid_names.append(["./Video1/Video1Phone{:03d}.mp4".format(i),"./Video1/Video1Desktop{:03d}.mp4".format(i)])

# print(vid_names)
# trainModel(vid_names)

processVideo("./Video1Long/Video1Phone008.mp4")



