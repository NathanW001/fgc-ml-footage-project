import ffmpeg
import numpy as np

def getVideoInfo(input_video):
    probe = ffmpeg.probe(input_video)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return video_info

def getFramesFromVideo(input_video):
    # Returns each frame of the video as a numpy array. 
    in_filename = input_video

    video_info = getVideoInfo(input_video)
    width = video_info['width']
    height = video_info['height']

    process1 = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )

    ret_data = []
    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        ret_data.append(in_frame)

    process1.wait()

    return np.array(ret_data)
    

def getVideoFromFrames(numpy_arr, width, height, vid_name="test_video.mp4"):
    # Takes an array of numpy video frames and reconstructs them
    # doesnt get audio rn, can fix after

    process2 = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(vid_name, pix_fmt='yuv420p', crf='24')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for out_frame in numpy_arr:
        process2.stdin.write(
            out_frame
            .astype(np.uint8)
            .tobytes()
        )

    process2.stdin.close()
    process2.wait()