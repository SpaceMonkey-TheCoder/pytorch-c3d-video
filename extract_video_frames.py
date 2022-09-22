import os, shutil

def extract_video_frames (video_path, frame_path):
    os.system('ffmpeg -i ' + video_path + ' -q:v 2 -f image2 ' + frame_path + '/image_%5d.jpg')
    return

def list_video_dir (data_dir):
    if (not os.path.exists(data_dir)):
        print("Video director {} not found!".format(data_dir))
    return os.listdir(data_dir)

def make_frame_dir (frame_dir):
    if not os.path.exists(frame_dir):
        return os.mkdir(frame_dir)
    shutil.rmtree(frame_dir)
    return os.mkdir(frame_dir)
    