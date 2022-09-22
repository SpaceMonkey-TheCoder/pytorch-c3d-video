Scripts for extracting C3D feature from FC6 layer

1. Install `ffmpeg`, `pytorch`
2. Download pretrained C3D weights trained on sports1m dataset: http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle
3. Put it in the root directory
4. Copy your video files to "videos" directory
5. Run the script with ```python predict.py```
6. Your extracted features will be available on the `extracted_features` directory

Main repo forked from: https://github.com/DavideA/c3d-pytorch
