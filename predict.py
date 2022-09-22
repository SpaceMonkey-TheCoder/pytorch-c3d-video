""" How to use C3D network. """
from pydoc import cli
import numpy as np

import torch
from torch.autograd import Variable

from os.path import join, abspath
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D

import math
import extract_video_frames
import os


def get_sport_clip(clip_name, verbose=False):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?
    
    Parameters
    ----------
    clip_name: str
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """
    print(clip_name)
    clip = sorted(glob(abspath(join(clip_name, '*.jpg'))))
    print(len(clip))
    clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    batch_count = math.floor(clip.shape[2] / 16)

    print(batch_count)

    clip = clip[:, :, :16*batch_count, :]

    clip_lst = np.split(clip, int(batch_count), axis=2)

    return clip_lst


def read_labels_from_file(filepath):
    """
    Reads Sport1M labels from file
    
    Parameters
    ----------
    filepath: str
        the file.
        
    Returns
    -------
    list
        list of sport names.
    """
    with open(filepath, 'r', encoding='utf8') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

activation = {'fc6': []}
def get_activation (layer_name):
    def hook (model, input, output):
        activation['fc6'].append(output.detach().cpu())
    return hook


def main():
    """
    Main function.
    """
    frame_dir = []
    video_lst = extract_video_frames.list_video_dir('./videos')
    for v in video_lst:
        cur_dir = os.path.abspath(os.path.join(os.curdir, './data/') +v.replace(' ', '_'))
        frame_dir.append(cur_dir)
        extract_video_frames.make_frame_dir(cur_dir)

    for i in range(len(video_lst)):
        extract_video_frames.extract_video_frames('./videos/' + video_lst[i], frame_dir[i])

    for i in range(len(video_lst)):
        print("Loading data from {} . . .".format(frame_dir[i]))
        # load a clip to be predicted
        clip_lst = get_sport_clip(frame_dir[i])
        print('Data loaded!')
        with torch.no_grad():
            # get network pretrained model
            net = C3D()
            print('Loading weights . . .')
            net.load_state_dict(torch.load(os.path.abspath(os.path.join(os.curdir, 'c3d.pickle'))))
            print('Weights loaded!')
            net.fc6.register_forward_hook(get_activation('fc6'))
            print('Adding hook . . .')
            net.cuda()
            net.eval()
            k = 1
            for X in clip_lst:
                print('Batch shape: ' + str(X.shape))
                torch.cuda.empty_cache()
                X = Variable(torch.from_numpy(X))
                X = X.cuda()

                # perform prediction
                prediction = net(X)
                print("Completed batch {}/{}".format(k, len(clip_lst)))
                k +=1

            print(len(activation['fc6']))
            np.save(os.path.abspath(os.path.join(os.curdir , './extracted_features/' + video_lst[i].split('.')[0] + '.npy')), np.array(activation['fc6'], dtype=object), allow_pickle=True)


# entry point
if __name__ == '__main__':
    main()
