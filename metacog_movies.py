from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import backend as K
import scipy.misc
import numpy as np
import cv2
import os.path
import os
import tempfile
import shutil

from misc_functions import *
from metacog_datasets import *
from metacog_buildingblock_functions import *
from metacog_globals import *
from metacog_models import *



def createMovieFromImages(movie_name, img_directory):
    make_movie = '''ffmpeg -y -framerate %d -i %s/epoch%%05d.jpg -c:v libx264 -profile:v high -crf %d -pix_fmt yuv420p %s''' % (MOVIE_FRAME_RATE, img_directory, MOVIE_QUALITY, movie_name)
    os.system(make_movie)


def createActivationMovies(all_activations):

    # Every function class, individual model, epoch, problem #, hidden node
    for func_num in range(META_NUMBER_FUNCTION_CLASSES):
        for model_num in range(META_NUMBER_MODELS):
            tmp_dir = tempfile.mkdtemp(dir  =  ".")
            for epoch_num in range(EPOCHS):
                x = all_activations[func_num, model_num, epoch_num, :, :]
                np.ndarray.sort(x, -1)
                newimg = cv2.resize(x,(MOVIE_SIZE, MOVIE_SIZE))
                scipy.misc.imsave("%s/epoch%05d.jpg" % (tmp_dir, epoch_num), newimg)
            movie_name = "activation_movies/function%d_model%d.mp4" % (func_num, model_num)
            createMovieFromImages(movie_name, tmp_dir)
            shutil.rmtree(tmp_dir)


def createWeightMovies(all_weights):

    # Every function class, individual model, epoch, problem #, hidden node
    for func_num in range(META_NUMBER_FUNCTION_CLASSES):
            tmp_dir = tempfile.mkdtemp(dir  =  ".")
            for epoch_num in range(EPOCHS):
                x = all_weights[func_num, :, epoch_num, :]
                np.ndarray.sort(x, -1)
                newimg = cv2.resize(x,(MOVIE_SIZE, MOVIE_SIZE), interpolation=cv2.INTER_NEAREST)
                scipy.misc.imsave("%s/epoch%05d.jpg" % (tmp_dir, epoch_num), newimg)
            movie_name = "weight_movies/function%d.mp4" % (func_num)
            createMovieFromImages(movie_name, tmp_dir)
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    all_weights = np.load(WEIGHT_HISTORY_FILENAME)
    createWeightMovies(all_weights)










