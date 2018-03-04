# __author__ = 'ktc312'
#  -*- coding: utf-8 -*-
# coding: utf-8
import os
from keras.models import load_model
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'classifier/data/')
model = load_model(data_path + "model_f.h5")
graph = tf.get_default_graph()


def get_predict(input_array):
    with graph.as_default():
        p = model.predict(input_array)
    return p
