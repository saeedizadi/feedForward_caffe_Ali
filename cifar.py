#! /usr/bin/env python

import cPickle as pickle
import math

def unpickle(file_path):
    with open(file_path,'rb') as fo:
        dict = pickle.load(fo)
    return dict

class CIFAR_Image:
    def __init__(self,name,data, mat_mode=True):
        self.data = data
        self.name = name
        if mat_mode:
            self.red_mat = self.color_matrix(0,32,32)
            self.green_mat = self.color_matrix(1024,32,32)
            self.blue_mat = self.color_matrix(2014,32,32)
            del self.data

    def color_matrix(self,start_index,num_rows,num_cols):
        mat = [[None] * num_cols] * num_rows
        print mat.shape
        num_elements = num_rows * num_cols
        for i in range(num_elements):
            mat[i/num_cols][i % num_rows] = self.data[i + start_index]
        
        return mat

