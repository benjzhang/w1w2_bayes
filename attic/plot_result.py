#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

def plot_2d(P_T, P, Q, iter, save_path, scale=None):
    plt.figure(figsize=[9,4])
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.subplot(1,2,1)
    plt.scatter(P_T[:,0],P_T[:,1], marker = 'o', color= 'r', label = 'Generated Q distribution', s = 2)
    plt.scatter(Q[:,0],Q[:,1], marker = 'o', color= 'k', label = 'Q distribution', s = 2)
    if scale != None:
        plt.xlim([-scale-0.5, scale+0.5])
        plt.ylim([-scale-0.5, scale+0.5])
    plt.title("Step: "+str(iter))
    plt.legend()
    
    plt.subplot(1,2,2)
    for i in range(100):
        index = i
        plt.arrow(P[index,0], P[index,1],P_T[index,0] - P[index,0],P_T[index,1] - P[index,1],
                  head_width=0.1,
                  head_length=0.1)
    if scale != None:
        plt.xlim([-scale-0.5, scale+0.5])
        plt.ylim([-scale-0.5, scale+0.5])
    plt.title("Learned Mapping Between P and Q")
    
    plt.savefig(save_path + str(iter)+'.png', dpi=50)
    plt.clf()
    plt.close()
    
def plot_orth_axes_saturation(P_T, iter, save_path):
    from scipy.stats import gaussian_kde
    
    complement_data = P_T[:, 2:]
    complement_data = complement_data.flatten()
    
    x= np.linspace(complement_data.min(), complement_data.max(), 1000)
    z = gaussian_kde(complement_data)(x)
        
    plt.plot(x, z, linestyle='-')
    plt.title("Step: "+str(iter))
    plt.tight_layout()
    
    plt.savefig(save_path + "orthogonal_" + str(iter)+'.png', dpi=50)
    plt.clf()
    plt.close()

def plot_image(P_T, iter, save_path):
    zero_arr = np.zeros_like(P_T[0])[np.newaxis,:]
    num_classes = 10
    print_multiplier = 10
    for i in range(num_classes):
        i_data = P_T[i*print_multiplier:(i+1)*print_multiplier]
        
        try:
            samples_ = i_data.transpose(1,0,2,3)
            newrows = np.reshape(samples_, (samples_.shape[0], samples_.shape[1]*samples_.shape[2], samples_.shape[3]))
        except:
            samples_ = i_data.transpose(1,0,2)
            newrows = np.reshape(samples_, (samples_.shape[0], samples_.shape[1]*samples_.shape[2]))
        if i == 0:
            rows = newrows
        else:
            rows = np.concatenate((rows, newrows), axis=0)
    plt.imshow(rows, interpolation='nearest', vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(save_path + str(iter)+'.png')
    plt.clf()
    plt.close()
    
