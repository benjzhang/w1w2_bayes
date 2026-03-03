#!/usr/bin/env python
# coding: utf-8
# Proximal OT Flow GAN with Wasserstein-p proximals with p>1

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid tensorflow warning
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import pickle



import argparse
parser = argparse.ArgumentParser()

# Data configuration
parser.add_argument(
    '--example', type=str, choices=['Radius_conditioning'],
)
parser.add_argument(
    '--Rep', type=int, help='random seed', default=0,
)

# Loss/flow configuration
parser.add_argument(
    '--power', type=float, help='power p for W_p distance',  default=2.0,
)
parser.add_argument(
    '--alpha1', type=float, help='coefficient for L2 regularizer', default=0.01#0.5
)
parser.add_argument(
    '--alpha2', type=float, help='coefficient for HJB regularizer', default=0.1
)
parser.add_argument(
    '--alpha3', type=float, help='coefficient for terminal condition regularizer', default=0.1
)
parser.add_argument(
    '--loss_case', type=str, help='OT: OT regularization only, PG: potential generator, OC: optimality conditions, No_OT: no OT regularization', default='OT',
)
parser.add_argument(
    '--f', type=str, choices=['KL', 'alpha', 'reverse_KL', 'W1'], default='reverse_KL',
)
parser.add_argument(
    '-L', type=float, help='Lipschitz constant for discriminator',
)
parser.add_argument(
    '--T', type=float, help='terminal time', default=1.0
)
parser.add_argument(
    '--dt', type=float, help='time step size', default=0.2
)
parser.add_argument(
    '--sigma', type=float, help='noise level for SDE', default=0.0
)

# Neural network/training configuration
parser.add_argument(
    '--gen_dims', type=int, nargs="+", default= [128, 128, 128, 128]#[48, 48, 48, 48, 48]
)
parser.add_argument(
    '--disc_dims', type=int, nargs="+", default= [128, 128, 128,]#[48, 48, 48]
)
parser.add_argument(
    '--iterations', type=int, help='number of training interations', default=10000,
)
parser.add_argument(
    '--lr_disc', type=float, help='learning rate for discriminator', default=2e-5
)
parser.add_argument(
    '--lr_gen', type=float, help='learning rate for generator(potential)', default=1e-4
)
parser.add_argument(
    '--lamda', type=float, help='coefficient for Gradient Penalty (discriminator)', default=0.1
)

# Miscellaneous parameters
parser.add_argument(
    '--CUDA', type=str, help='number of CUDA devices', default='0',
)
parser.add_argument(
    '-savedir', type=str, help='specific save directory',
)


p = parser.parse_args()

if p.power <= 1.0:
    raise ValueError

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= p.CUDA

Rep = p.Rep

tf.reset_default_graph()
tf.set_random_seed(Rep)
np.random.seed(Rep)

# data generation
class PriorDataGenerator:
    def __init__(self, n_samples, dataset):
        self.n_samples = n_samples
        if dataset == 'Radius_conditioning':
            self.data, self.label = self.Uniform_disk()
        # Here for other datasets
        self.data = self.data.astype(np.float32)
        self.dim = self.data.shape[1]
        self.label_dim = self.label.shape[1]
        
        
    def nextbatch(self, bs):
        indices = np.random.choice(self.n_samples, bs, replace = False)
        return self.data[indices,:], self.label[indices,:]

    def Uniform_disk(self):
        # sample from inside of a unit disk
        theta = np.random.uniform(0.0, 2*np.pi, [self.n_samples, 1])
        r = np.sqrt(np.random.uniform(0.0, 5**2, [self.n_samples, 1])) # sqrt(uniform([0,r^2]))
        X = np.concatenate([r * np.cos(theta), r * np.sin(theta)], axis=1)
        return X, r

class PosteriorDataGenerator:
    def __init__(self, n_samples, dataset, label):
        self.n_samples = n_samples
        self.label = label.astype(np.float32) #* np.ones([self.n_samples,1], dtype=np.float32)
            
        if dataset == 'Radius_conditioning':
            self.data = self.Radius_conditioning()
        self.data = self.data.astype(np.float32)
        # Here for other datasets
        self.dim = self.data.shape[1]
        self.label_dim = self.label.shape[1]

    def nextbatch(self, bs):
        indices = np.random.choice(self.n_samples, bs, replace = False)
        return self.data[indices,:], self.label[indices,:]
       
        
    def Radius_conditioning(self):
        rad = np.random.uniform(0.0, 2*np.pi, [self.n_samples, 1])
        X_y = self.label * np.concatenate([np.cos(rad), np.sin(rad)], axis=1)
        return X_y

# input dataset for training is fixed
if p.example == 'Radius_conditioning':
    dim = 2
    label_dim = 1

bs = 1000
lamda = p.lamda
alpha1 = p.alpha1
alpha2 = p.alpha2
alpha3 = p.alpha3
T = p.T
dt = p.dt



# neural network structure
def feed_NN(X, W, b, act = tf.nn.tanh):
    A = X
    L = len(W)
    for i in range(L-1):
        A = act(tf.add(tf.matmul(A, W[i]), b[i]))
    return tf.add(tf.matmul(A, W[-1]), b[-1])

layer_dims = [dim + label_dim*4 + 1] + p.gen_dims + [1]
L = len(layer_dims)
G_W = [tf.get_variable("G_W"+str(l), [layer_dims[l-1], layer_dims[l]], initializer=tf.keras.initializers.glorot_normal) for l in range(1, L)]
G_b = [tf.get_variable("G_b"+str(l), [1,layer_dims[l]], initializer=tf.zeros_initializer()) for l in range(1, L)]

layer_dims = [label_dim] + [label_dim*2] + [label_dim*4]
L = len(layer_dims)
G_W_y = [tf.get_variable("G_W_y"+str(l), [layer_dims[l-1], layer_dims[l]], initializer=tf.keras.initializers.glorot_normal) for l in range(1, L)]
G_b_y = [tf.get_variable("G_b_y"+str(l), [1,layer_dims[l]], initializer=tf.zeros_initializer()) for l in range(1, L)]


layer_dims = [dim + label_dim*4] + p.disc_dims + [1]
L = len(layer_dims)
D_W = [tf.get_variable("D_W"+str(l), [layer_dims[l-1], layer_dims[l]], initializer=tf.keras.initializers.glorot_normal) for l in range(1, L)]
D_b = [tf.get_variable("D_b"+str(l), [1,layer_dims[l]], initializer=tf.zeros_initializer()) for l in range(1, L)]

layer_dims = [label_dim] + [label_dim*2] + [label_dim*4]
L = len(layer_dims)
D_W_y = [tf.get_variable("D_W_y"+str(l), [layer_dims[l-1], layer_dims[l]], initializer=tf.keras.initializers.glorot_normal) for l in range(1, L)]
D_b_y = [tf.get_variable("D_b_y"+str(l), [1,layer_dims[l]], initializer=tf.zeros_initializer()) for l in range(1, L)]

def discriminator(X,y, W, b, W_y, b_y):
    y_embedding = feed_NN(y, W_y, b_y, act= tf.nn.relu)
    input = tf.concat([X, y_embedding], axis = 1)
    output = feed_NN(input,W,b, act= tf.nn.relu)
    return output

def gradient_penalty(Gen, GenLabel, Tar, TarLabel, W, b, W_y, b_y, L=1.0, batch_size = bs):
    if L == None:
        return 0
    zu = tf.random_uniform([batch_size,1], minval=0, maxval=1, dtype=tf.float32)
    D_interpolates = zu * Tar + (1 - zu) * Gen
    D_label_interpolates = zu * TarLabel + (1 - zu) * GenLabel
    D_disc_interpolates = discriminator(D_interpolates, D_label_interpolates, W, b, W_y, b_y)
    D_gradients = tf.gradients(D_disc_interpolates, [D_interpolates])[0]
    D_slopes = tf.reduce_sum(tf.square(D_gradients), reduction_indices=[1])
    D_gradient_penalty = tf.reduce_mean(tf.nn.relu(D_slopes-L**2))

    return D_gradient_penalty


def fun_U(X, y, t, W, b, W_y, b_y):
    y_embedding = feed_NN(y, W_y, b_y)
    inputs = tf.concat([X, y_embedding, t], axis = 1)
    U = feed_NN(inputs, W, b)
    return U

def running_cost(x, y, t, G_W, G_b, G_W_y, G_b_y):
    N_samples = tf.cast(tf.shape(x)[0], dtype=tf.float32)
    thisx = tf.stop_gradient(x)
    thist = t
    U = fun_U(thisx, y, thist, G_W, G_b, G_W_y, G_b_y)
    grad_U = tf.gradients(U,thisx)[0]
    lapl_U = 0
    if p.sigma > 0.:
        diag_hess_U = [tf.gradients(grad_U[:,j],thisx)[0][:,j] for j in range(x.dim)]
        lapl_U = sum(diag_hess_U)
    res = tf.gradients(U, thist)[0][:,0] + (1.0 -1.0/p.power)/alpha1 * tf.pow(tf.math.reduce_sum(tf.math.square(grad_U), axis = [1]), p.power/(p.power -1.0)/2.0) - 0.5* p.sigma**2 * lapl_U
    HJB = tf.reduce_mean(tf.abs(res))
    
    Lp = 1.0/p.power * tf.pow(tf.reduce_sum(tf.math.square(grad_U/alpha1), axis = [1]), p.power/2.0)
    Lp = tf.reduce_mean(Lp)
        
    return Lp, HJB

def generator(x, y, steps, dt, W, b, W_y, b_y, calc_reg = True):
    tt = tf.fill(tf.shape(x), 1.0)[:,:1]

    currentx = x
    x_shape = tf.shape(x)
    
    # running cost
    loss_OT = [0.0 for i in range(steps + 1)]
    loss_OT_Lp = [0.0 for i in range(steps + 1)]
    
    for i in range(steps):
        currentt = tt * i * dt
        currentU = fun_U(currentx, y, currentt, W, b, W_y, b_y)
        grad_U = tf.gradients(currentU, currentx)[0]
        
        currentv = -grad_U/alpha1
        if p.power > 2.0 or p.power < 2.0:
            currentv *= tf.pow(tf.norm(grad_U, axis = 1, keepdims = True), (2.0-p.power)/(p.power -1.0))
        currentx += dt * currentv
        if p.sigma>0:
            currentx += p.sigma* tf.sqrt(dt)*tf.random_normal(x_shape)
        
        # running cost
        if calc_reg == True:
            loss_OT_Lp[i], loss_OT[i] = running_cost(currentx, y, currentt, W, b, W_y, b_y)
        
    loss_OT = tf.reduce_sum(loss_OT) * dt
    loss_OT_Lp = tf.reduce_sum(loss_OT_Lp) * dt
        
    return currentx, loss_OT, loss_OT_Lp

# GAN loss

steps = int(T/dt)
Pdata = tf.placeholder(tf.float32, [None, dim])
Plabel = tf.placeholder(tf.float32, [None, label_dim])

Qdata = tf.placeholder(tf.float32, [None, dim])
Qlabel = tf.placeholder(tf.float32, [None, label_dim])

GPdata, loss_OT, loss_OT_L2 = generator(Pdata, Plabel, steps, dt, G_W, G_b, G_W_y, G_b_y)

fake_score = discriminator(GPdata,Plabel,D_W,D_b,D_W_y,D_b_y)
real_score = discriminator(Qdata,Qlabel,D_W,D_b,D_W_y,D_b_y)
GP = gradient_penalty(GPdata, Plabel, Qdata, Qlabel, D_W, D_b, D_W_y,D_b_y, L=p.L)

# discriminator loss
D_loss_1 = tf.reduce_mean(fake_score)
if p.f == 'W1':
    D_loss_2 = tf.reduce_mean(real_score)
elif p.f == 'reverseKL':
    D_loss_2 = -1 - tf.reduce_mean(tf.log(tf.nn.relu(-real_score)+1e-6))
else:
    D_loss_2 = tf.log(tf.reduce_mean(tf.exp(real_score)))
D_loss_intermediate = -D_loss_1 + D_loss_2
D_loss =  D_loss_intermediate + lamda * GP

# test discriminator loss
test_dt = dt/2
test_steps = int(T/test_dt)
test_GPdata, _, _ = generator(Pdata, Plabel, test_steps, test_dt, G_W, G_b, G_W_y, G_b_y, calc_reg=False)

test_fake_score = discriminator(test_GPdata, Plabel, D_W,D_b, D_W_y,D_b_y)
test_real_score = discriminator(Qdata, Qlabel, D_W,D_b, D_W_y,D_b_y)

D_loss_1_test = tf.reduce_mean(test_fake_score)
if p.f == 'W1':
    D_loss_2_test = tf.reduce_mean(test_real_score)
elif p.f == 'reverseKL':
    D_loss_2_test = -1 - tf.reduce_mean(tf.log(tf.nn.relu(-test_real_score)+1e-6))
else:
    D_loss_2_test = tf.log(tf.reduce_mean(tf.exp(test_real_score)))
D_loss_intermediate_test = -D_loss_1_test + D_loss_2_test


# Comparing potential and discriminator
x_T = tf.stop_gradient(GPdata)
U_T = fun_U(x_T, Plabel, tf.fill(tf.shape(x_T), T)[:,:1], G_W, G_b, G_W_y, G_b_y)
disc = discriminator(x_T, Plabel, D_W, D_b, D_W_y, D_b_y)
v1 = tf.gradients(disc, x_T)[0]
v2 = tf.gradients(U_T, x_T)[0]

N_samples_ = tf.cast(tf.shape(x_T)[0], dtype=tf.float32)
loss_terminal = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(v1-v2)))/N_samples_



if p.loss_case == 'OT':
    G_loss = D_loss_1 + alpha1 * loss_OT_L2
elif p.loss_case == 'No_OT':
    G_loss = D_loss_1
elif p.loss_case == 'PG':
    G_loss = D_loss_1 + alpha2 * loss_OT + alpha1 * loss_OT_L2
elif p.loss_case == 'OC':
    G_loss = D_loss_1 + alpha2 * loss_OT + alpha1 * loss_OT_L2 + alpha3 * loss_terminal


D_op = tf.train.AdamOptimizer(learning_rate=p.lr_disc, beta1=0.5, beta2=0.9).minimize(D_loss, var_list= D_W + D_b + D_W_y + D_b_y)
G_op = tf.train.AdamOptimizer(learning_rate=p.lr_gen, beta1=0.5, beta2=0.9).minimize(G_loss, var_list= G_W + G_b + G_W_y + G_b_y)

# configuration / initialization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# training
from plot_result import plot_2d, plot_orth_axes_saturation, plot_image

if p.savedir == None:
    savedir = f"assets/{p.example}/"
    savedir += "D1" if p.L != None else ""
    savedir += "P2" if p.loss_case == "OT" else ""
    savedir += "No_regul" if p.loss_case == "No_OT" and p.L == None else ""
    savedir += f"/Flow-GAN-Rep{Rep}/" #if p.submnfld_dim == None else f"/Flow-GAN-dim{p.total_dim}-Rep{p.Rep}/"
 
if not os.path.exists(savedir):
    os.makedirs(savedir)
saver = tf.train.Saver(max_to_keep=1000)

iterations = p.iterations + 1
save_iter = np.min((100, int(p.iterations/50)))
disc_iter = 5

loss_OT_evals, loss_OT_L2_evals, loss_terminal_evals, D_loss_evals = [], [], [], []
D_loss_test_evals = []
for iter in range(iterations):
    if p.example == 'Radius_conditioning':
        y = np.sqrt(np.random.uniform(0.0, 5.0**2, [10, 1]))
        y = np.repeat(y, int(bs/10), axis = 0)
    Q = PosteriorDataGenerator(len(y), p.example, y)
    P = PriorDataGenerator(bs, p.example)
    for i in range(disc_iter):
        thisP, thisPlabel = P.nextbatch(bs)
        thisQ, thisQlabel = Q.nextbatch(bs)
        sess.run(D_op, feed_dict= {Pdata: thisP, Plabel: thisPlabel, Qdata: thisQ, Qlabel: thisQlabel})

    thisP, thisPlabel = P.nextbatch(bs)
    sess.run(G_op, feed_dict= {Pdata: thisP, Plabel: thisPlabel})
    
    if iter % 500 ==0:
        save_path = saver.save(sess, savedir+"/" + str(iter) + ".ckpt")
    if iter % save_iter ==0:
        thisP, thisPlabel = P.nextbatch(bs)
        thisQ, thisQlabel = Q.nextbatch(bs)
        out = sess.run(GPdata, feed_dict={Pdata: thisP, Plabel: thisPlabel})
        
        # print losses
        loss_OT_evals.append(sess.run(loss_OT, feed_dict= {Pdata: thisP, Plabel: thisPlabel, Qdata: thisQ, Qlabel: thisQlabel}))
        loss_OT_L2_evals.append(sess.run(loss_OT_L2, feed_dict= {Pdata: thisP, Plabel: thisPlabel, Qdata: thisQ, Qlabel: thisQlabel}))
        loss_terminal_evals.append(sess.run(loss_terminal, feed_dict= {Pdata: thisP, Plabel: thisPlabel, Qdata: thisQ, Qlabel: thisQlabel}))
        D_loss_evals.append(np.abs(-sess.run(D_loss_intermediate, feed_dict= {Pdata: thisP, Plabel: thisPlabel, Qdata: thisQ, Qlabel: thisQlabel})))
        print("iter %06d : divergence %.4f, kinetic energy %.4f, HJB residual %.4f, terminal loss %.4f" % (iter, D_loss_evals[-1], loss_OT_L2_evals[-1], loss_OT_evals[-1], loss_terminal_evals[-1]))
        
        

        # plot
        plot_count = bs
        for r in [0.2, 0.5, 0.99]:
            thisP, thisPlabel = P.nextbatch(plot_count)
            thisPlabel = 5 * r * np.ones([plot_count, 1], dtype=np.float32)
            out = sess.run(GPdata, feed_dict={Pdata: thisP, Plabel: thisPlabel})
            plt.scatter(out[:,0],out[:,1], marker = 'o', label = f'r={5*r}', s = 2)
        plt.legend()
        plt.savefig(savedir+"/" + str(iter) +'.png', dpi=50)
        plt.clf()
        
        
        # evaluate on test data set
        thisP, thisPlabel = PriorDataGenerator(1000, p.example).nextbatch(1000)
        thisQ, thisQlabel = Q.nextbatch(1000)
        out = sess.run(test_GPdata, feed_dict={Pdata: thisP, Plabel: thisPlabel})
        
        plt.scatter(thisQlabel[:,0], sess.run(real_score, feed_dict={Qdata: thisQ, Qlabel: thisQlabel})[:,0])
        plt.savefig(savedir+"/" + str(iter)+'_scatter.png', dpi=50)
        plt.clf()
        
        # print losses
        D_loss_test_evals.append(np.abs(-sess.run(D_loss_intermediate_test, feed_dict= {Pdata: thisP, Plabel: thisPlabel, Qdata: thisQ, Qlabel: thisQlabel})))
        print("iter %06d : divergence %.4f on test data" % (iter, D_loss_test_evals[-1]))


# Plot losses
fig = plt.figure()
plt.semilogy(range(0, iterations, save_iter), D_loss_evals, label='Terminal cost')
plt.semilogy(range(0, iterations, save_iter), loss_OT_L2_evals, label='Kinetic energy')
#plt.semilogy(range(0, iterations, save_iter), loss_OT_evals, label='HJB residual(squared)')
#plt.semilogy(range(0, iterations, save_iter), loss_terminal_evals, label='Terminal condition residual')
plt.semilogy(range(0, iterations, save_iter), D_loss_test_evals, label='Terminal cost - test')


#plt.ylim([1e-4, 1e+2])
plt.xlabel('Iterations for the generator update', fontsize=17)
plt.legend(fontsize=17)
plt.savefig(savedir + '/costs.png', dpi=50)
plt.clf()
plt.close()

print("Terminal cost: ", D_loss_evals[-1], ", Kinetic energy: ", loss_OT_L2_evals[-1])


G_W = [sess.run(w) for w in G_W]
G_b = [sess.run(b) for b in G_b]
G_W_y = [sess.run(w) for w in G_W_y]
G_b_y = [sess.run(b) for b in G_b_y]
D_W = [sess.run(w) for w in D_W]
D_b = [sess.run(b) for b in D_b]
D_W_y = [sess.run(w) for w in D_W_y]
D_b_y = [sess.run(b) for b in D_b_y]
with open(savedir+"/result.pickle", "wb") as fw:
    pickle.dump([G_W, G_b, G_W_y, G_b_y, D_W, D_b, D_W_y, D_b_y, \
    T, dt, \
    D_loss_evals, loss_OT_L2_evals, loss_OT_evals, loss_terminal_evals, D_loss_test_evals, \
    alpha1, alpha2, alpha3], fw)





