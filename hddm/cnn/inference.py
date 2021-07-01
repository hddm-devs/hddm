from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os, pickle
import tensorflow as tf
from train_detector import cnn_model_struct 
import config
from scipy.optimize import differential_evolution
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool
import pickle
from functools import partial

plt.rcParams['font.size']= 6
# just to prevent tensorflow from printing logs
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
tf.logging.set_verbosity(tf.logging.ERROR)

class Infer:
    def __init__(self, config):
	self.cfg = config
	#self.target = pickle.load(open(self.cfg.inference_dataset,'rb'))[0][0].reshape((-1,))
	self.target = []
	self.inp = tf.placeholder(tf.float32, self.cfg.test_param_dims)
	self.initialized = False
	with tf.device('/gpu:0'):
	    with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope:
		self.model = cnn_model_struct()
		self.model.build(self.inp, self.cfg.test_param_dims[1:], self.cfg.output_hist_dims[1:], train_mode=False, verbose=False)
	    self.gpuconfig = tf.ConfigProto()
	    self.gpuconfig.gpu_options.allow_growth = True
	    self.gpuconfig.allow_soft_placement = True
	    self.saver = tf.train.Saver()

    def __getitem__(self, item):
	return getattr(self, item)

    def __contains__(self, item):
	return hasattr(self, item)

    def klDivergence(self, x, y, eps1=1e-7, eps2=1e-30):
	return np.sum(x * np.log(eps2 + x/(y+eps1)))

    def likelihood(self, x, y, eps=1e-7):
	return np.sum(-np.log(x+eps)*y)

    def objectivefn(self, params):
	if self.initialized == False:
	    self.sess = tf.Session(config=self.gpuconfig)
	    ckpts = tf.train.latest_checkpoint(self.cfg.model_output)
	    self.saver.restore(self.sess, ckpts)
	    self.initialized = True
	pred_hist = self.sess.run(self.model.output, feed_dict={self.inp:params.reshape(self.cfg.test_param_dims)})
	#return self.klDivergence(pred_hist, self.target)
	return self.likelihood(pred_hist, self.target)

def model_inference(args):
    simdata = args['data']
    inf_cfg = config.Config(model=args['model'], bins=args['nbin'], N=args['N'])
    inference_class = Infer(config=inf_cfg)
    bounds = inf_cfg.bounds
    inference_class.target = simdata.reshape((-1,))
    output = differential_evolution(inference_class.objectivefn,bounds)
    inference_class.sess.close()
    return output.x

def run(workers, model, nbin, N):
    my_config = config.Config(model=model, bins=nbin, N=N)
    for infdata in my_config.inference_dataset:
        simulated_data = pickle.load(open(infdata,'rb'))
        simulated_data = simulated_data[1].reshape((-1,simulated_data[1].shape[2],simulated_data[1].shape[3]))
        nsamples = simulated_data.shape[0]    
        rec_params = []
	tuples = [{'data':x,'model':model, 'nbin':nbin, 'N':N} for x in simulated_data]
        for _ in tqdm.tqdm(workers.imap(model_inference, tuples), total=nsamples):
    	    rec_params.append(_)
            pass
        # plot the results
        rec_params = np.array(rec_params)
        GT = np.tile(pickle.load(open(infdata,'rb'))[0][:nsamples],(10,1))
	result = {'pred':rec_params, 'gt':GT}
	pickle.dump(result,open(os.path.join(cfg.results_dir,infdata.split('/')[-1].split('.')[0]+'_recovery.p'),'wb'))
	
        cmap = mpl.cm.get_cmap('Paired')
        nplots = GT.shape[1]
        fig, ax = plt.subplots(int(np.ceil(nplots/3.)), 3)
        for k in range(GT.shape[1]):
    	    ax[int(k/3),(k%3)].scatter(GT[:,k],rec_params[:,k],5,alpha=0.5)
	    ax[int(k/3),(k%3)].set_title('Parameter {}'.format(k+1),fontsize=6)
	    ax[int(k/3),(k%3)].set_xlabel('true parameters')
	    ax[int(k/3),(k%3)].set_ylabel('recovered parameters')
        plt.savefig(os.path.join(cfg.results_dir,infdata.split('/')[-1].split('.')[0]+'_recovery.png'), dpi=300, bbox_inches='tight')
        plt.close()
	
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--nbin', type=int)
    parser.add_argument('--N', type=int)
    args = parser.parse_args()

    cfg = config.Config()
    n_workers = 20
    workers = Pool(n_workers)

    run(workers, args.model, args.nbin, args.N)
    #models = ['ddm', 'angle', 'weibull', 'ornstein', 'fullddm']
    #nbins = [256, 512]
    #Ns = [128, 1024, 8192]
