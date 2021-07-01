import numpy as np
from .train_detector import cnn_model_struct
from .config import *
import tensorflow as tf
import tqdm, gzip, cProfile, time, argparse, pickle, os
# just to prevent tensorflow from printing logs
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
tf.logging.set_verbosity(tf.logging.ERROR)

class Infer:
	def __init__(self, config):
		tf.reset_default_graph()
		self.cfg = config
		self.target = []
		self.inp = tf.placeholder(tf.float32, self.cfg.test_param_dims)
		
		# AF ADD
		self.inp_flex = tf.placeholder(tf.float32, self.cfg.param_dims)

		self.initialized = False

		#with tf.device('/gpu:0'):
		with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope:
			self.model = cnn_model_struct()
			# self.model.build(self.inp, self.cfg.test_param_dims[1:], self.cfg.output_hist_dims[1:], train_mode=False, verbose=False)
			# AF: ADD changed self.inp --> self.inp_flex
			self.model.build(self.inp_flex, self.cfg.test_param_dims[1:], self.cfg.output_hist_dims[1:], train_mode=False, verbose=False)

			#	self.gpuconfig = tf.ConfigProto()
			#	self.gpuconfig.gpu_options.allow_growth = True
			#	self.gpuconfig.allow_soft_placement = True
		self.saver = tf.train.Saver()
		self.sess = tf.Session()
		print(self.cfg.model_output)
		ckpts = tf.train.latest_checkpoint(self.cfg.model_output)
		self.saver.restore(self.sess, ckpts)

	def __getitem__(self, item):
		return getattr(self, item)

	def __contains__(self, item):
		return hasattr(self, item)

	def forward(self, params):
		pred_hist = self.sess.run(self.model.output, feed_dict = {self.inp:params.reshape(self.cfg.test_param_dims)})
		return pred_hist
	
	# AF: ADD
	def forward_flex(self, params):
		pred_hist = self.sess.run(self.model.output, feed_dict = {self.inp_flex:params.reshape(self.cfg.flex_param_dims)})
		return pred_hist	

def load_cnn(model, nbin):
	cfg = Config(model = model, bins = nbin)
	inference_class = Infer(config = cfg)
	#return inference_class.forward
	return inference_class.forward_flex


# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--model', type=str)
# 	parser.add_argument('--nbin', type=int)
# 	args = parser.parse_args()

# 	cfg = Config(model=args.model, bins=args.nbin)
# 	inference_class = Infer(config=cfg)

# 	example_params = np.array([0., 1.5, 0.5, 1])
# 	print(inference_class.forward(example_params))
