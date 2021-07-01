import os
import numpy as np
import glob
import hddm

# model_checkpoint_path: "/Users/afengler/OneDrive/git_repos/hddm/hddm/cnn_models/ddm_training_data_binned_1_nbins_512_n_100000/ddm_210500.ckpt-210500"
# all_model_checkpoint_paths: "/Users/afengler/OneDrive/git_repos/hddm/hddm/cnn_models/ddm_training_data_binned_1_nbins_512_n_100000/ddm_209500.ckpt-209500"
# all_model_checkpoint_paths: "/Users/afengler/OneDrive/git_repos/hddm/hddm/cnn_models/ddm_training_data_binned_1_nbins_512_n_100000/ddm_209750.ckpt-209750"
# all_model_checkpoint_paths: "/Users/afengler/OneDrive/git_repos/hddm/hddm/cnn_models/ddm_training_data_binned_1_nbins_512_n_100000/ddm_210000.ckpt-210000"
# all_model_checkpoint_paths: "/Users/afengler/OneDrive/git_repos/hddm/hddm/cnn_models/ddm_training_data_binned_1_nbins_512_n_100000/ddm_210250.ckpt-210250"
# all_model_checkpoint_paths: "/Users/afengler/OneDrive/git_repos/hddm/hddm/cnn_models/ddm_training_data_binned_1_nbins_512_n_100000/ddm_210500.ckpt-210500"

# model_checkpoint_path: "./ddm_210500.ckpt-210500"
# all_model_checkpoint_paths: "./ddm_209500.ckpt-209500"
# all_model_checkpoint_paths: "./ddm_209750.ckpt-209750"
# all_model_checkpoint_paths: "./ddm_210000.ckpt-210000"
# all_model_checkpoint_paths: "./ddm_210250.ckpt-210250"
# all_model_checkpoint_paths: "./ddm_210500.ckpt-210500"


# ddm_rel_checkpoint_dict = dict('model_checkpoint_path:': "/ddm_210500.ckpt-210500")

# for key, val in ddm_rel_checkpoint_dict:
#     ddm_rel_checkpint_dict[key] = hddm.__path__[0] + ddm_rel_checkpint_dict[val]

class Config(object):

    def __init__(self, model=None, bins=None, N=None):
        # Directory setup
        self.base_dir = hddm.__path__[0]  #+ '/Users/afengler/OneDrive/git_repos/hddm/hddm/'
        self.data_dir = 'data'
        self.tfrecord_dir = 'tfrecords'
        self.summary_dir = 'summaries'

        # training dataset features
        self.isBinned = True
        if model is None:
            model = 'ddm_sdv'
        if bins != None:
            self.nBins = bins
        else:
            self.nBins = 512
        self.nDatapoints = 100000

        if N != None:
            self.N = N
        else:
            self.N = 1024

        self.method_options = {'ddm': self.ddm_initialize, 'angle': self.angle_initialize, 'weibull_cdf': self.weibull_initialize, 'ornstein': self.ornstein_initialize,
                               'full_ddm': self.full_ddm_initialize, 'race_model_3': self.race_model_3_initialize, 'race_model_4': self.race_model_4_initialize,'race_model_6':self.race_model_6_initialize,
                               'lca_3': self.lca_3_initialize, 'lca_4': self.lca_4_initialize,
                               'ddm_seq2': self.ddm_seq2_initialize,
                               'ddm_par2': self.ddm_par2_initialize,
                               'ddm_mic2': self.ddm_mic2_initialize,
                               'levy': self.levy_initialize,
                               'full_ddm2': self.full_ddm2_initialize,
                               'ddm_sdv': self.ddm_sdv_initialize}

        # select model
        self.method_options[model](self.nBins)

        # select dataset
        param_recovery_folder = os.path.join('../data/{}'.format(self.model_name),
                                        'parameter_recovery_data_binned_{}_nbins_{}_n_{}'.format(int(self.isBinned), self.nBins, self.N))

        self.inference_dataset = glob.glob('{}/*'.format(param_recovery_folder))
        
        self.dataset_dir = os.path.join(self.model_name, 'training_data_binned_{}_nbins_{}_n_{}'.format(int(self.isBinned),self.nBins,self.nDatapoints))
        self.refname = self.dataset_dir.replace('/','_')

        self.train_tfrecords = self.refname+'_train.tfrecords'
        self.val_tfrecords = self.refname+'_val.tfrecords'
        self.test_tfrecords = self.refname+'_test.tfrecords'

        """
        Specify whether or not to treat gaussian errors as isotropic
        This only matters for the "reverse" model
        """
        self.full_cov_matrix = False
        self.model_suffix = 'full_cov' if self.full_cov_matrix else 'isotropic'

        # Data configuration
        self.results_dir = '/media/data_cifs/projects/prj_approx-bayes/projectABC/results/'
        self.model_output = os.path.join(self.base_dir,
                                        'cnn_models',
                                         self.refname)
        
        print(self.model_output)
        
        with open(os.path.join(self.model_output, 'checkpoint'), 'r') as f:
            ckpt_meta = f.readlines()
        
        ckpt_meta = [x.strip().split(':')[0] + ': "' + os.path.join(hddm.__path__[0],'cnn_models', self.refname, x.strip().split(':')[-1].split('/')[-1]) + "\n" for x in ckpt_meta]
        
        # For simple local check in dev environ
        # ckpt_meta = [x.strip().split(':')[0] + ': "' + os.path.join('hddm/','cnn_models', 'ddm_training_data_binned_1_nbins_512_n_100000/', x.strip().split(':')[-1].split('/')[-1]) + "\n" for x in ckpt_meta]

        with open(os.path.join(self.model_output, 'checkpoint'), 'w') as f:
            f.writelines(ckpt_meta)
        
        self.data_prop = {'train':0.9, 'val':0.05, 'test':0.05}
        self.min_param_values = np.array([x[0] for x in self.bounds])
        self.param_range = np.array([x[1] - x[0] for x in self.bounds])

        # Model hyperparameters
        self.epochs = 50
        self.train_batch = 128
        self.val_batch = 64
        self.test_batch = 128
        # how often should the training stats be printed?
        self.print_iters = 250
        # how often do you want to validate?
        self.val_iters = 1000

    def angle_initialize(self, nbins):
        self.dataset = 'angle_nchoices*'
        self.model_name = 'angle'
        self.param_dims = [None, 1, 5, 1]
        self.test_param_dims = [1, 1, 5, 1]
        self.output_hist_dims = [None, 1, nbins, 2]
        self.bounds = [(-2.5, 2.5), (0.2, 2.0), (0.1, 0.9), (0.0, 2.0), (0, (np.pi / 2 - .2))]

    def ddm_initialize(self, nbins):
        self.dataset = 'ddm_nchoices*'
        self.model_name = 'ddm'
        self.param_dims = [None, 1, 4, 1]
        self.test_param_dims = [1, 1, 4, 1]
        self.flex_param_dims = [-1, 1, 4, 1]
        self.output_hist_dims = [None, 1, nbins, 2]
        self.bounds = [(-2.5, 2.5), (0.5, 2.2), (0.25, 0.75), (0.05, 1.95)] # done
        self.param_names = ['v', 'a', 'w', 'ndt']

    def weibull_initialize(self, nbins):
        self.model_name = 'weibull'
        self.dataset = 'weibull_cdf_nchoices*'
        self.param_dims = [None, 1, 6, 1]
        self.test_param_dims = [1, 1, 6, 1]
        self.output_hist_dims = [None, 1, nbins, 2]
        self.bounds = [(-2.5, 2.5), (0.2, 2.0), (0.1, 0.9), (0.0, 2.0), (0.5, 5.0), (0.5, 7.0)]

    def full_ddm_initialize(self, nbins):
        self.dataset = 'full_ddm_nchoices*'
        self.model_name = 'full_ddm'
        self.param_dims = [None, 1, 7, 1]
        self.test_param_dims = [1, 1, 7, 1]
        self.output_hist_dims = [None, 1, nbins, 2]
        self.bounds = [(-2.5, 2.5), (0.2, 2.0), (0.1, 0.9), (0.25, 2.5), (0, 0.4), (0, 1), (0.0, 0.5)]

    def ornstein_initialize(self, nbins):
        self.model_name = 'ornstein'
        self.dataset = 'ornstein_nchoices*'
        self.param_dims = [None, 1, 5, 1]
        self.test_param_dims = [1, 1, 5, 1]
        self.output_hist_dims = [None, 1, nbins, 2]
        self.bounds = [(-2.5, 2.5), (0.2, 2.0), (0.1, 0.9), (-1.0, 1.0), (0.0, 2.0)]

    def ddm_seq2_initialize(self, nbins):
        self.model_name = 'ddm_seq2'
        self.dataset = 'ddm_seq2_nchoices*'
        self.param_dims = [None, 1, 8, 1]
        self.test_param_dims = [1, 1, 8, 1]
        self.output_hist_dims = [None, 1, nbins, 4]
        self.bounds = [(-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5),(0.2, 2), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.0, 2.0)]

    def ddm_par2_initialize(self, nbins):
        self.model_name = 'ddm_par2'
        self.dataset = 'ddm_par2_nchoices*'
        self.param_dims = [None, 1, 8, 1]
        self.test_param_dims = [1, 1, 8, 1]
        self.output_hist_dims = [None, 1, nbins, 4]
        self.bounds = [(-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5),(0.2, 2), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.0, 2.0)]

    def ddm_mic2_initialize(self, nbins):
        self.model_name = 'ddm_mic2'
        self.dataset = 'ddm_mic2_nchoices*'
        self.param_dims = [None, 1, 9, 1]
        self.test_param_dims = [1, 1, 9, 1]
        self.output_hist_dims = [None, 1, nbins, 4]
        self.bounds = [(-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5),(0.2, 2), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.0, 1.0), (0.0, 2.0)]

    def race_model_3_initialize(self, nbins):
        self.model_name = 'race_model_3'
        #self.dataset_dir = 'race_model_3'
        self.dataset = 'race_model_nchoices*'
        self.param_dims = [None, 1, 8, 1]
        self.test_param_dims = [1, 1, 8, 1]
        self.output_hist_dims = [None, 1, nbins, 3]
        self.bounds = [(0.0, 2.5), (0.0, 2.5), (0.0, 2.5), (1.0, 3.0), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.0, 2.0)]

    def lca_3_initialize(self, nbins):
        self.model_name = 'lca_3'
        #self.dataset_dir = 'lca_3'
        self.dataset = 'lca_nchoices*'
        self.param_dims = [None, 1, 10, 1]
        self.test_param_dims = [1, 1, 10, 1]
        self.output_hist_dims = [None, 1, nbins, 3]
        self.bounds = [(0, 2.5), (0., 2.5), (0., 2.5), (1.0, 3.0), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)]

    def race_model_4_initialize(self, nbins):
        self.model_name = 'race_model_4'
        #self.dataset_dir = 'race_model_4'
        self.dataset = 'race_model_nchoices*'
        self.param_dims = [None, 1, 10, 1]
        self.test_param_dims = [1, 1, 10, 1]
        self.output_hist_dims = [None, 1, nbins, 4]
        self.bounds = [(0.0, 2.5), (0.0, 2.5), (0.0, 2.5), (0.0, 2.5), (1.0, 3.0), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.0, 2.0)]

    def lca_4_initialize(self, nbins):
        self.model_name = 'lca_4'
        #self.dataset_dir = 'lca_4'
        self.dataset = 'lca_nchoices*'
        self.param_dims = [None, 1, 12, 1]
        self.test_param_dims = [1, 1, 12, 1]
        self.output_hist_dims = [None, 1, nbins, 4]
        self.bounds = [(0, 2.5), (0., 2.5), (0., 2.5), (0., 2.5), (1.0, 3.0), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)]

    def race_model_5_initialize(self):
        self.model_name = 'race_model_5'
        self.dataset_dir = 'race_model_5'
        self.dataset = 'race_model_base*'
        self.param_dims = [None, 1, 12, 1]
        self.test_param_dims = [1, 1, 12, 1]
        self.output_hist_dims = [None, 1, 256, 5]

    def lca_5_initialize(self):
        self.model_name = 'lca_5'
        self.dataset_dir = 'lca_5'
        self.dataset = 'lca_base*'
        self.param_dims = [None, 1, 14, 1]
        self.test_param_dims = [1, 1, 14, 1]
        self.output_hist_dims = [None, 1, 256, 5]

    def race_model_6_initialize(self, nbins):
        self.model_name = 'race_model_6'
        #self.dataset_dir = 'race_model_3'
        self.dataset = 'race_model_nchoices*'
        self.param_dims = [None, 1, 14, 1]
        self.test_param_dims = [1, 1, 14, 1]
        self.output_hist_dims = [None, 1, nbins, 6]
        self.bounds = [(0.0, 2.5), (0.0, 2.5), (0.0, 2.5), (0.0, 2.5), (0.0, 2.5), (0.0, 2.5), (1.0, 3.0), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.1, 0.9), (0.0, 2.0)]

    def lca_6_initialize(self):
        self.model_name = 'lca_6'
        self.dataset_dir = 'lca_6'
        self.dataset = 'lca_base*'
        self.param_dims = [None, 1, 16, 1]
        self.test_param_dims = [1, 1, 16, 1]
        self.output_hist_dims = [None, 1, 256, 6]

    def levy_initialize(self, nbins):
        self.model_name = 'levy'
        #self.dataset_dir = 'race_model_3'
        self.dataset = 'levy_nchoices*'
        self.param_dims = [None, 1, 5, 1]
        self.test_param_dims = [1, 1, 5, 1]
        self.output_hist_dims = [None, 1, nbins, 2]
        self.bounds = [(-2.5, 2.5), (0.2, 2), (0.1, 0.9), (1.0, 2.0), (0.0, 2.0)]

    def full_ddm2_initialize(self, nbins):
        self.model_name = 'full_ddm2'
        self.dataset = 'full_ddm2_nchoices*'
        self.param_dims = [None, 1, 7, 1]
        self.test_param_dims = [1, 1, 7, 1]
        self.output_hist_dims = [None, 1, nbins, 2]
        self.bounds = [(-3, 3), (0.3, 2.0), (0.3, 0.7), (0.25, 2.5), (0.0, 0.5), (0.0, 2.0), (0.05, 0.2)]

    def ddm_sdv_initialize(self, nbins):
        self.model_name = 'ddm_sdv'
        self.dataset = 'ddm_sdv_nchoices*'
        self.param_dims = [None, 1, 5, 1]
        self.test_param_dims = [1, 1, 5, 1]
        self.output_hist_dims = [None, 1, nbins, 2]
        self.bounds = [(-3, 3), (0.3, 2.5), (0.1, 0.9), (0.0, 2.0), (0.0, 2.5)]
