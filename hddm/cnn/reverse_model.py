from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, sys, tempfile, os, glob, pickle, tqdm, math, time
import tensorflow as tf
import config
from tf_data_handler import inputs
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

class cnn_reverse_model:
    def __init__(self, trainable=True):
        self.trainable = trainable
        self.data_dict = None
        self.var_dict = {}

    def __getitem__(self, item):
        return getattr(self,item)

    def __contains__(self, item):
        return hasattr(self, item)

    def get_size(self, input_data):
        return np.prod([int(x) for x in input_data.get_shape()[1:]])

    def build(self, input_data, input_shape, output_shape, train_mode=None, verbose=True, full_cov=False):
        if verbose:
            print ("Building the network...")
        network_input = tf.identity(input_data, name='input')
        with tf.name_scope('reshape'):
            x_data = tf.reshape(network_input, [-1, input_shape[0], input_shape[1], input_shape[2]])
 
        # conv layer 1
        with tf.variable_scope('conv1'):
            self.W_conv1 = self.weight_variable([1, 5, input_shape[2], 8],var_name='wconv1')
            self.b_conv1 = self.bias_variable([8],var_name='bconv1')
            #self.norm1 = tf.layers.batch_normalization(self.conv2d(x_data, self.W_conv1,stride=[1,1,1,1]) + self.b_conv1,scale=True,center=True,training=train_mode,name='batchnorm1')
            self.norm1 = self.conv2d(x_data, self.W_conv1,stride=[1,1,1,1]) + self.b_conv1
            self.h_conv1 = tf.nn.leaky_relu(self.norm1, alpha=0.1)
        if verbose:
            print(self.h_conv1.get_shape())

        # conv layer 2
        with tf.variable_scope('conv2'):
            self.W_conv2 = self.weight_variable([1, 5, 8, 16],var_name='wconv2')
            self.b_conv2 = self.bias_variable([16],var_name='bconv2')
            #self.norm2 = tf.layers.batch_normalization(self.conv2d(self.h_conv1, self.W_conv2, stride=[1, 2, 2, 1]) + self.b_conv2,scale=True,center=True,training=train_mode,name='batchnorm2')
            self.norm2 = self.conv2d(self.h_conv1, self.W_conv2, stride=[1, 2, 2, 1]) + self.b_conv2
            self.h_conv2 = tf.nn.leaky_relu(self.norm2, alpha=0.1)
        if verbose:
            print(self.h_conv2.get_shape())

        # conv layer 3
        with tf.variable_scope('conv3'):
            self.W_conv3 = self.weight_variable([1, 5, 16, 32],var_name='wconv3')
            self.b_conv3 = self.bias_variable([32],var_name='bconv3')
            #self.norm3 = tf.layers.batch_normalization(self.conv2d(self.h_conv2, self.W_conv3, stride=[1, 2, 2, 1]) + self.b_conv3, scale=True,center=True,training=train_mode,name='batchnorm3')
            self.norm3 = self.conv2d(self.h_conv2, self.W_conv3, stride=[1, 2, 2, 1]) + self.b_conv3
            self.h_conv3 = tf.nn.leaky_relu(self.norm3,alpha=0.1)
        if verbose:
            print(self.h_conv3.get_shape())

        self.fc1 = self.fc_layer(self.h_conv3, self.get_size(self.h_conv3), 256, 'fc1')
        if verbose:
            print(self.fc1.get_shape())

        #self.fc2 = self.fc_layer(self.fc1, self.get_size(self.fc1), 512, 'fc2')
        #if verbose:
        #    print(self.fc2.get_shape())

        self.fc2 = self.fc_layer(self.fc1, self.get_size(self.fc1), 128, 'fc2')
        if verbose:
            print(self.fc2.get_shape())

        #self.fc4 = self.fc_layer(self.fc3, self.get_size(self.fc3), 64, 'fc4')
        #if verbose:
        #    print(self.fc4.get_shape())

        nparams = np.prod(output_shape)
        if full_cov:
            self.final_layer = self.fc_layer(self.fc2, self.get_size(self.fc2), nparams + nparams ** 2, 'final_layer')
        else:
            self.final_layer = self.fc_layer(self.fc2, self.get_size(self.fc2), nparams * 2, 'final_layer')
            self.final_layer = tf.concat([self.final_layer[:, :nparams], tf.nn.softplus(self.final_layer[:, nparams:])], 1)

        self.output = tf.identity(self.final_layer,name='output')
        if verbose:
            print(self.output.get_shape())

    def conv2d(self, x, W, stride=[1,1,1,1]):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def max_pool_2x2_1(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 1, 1, 1], padding='SAME')

    def weight_variable(self, shape, var_name):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.get_variable(name=var_name,initializer=initial)

    def bias_variable(self, shape, var_name):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.001, shape=shape)
        return tf.get_variable(name=var_name,initializer=initial)

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(self, in_size, out_size, name, init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [in_size, out_size],
                tf.contrib.layers.xavier_initializer(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [in_size, out_size], 0.0, 0.001)
        bias_init = tf.truncated_normal([out_size], .0, .001)
        weights = self.get_var(weight_init, name, 0, name + "_weights")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return weights, biases

    def get_var(
            self, initial_value, name, idx,
            var_name, in_size=None, out_size=None):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # get_variable, change the boolean to numpy
            if type(value) is list:
                var = tf.get_variable(
                    name=var_name, shape=value[0], initializer=value[1])
            else:
                var = tf.get_variable(name=var_name, initializer=value)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)
            #var = tf.get_variable(name=var_name, initializer=value)

        self.var_dict[(name, idx)] = var

        return var

'''
# assumes isotropicity
'''
def heteroskedastic_loss(p, q, nparams):
    param_est = p[:,:nparams]
    var = p[:, nparams:]
    diff_tensor = (param_est - q) ** 2
    return tf.reduce_sum( diff_tensor/var + tf.log(var) )

'''
# train the full covariance matrix instead
'''
def heteroskedastic_cov_loss(p, q, nparams, eps=10):
    param_est = p[:, :nparams]
    # reshape to a matrix
    cov = tf.nn.softplus(tf.reshape(p[:,nparams:],[-1,nparams,nparams]))
    # extract the upper triangular matrix
    cov_upper = tf.matrix_band_part(cov, 0, 0)
    # enforce symmetry
    cov_sym = 0.5 * (cov_upper + tf.linalg.transpose(cov_upper))
    # determinant of covariance matrix
    cov_det = tf.linalg.det(cov_sym)
    # inverse of the covariance matrix
    cov_inv = tf.linalg.inv(cov_sym)
    # eigen values
    cov_eig = tf.linalg.eigvalsh(cov_sym)

    # diff
    diff = tf.expand_dims(param_est - q, axis = -1)
    term1 = tf.squeeze(tf.matmul(tf.matmul(tf.linalg.transpose(diff), cov_inv), diff))
    loss = tf.reduce_sum( term1 + tf.log(1e-30 + tf.abs(cov_det)) - eps * tf.minimum(tf.reduce_min(cov_eig, axis=-1), 0))
    #loss = tf.reduce_sum( term1 + tf.log(1e-30 + tf.abs(cov_det)) )
    return loss, cov_sym

def train_reverse_model(config):

    train_files = os.path.join(
                        config.base_dir,
                        config.tfrecord_dir,
                        config.train_tfrecords)
    val_files = os.path.join(
                        config.base_dir,
                        config.tfrecord_dir,
                        config.val_tfrecords)

    with tf.device('/cpu:0'): 
        train_labels, train_data = inputs(
                                        tfrecord_file=train_files,
                                        num_epochs=config.epochs,
                                        batch_size=config.train_batch,
                                        target_data_dims=config.param_dims,
                                        target_label_dims=config.output_hist_dims)
        val_labels, val_data = inputs(
                                        tfrecord_file=val_files,
                                        num_epochs=config.epochs,
                                        batch_size=config.val_batch,
                                        target_data_dims=config.param_dims,
                                        target_label_dims=config.output_hist_dims)
    with tf.device('/gpu:0'):
        with tf.variable_scope("reversemodel") as scope:
            print ("creating the model")
            model = cnn_reverse_model()
            model.build(train_data, config.output_hist_dims[1:], config.param_dims[1:], train_mode=True, full_cov=config.full_cov_matrix)
            y_conv = model.output
            nparams = np.prod(config.param_dims[1:])

            # Define loss and optimizer
            with tf.name_scope('loss'):
                labels = tf.reshape(train_labels, [-1, nparams])

                #### depending on the config, use the appropriate loss
                if config.full_cov_matrix:
                    hke_loss, cov_sym = heteroskedastic_cov_loss(y_conv, labels, nparams)
                else:
                    hke_loss = heteroskedastic_loss(y_conv, labels, nparams)

            with tf.name_scope('adam_optimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_step = tf.train.AdamOptimizer(1e-4).minimize(hke_loss)

            #####
            ## VALIDATION
            #####
            print("building a validation model")
            scope.reuse_variables()
            val_model = cnn_reverse_model()
            val_model.build(val_data, config.output_hist_dims[1:], config.param_dims[1:], train_mode=False, full_cov=config.full_cov_matrix)
            val_res = val_model.output
            norm_val_labels = tf.reshape(val_labels, [-1,nparams])
            
            #### select loss function for the val model as well
            if config.full_cov_matrix:
                val_loss, _ = heteroskedastic_cov_loss(val_res, norm_val_labels, nparams)
            else:
                val_loss =  heteroskedastic_loss(val_res, norm_val_labels, nparams)

            tf.summary.scalar("loss", hke_loss)
            summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables())

    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True
    gpuconfig.allow_soft_placement = True

    with tf.Session(config=gpuconfig) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(config.base_dir,config.summary_dir,config.model_name))
        train_writer.add_graph(tf.get_default_graph())

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step = 0
        start = time.time()
        try:
            while not coord.should_stop():
                # train for a step
                if config.full_cov_matrix:
                        _, loss, outputs, tr_data, tr_labels, norm_tr_labels, cov_mat = sess.run([train_step, hke_loss, y_conv, train_data, train_labels, labels, cov_sym])
                else:
                        _, loss, outputs, tr_data, tr_labels, norm_tr_labels = sess.run([train_step, hke_loss, y_conv, train_data, train_labels, labels])

                step+=1
                if step % config.print_iters == 0:
                    finish = time.time()
                    print("step={}, loss={}, time_elapsed={} s/step".format(step,loss,(finish-start)/float(config.print_iters)))
                    start = finish
                    saver.save(sess,os.path.join(
                        config.model_output,
                        config.model_name+'_'+str(step)+'.ckpt'
                    ),global_step=step)
                    if config.full_cov_matrix:
                        print(cov_mat)

                if step % config.val_iters == 0:
                    val_forward_pass_time = time.time()
                    v_data, v_labels, norm_v_labels, v_res, v_loss = sess.run([val_data, val_labels, norm_val_labels, val_res, val_loss])

                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                    print("\t val loss = {}, time_elapsed = {}s".format(v_loss, time.time() - val_forward_pass_time))
                    '''
                    nparams = np.prod(config.param_dims[1:])
                    color_v = ['r', 'g', 'b', 'k', 'm', 'c', 'y']
                    for k in range(nparams): 
                        plt.scatter(norm_v_labels[:, k], v_res[:, k], c = color_v[k], alpha=0.5); 

                    plt.pause(1);
                    plt.clf()
                    '''
                    if config.full_cov_matrix:
                        data_dump = {'predictions': outputs, 'labels': norm_tr_labels, 'cov':cov_mat}
                        pickle.dump(data_dump, open( os.path.join(config.base_dir,config.summary_dir,config.model_name,'step%d.pickle'%step), 'wb'))
                    
        except tf.errors.OutOfRangeError:
            print("Finished training for %d epochs" % config.epochs)
        finally:
            coord.request_stop()
            coord.join(threads)


def test_rev_model_eval(config):
    test_files = os.path.join(
                        config.base_dir,
                        config.tfrecord_dir,
                        config.test_tfrecords)

    errors = []
    data, labels, preds = [], [], []

    with tf.device('/cpu:0'): 
        '''
        test_labels, test_data = inputs(
                                        tfrecord_file=test_files,
                                        num_epochs=1,
                                        batch_size=config.test_batch,
                                        target_data_dims=config.param_dims,
                                        target_label_dims=config.output_hist_dims)
        '''
        test_data = tf.placeholder(tf.float32, [1000, 1, 256, 2])
        test_labels = tf.placeholder(tf.float32, [1000, 1, 4, 1])

    with tf.device('/gpu:0'):
        with tf.variable_scope("model") as scope:
            model = cnn_reverse_model()
            model.build(test_data, config.output_hist_dims[1:], config.param_dims[1:], train_mode=False)
            y_conv = model.output
            nparams = np.prod(config.param_dims[1:])
            labels = tf.reshape(test_labels, [-1, nparams])
            #labels = (labels - config.min_param_values)/config.param_range
            error = heteroskedastic_loss(y_conv, labels, nparams)

        gpuconfig = tf.ConfigProto()
        gpuconfig.gpu_options.allow_growth = True
        gpuconfig.allow_soft_placement = True
        saver = tf.train.Saver()

        X = pickle.load(open('../data/ddm/parameter_recovery/ddm_param_recovery_data_n_3000.pickle', 'rb'))
        with tf.Session(config=gpuconfig) as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            step=0
            try:
                while not coord.should_stop():
                    # load the model here
                    ckpts=tf.train.latest_checkpoint(config.model_output)
                    saver.restore(sess,ckpts)
                    #ip , op, pred, err, norm_labels = sess.run([test_data, test_labels, y_conv, error, labels])
                    pred, err, norm_labels = sess.run([y_conv,error,labels],feed_dict={test_data: np.expand_dims(X[0],axis=1),test_labels:np.expand_dims(np.expand_dims(X[1],axis=-1),axis=1)})
                    plt.figure(); sc = plt.scatter(norm_labels[:,0], pred[:,0], c=pred[:,4], edgecolors='none', cmap='jet', alpha=0.5); plt.colorbar(sc);
                    plt.figure(); sc = plt.scatter(norm_labels[:,1], pred[:,1], c=pred[:,5], edgecolors='none', cmap='jet', alpha=0.5); plt.colorbar(sc);
                    plt.figure(); sc =plt.scatter(norm_labels[:,2], pred[:,2], c=pred[:,6], edgecolors='none', cmap='jet', alpha=0.5); plt.colorbar(sc);
                    plt.figure(); sc = plt.scatter(norm_labels[:,3], pred[:,3], c=pred[:,7], edgecolors='none', cmap='jet', alpha=0.5); plt.colorbar(sc)
                    plt.show()
        
                    import ipdb; ipdb.set_trace()
                    batch_err = np.sum(err, axis=1)
                    errors.append(batch_err)
                    data.append(ip)
                    labels.append(op)
                    preds.append(pred)
                    print('{} batches complete..'.format(len(errors)))
            except tf.errors.OutOfRangeError:
                print('Epoch limit reached!')
            finally:
                coord.request_stop()
            coord.join(threads)
    import ipdb; ipdb.set_trace()
    '''
    err_vals = np.array(errors).reshape((-1,))
    plt.hist(err_vals, bins=1000)
    plt.title('Model: %s, min error=%0.3f, max error=%0.3f'%(config.model_name,np.min(err_vals), np.max(err_vals)), fontsize=12)
    plt.gca().tick_params(axis='both', which='major', labelsize=6)
    plt.gca().tick_params(axis='both', which='minor', labelsize=6)
    #import ipdb; ipdb.set_trace()
    plt.savefig(os.path.join(config.results_dir, '{}_eval.png'.format(config.model_name)), dpi=300)
    plt.close()

    inp_data = np.array(data)
    inp_data = inp_data.reshape((inp_data.shape[0]*inp_data.shape[1],inp_data.shape[2],inp_data.shape[3]))
    inp_labs = np.array(labels)
    inp_labs = inp_labs.reshape((inp_labs.shape[0]*inp_labs.shape[1],inp_labs.shape[2],inp_labs.shape[3]))
    idx = np.argsort(err_vals)
    net_preds = np.array(preds)
    net_preds = net_preds.reshape((net_preds.shape[0]*net_preds.shape[1],net_preds.shape[2]))
    net_preds = net_preds.reshape(inp_labs.shape)

    # lets draw a 3x3 grid with
    fig, ax = plt.subplots(3,3)
    for k in range(9):
        r, c = int(k/3), k%3
        cur_idx = idx[-1 * (k+1)]
        parameters = np.around(inp_data[cur_idx].flatten(),decimals=2)
        err = err_vals[cur_idx]
        ax[r,c].plot(inp_labs[cur_idx],'r',alpha=0.5)
        ax[r,c].plot(net_preds[cur_idx],'-.g',alpha=0.5)
        mystr = 'err=%0.2f'%(err)
        ax[r,c].text(0.9,.9, "\n".join(wrap('{}, params:{}'.format(mystr, parameters),30)), fontsize=6, horizontalalignment='right', verticalalignment='center', transform=ax[r,c].transAxes)
        #plt.show() 
        ax[r,c].tick_params(axis='both', which='major', labelsize=6)
        ax[r,c].tick_params(axis='both', which='minor', labelsize=6)
    plt.savefig(os.path.join(config.results_dir, '{}_debug.png'.format(config.model_name)),dpi=300)
    plt.close()
    '''
