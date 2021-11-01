import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import hddm
import numpy as np

def read_data():
	cols_names = ['a', 'abs_z', 't', 'sv', 'abs_sz', 'st', 'v'];

	rec = np.genfromtxt('Practical_distribution_diffusionmodel.txt', delimiter='\t', comments='#', names=cols_names)
	df = pd.DataFrame(rec)

	df['z'] = df['abs_z']/df['a']
	df['sz'] = df['abs_sz']/df['a']
	df['a'] *= 10
	df['v'] *= 10
	df['sv'] *= 10

	return df

params =  ['a', 'z', 't', 'sv', 'sz', 'st', 'v']


def plot_all_priors(model, data=None, unique=True, model_kwargs=None):
	"""
	plot the priors of an HDDM model

	Input:
		data <DataFrame> - data to be plot against the priors
		unique <bool> - whether to unique each column in data before before ploting it
	"""

	#set limits for plots
	lb = {'v': -10}
	ub = {'a': 4, 't':1, 'v':10, 'z':1, 'sz': 1, 'st':1, 'sv':15, 'p_outlier': 1}

	#plot all priors
	n_rows=4
	n_cols=2
	for n_subjs in [1]: #,2]:

		#create a model
		h_data, _ = hddm.generate.gen_rand_data(subjs=n_subjs, size=2)
		if model_kwargs is None:
			model_kwargs = {}
		h = model(h_data, include='all', **model_kwargs)

		fig = plt.figure()
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=.7)

		counter = 0
		for name, node_row in h.iter_group_nodes():
			if 'var' in name or 'p_outlier' in name:
				continue
			if 'trans' in name:
				trans = True
				name = name.replace('_trans','')
			else:
				trans = False
			counter += 1
			node = node_row['node']

			#plot a single proir
			ax = plt.subplot(n_rows, n_cols, counter)
                        ax.set_yticklabels([])
			#if data is given then plot it
			if data is not None:
				try:
					if unique:
						t_data = data[name].dropna().unique()
					else:
						t_data = data[name].dropna().values

					# if name == 'v':
						# t_data = np.concatenate((t_data, -t_data))
					ax.hist(t_data, 20, normed=True)
				except KeyError:
					pass

			#generate pdf
			xlim = np.arange(lb.get(name, 0.001), ub[name], 0.01)
			pdf = np.zeros(len(xlim))
			for i in range(len(pdf)):
				if not trans:
					node.value = xlim[i]
					pdf[i] = np.exp(node.logp)
				else:
					node.value = pm.logit(xlim[i])
					pdf[i] = np.exp(node.logp)*10

			#plot shit
			plt.plot(xlim, pdf)
			plt.title(name)

		#add suptitle
		if n_subjs > 1:
			plt.suptitle('Group model')
		else:
			plt.suptitle('HDDM Informative model')


def sample_from_group_distrbution(n_rows=5, n_cols=4):
	"""
	present samples from the group distributions of the group parameters
	of HDDMGamma
	"""

	data, _ = hddm.generate.gen_rand_data(subjs=5, size=2)
	h = hddm.models.hddm_gamma.HDDMGamma(data, include=['st', 'sz', 'sv', 'z'])

	params = ['a', 'v', 't','z']
	for p in params:
		plt.figure()

	 	#get the nodes
	 	if p in ['a', 'v', 't']:
		 	g_node = h.nodes_db.ix[p]['node']
		 	var_node = h.nodes_db.ix[p + '_var']['node']
		 	subj_node = h.nodes_db.ix[p + '_subj.1']['node']
		elif p == 'z':
		 	g_node = h.nodes_db.ix['z_trans']['node']
		 	var_node = h.nodes_db.ix['z_var']['node']
		 	z_subj_trans = h.nodes_db.ix['z_subj_trans.1']['node']
		 	z_subj = h.nodes_db.ix['z_subj.1']['node']


	 	for counter in range(n_rows*n_cols):
 			plt.subplot(n_rows, n_cols, counter+1)

 			#samples random values for the group nodes
 			g_node.random()
 			var_node.random()

 			#sample many subjects nodes
 			if p == 'z':
 				subjs_values = np.zeros(500)
 				for i in range(len(subjs_values)):
 					z_subj_trans.random()
 					subjs_values[i] = z_subj.value
 			else:
 				subjs_values = np.concatenate([subj_node.random().flatten() for x in range(500)])

 			#plot it
 			plt.hist(subjs_values, 20)
	 	plt.suptitle(p)



if __name__ == '__main__':
    data = read_data()
    plot_all_priors(hddm.HDDMInfo, data)
    plt.savefig('hddm_info_priors.svg')
    plt.show()
