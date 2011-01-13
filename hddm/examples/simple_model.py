import hddm
import kabuki
import pylab

# Load data from csv file into a NumPy structured array
data = kabuki.utils.load_csv('simple_subj_data.csv')

# Create a HDDM model multi object
model = hddm.HDDM(data, is_subj_model=False)

# Create model and start MCMC sampling
model.mcmc()

# Print fitted parameters and other model statistics
print model.summary()

# Plot posterior distributions and theoretical RT distributions
hddm.plot_posteriors(model)
hddm.plot_rt_fit(model)

pylab.show()
