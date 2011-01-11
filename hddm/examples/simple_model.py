import hddm

# Load data from csv file into a NumPy structured array
data = hddm.utils.csv2rec('example.csv')

# Create a HDDM model multi object
model = hddm.models.Multi(data)

# Create model and start MCMC sampling
model.mcmc()

# Print fitted parameters and other model statistics
print model.summary()

# Plot posterior distributions and theoretical RT distributions
model.plot_posteriors()
model.plot_RT_fit()
