import hddm

# Load data from csv file into a NumPy structured array
data = hddm.load_csv('simple_subjs_difficulty.csv')

# Create a HDDM model multi object
model = hddm.HDDM(data, depends_on={'v':'difficulty'})

# Create model and start MCMC sampling
model.sample(1000, burn=500)

# Print fitted parameters and other model statistics
model.print_stats()

# Plot posterior distributions and theoretical RT distributions
model.plot_posteriors()
model.plot_posterior_predictive(savefig=True)
