import hddm

# Load data from csv file into a NumPy structured array
data = hddm.load_csv('simple_difficulty.csv')

# Create a HDDM model multi object
model = hddm.HDDM(data, depends_on={'v':'difficulty'})

# Create model and start MCMC sampling
model.sample(10000, burn=5000)

# Print fitted parameters and other model statistics
model.print_stats()

# Plot posterior distributions and theoretical RT distributions
model.plot_posteriors(save=True)
model.plot_posterior_predictive(save=True)
