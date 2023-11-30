#!/usr/bin/python
import hddm
import pylab as pl

m = hddm.utils.create_test_model(samples=10000, burn=5000, subjs=12, size=200)
m.plot_posterior_quantiles()

pl.show()
