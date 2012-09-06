## Process this file with automake to produce Makefile.in
# Copyright 2006  Jochen Voss

bin_PROGRAMS = fast-dm
noinst_PROGRAMS = plot-cdf construct-samples

fast_dm_SOURCES = main.c method-ks.c EZ-diff.c simplex2.c cdf.c phi.c	\
	pde.c experiment.c dataset.c container.c file.c xmalloc.c	\
	fast-dm.h

plot_cdf_SOURCES = plot-cdf.c cdf.c phi.c pde.c xmalloc.c fast-dm.h

construct_samples_SOURCES = construct-samples.c cdf.c phi.c pde.c	\
	xmalloc.c fast-dm.h jvrandom.h

EXTRA_DIST = MANUAL win32erf.c win32dir.c win32dir.h
