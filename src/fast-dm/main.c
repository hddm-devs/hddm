/* main.c - main program for the fast-dm project
 *
 * Copyright (C) 2006  Jochen Voss, Andreas Voss.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA.
 */

#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <math.h>
#include <float.h>
#include <time.h>
#include <assert.h>

#include "fast-dm.h"


/**********************************************************************
 * manage the distance to minimise, simultaneously for many z
 */

struct distance *
new_distance(int N, double zmin, double zmax)
/* Allocate a new structure to hold N+1 values in [zmin,zmax] */
{
	struct distance *d;

	assert(N >= 0);

	d = xnew(struct distance, 1);
	d->N = N;
	d->T = xnew(double, N+1);
	d->zmin = zmin;
	d->zmax = zmax;

	return d;
}

void
delete_distance(struct distance *d)
{
	xfree(d->T);
	xfree(d);
}

static double
distance_interpolate(const struct distance *d, double z)
/* Use linear interpolation to get the distance for a given z.  */
{
	int  N = d->N;
	double  dz, ratio;
	int  step_before;

	if (N == 0)
		return d->T[0];

	dz = (d->zmax - d->zmin) / N;
	step_before = (z - d->zmin) / dz;
	if (step_before >= N)  step_before = N-1;
	ratio = (z - (d->zmin + dz*step_before)) / dz;
	return  (1-ratio) * d->T[step_before]
		+ ratio * d->T[step_before+1];
}

/**********************************************************************
 * parameter fitting
 */

static void
find_best_log_p(struct distance *const*result, const struct dataset *ds,
		const int *z_used, double dist_ret[2], double *ret_z)
/* Combine the T-values from 'result' into a common p-value.
 *
 * This computes p-values as the product of the probabilities for all
 * Ts from result and optimises over 'z'.  This takes the different
 * experimental conditions into account.
 *
 * On entry 'z_ret' must point to an array of size 'ds->z->used'.
 *
 * The distance is returned in 'dist_ret'.  If the parameters are
 * invalid, dist_ret[0]>0 on exit.  Otherwise dist_ret[0]=0 and
 * dist_ret[1] = -log(p).  The z-parameter values corresponding to
 * this (minimal) distance are returned in '*z_ret'.
 */
{
	double  total_log;
	double  penalty;
	int  k;

	total_log = 0;
	penalty = 0;
	for (k=0; k<ds->z->used; k++) {
		double  dz;
		double  best_logp, best_z;
		double  zmin = result[0]->zmin;
		double  zmax = result[0]->zmax;
		int  i, j, n;

		for (j=1; j<ds->samples_used; ++j) {
			if (z_used[j]!=k) continue;
			if (zmin < result[j]->zmin) zmin = result[j]->zmin;
			if (zmax > result[j]->zmax) zmax = result[j]->zmax;
		}
		if (zmax < zmin) {
			/* no common z is available: abort the
			 * computation and return a penalty value >1.  */
			ret_z[k] = (zmax+zmin)/2.0;
			penalty = 1 + (zmin-zmax);
			break;
		}

		/* Use sub-sampling of the interval zmin..zmax
		 * to find the best z.  */
		n = (zmax - zmin) / 0.0001 + 1.5;
		dz = (zmax - zmin) / n;
		best_logp = - DBL_MAX;
		best_z = n/2;
		for (i=0; i<=n; ++i) {
			double  z = zmin + i*dz;
			double  logp = 0;

			for (j=0; j<ds->samples_used; ++j) {
				double	T;

				if (z_used[j] != k)  continue;
				T = distance_interpolate(result[j], z);
				logp += log(KS_T_to_p(T, ds->samples[j]));
			}
			if (logp > best_logp) {
				best_logp = logp;
				best_z = z;
			}
		}
		total_log += best_logp;
		ret_z[k] = best_z;
	}

	if (penalty > 0) {
		dist_ret[0] = penalty;
		dist_ret[1] = 0;
	} else {
		dist_ret[0] = 0;
		dist_ret[1] = - total_log;
	}
}

static void
find_fixed_log_p(struct distance *const*result, const struct dataset *ds,
		 double dist_ret[2], double z)
/* Combine the T-values from 'result' into a common p-value.
 *
 * 'z' gives the fixed z-value.
 *
 * If the parameters are invalid, dist_ret[0]>0 on exit.  Otherwise
 * dist_ret[0]=0 and dist_ret[1] = -log(p).
 */
{
	double  logp;
	double  zmin = result[0]->zmin;
	double  zmax = result[0]->zmax;
	int  j;

	for (j=1; j<ds->samples_used; ++j) {
		if (zmin < result[j]->zmin) zmin = result[j]->zmin;
		if (zmax > result[j]->zmax) zmax = result[j]->zmax;
	}
	if (z < zmin) {
		dist_ret[0] = 1 + (zmin-z);
		dist_ret[1] = 0;
	}
	if (z > zmax) {
		dist_ret[0] = 1 + (z-zmax);
		dist_ret[1] = 0;
	}

	logp = 0;
	for (j=0; j<ds->samples_used; ++j) {
		double	T;

		T = distance_interpolate(result[j], z);
		logp += log(KS_T_to_p(T, ds->samples[j]));
	}
	dist_ret[0] = 0;
	dist_ret[1] = -logp;
}

static double
check_bounds(const double *para)
/* Check whether the parameters 'para' are valid.
 *
 * If the parameter set is valid, 0 is returned.  In case of invalid
 * parameters, a value >1 is returned, the magnitude gives the
 * 'badness' of the violation.
 */
{
	double  penalty = 0;
	int  bad = 0;

	if (para[p_sz] < 0) {
		bad = 1;
		penalty += -para[p_sz];
	}
	if (para[p_st0] < 0) {
		bad = 1;
		penalty += -para[p_st0];
	}
	if (para[p_sv] < 0) {
		bad = 1;
		penalty += -para[p_sv];
	}

	if (para[p_a] < para[p_sz]) {
		bad = 1;
		penalty += para[p_sz] - para[p_a];
	}
	if (para[p_t0] < 0.5*para[p_st0]) {
		bad = 1;
		penalty += 0.5*para[p_st0] - para[p_t0];
	}

	/* avoid problems caused by rounding errors */
	return  bad ? penalty+1.0 : 0.0;
}

static void
distance(const struct dataset *ds, const double *x,
	 double dist_ret[2], double *z_ret)
/* Get the 'distance' between theoretical and target distribution.
 *
 * The observed distribution is described by the dataset 'ds', the
 * predicted distribution is described by the parameters 'x'.  The
 * correnspondence between entries of 'x' and the model parameters is
 * encoded in the 'ds->cmds' field.
 *
 * If the parameter 'z' is being optimised, 'z_ret' must on entry
 * point to an array of size 'ds->z->used'.  If the parameter 'z' is
 * fixed, the value 'z_ret' must be 'NULL'.
 *
 * If the parameters 'x' are invalid, a positive penalty value is
 * returned in dist_ret[0].  Otherwise dist_ret[0]=0 and the distance
 * between empirical and theoretically predicted distribution function
 * is returned in 'dist_ret[1]' as -log(p).  If 'z_ret' is non-null,
 * the z-parameter values corresponding to this (minimal) distance are
 * returned in '*z_ret'.
 */
{
	struct distance **result;
	int *z_used;
	double  para[p_count], para_z, penalty;
	int  i, z_idx = -1;

	result = xnew(struct distance *, ds->samples_used);
	for (i=0; i<ds->samples_used; ++i) result[i] = NULL;
	z_used = xnew(int, ds->samples_used);

	penalty = 0;
	for (i=0; i<ds->cmds_used; ++i) {
		int  arg1 = ds->cmds[i].arg1;
		int  arg2 = ds->cmds[i].arg2;
		switch (ds->cmds[i].cmd) {
		case c_copy_param:
			if (arg1 >= 0) {
				para[arg1] = x[arg2];
			} else {
				assert(z_ret);
				z_idx = arg2;
			}
			break;
		case c_copy_const:
			if (arg1 >= 0) {
				para[arg1] = ds->consts[arg2];
			} else {
				assert(! z_ret);
				para_z = ds->consts[arg2];
			}
			break;
		case c_run:
			penalty = check_bounds(para);
			if (! z_ret) {
				double z = para_z*para[p_a];
				if (z <= 0.5*para[p_sz])
					penalty = 1 + 0.5*para[p_sz] - z;
				if (z+0.5*para[p_sz] >= para[p_a])
					penalty = 1 + z+0.5*para[p_sz] - para[p_a];
			}
			if (penalty > 0) break;
			result[arg1] = KS_get_distance(ds->samples[arg1], para);
			z_used[arg1] = z_idx;
			break;
		}
		if (penalty>0) break;
	}

	if (penalty>0) {
		dist_ret[0] = penalty;
		dist_ret[1] = 0;
	} else if (z_ret) {
		find_best_log_p(result, ds, z_used, dist_ret, z_ret);
	} else {
		find_fixed_log_p(result, ds, dist_ret, para_z*para[p_a]);
	}

	xfree(z_used);
	for (i=0; i<ds->samples_used; ++i) {
		if (result[i]) delete_distance(result[i]);
	}
	xfree(result);
}

static void
minimiser(const double *x, double res[2], void *data)
/* Wrapper to call 'distance' from inside the 'simplex2' function.  */
{
	struct dataset *ds = data;
	double *z;

	z = xnew(double, ds->z->used);
	distance(ds, x, res, z);
	xfree(z);
}

static void
initialise_parameters(const struct dataset *ds, double *x, double *eps)
{
	double  def_x [p_count], def_eps [p_count];
	int  i;

	def_x[p_a] = 1;  def_eps[p_a] = 0.5;
	def_x[p_v] = 0;  def_eps[p_v] = 1;
	def_x[p_t0] = 0.3;  def_eps[p_t0] = 0.5;
	def_x[p_sz] = 0.2;  def_eps[p_sz] = 0.2;
	def_x[p_sv] = 0.2;  def_eps[p_sv] = 0.2;
	def_x[p_st0] = 0.15;  def_eps[p_st0] = 0.15;

	for (i=ds->cmds_used-1; i>=0; --i) {
		int  arg1 = ds->cmds[i].arg1;
		int  arg2 = ds->cmds[i].arg2;
		switch (ds->cmds[i].cmd) {
		case c_copy_param:
			if (arg1 < 0)  break;
			x[arg2] = def_x[arg1];
			eps[arg2] = def_eps[arg1];
			break;
		case c_run:
			EZ_par(ds->samples[arg1],
			       def_x+p_a, def_x+p_v, def_x+p_t0);
			def_x[p_sz] = 0.33*def_x[p_a];
			break;
		default:
			break;
		}
	}
}

static void
print_dist(const double *dist)
{
	if (dist[0] > 0)
		printf("  ... penalty %g\n", dist[0]);
	else
		printf("  ... p = %g\n", exp(-dist[1]));
}


/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
