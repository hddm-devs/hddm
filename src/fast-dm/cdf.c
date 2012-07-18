/* cdf.c - compute the CDF for the diffusion model
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

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "fast-dm.h"


/**********************************************************************
 * struct F_calculator:
 * Store information to calculate the cumulative distribution function F.
 *
 * Usage:
 * 1) Allocate a F_calculator structure with the 'F_new' function below.
 *    This initialises the appropriate method for the given variabilities.
 * 2) Set the initial condition for the PDE with 'F_start'
 * 3) Get an array of computed values at time t with 'F_get_F'.
 *    The field 'F_calculator.N' gives the length of the array.
 * 4) Get the z-value associated with array element 'i' using
 *    the function 'F_get_z'.
 *
 * The values returned by the functions F_get_F and F_get_val are not
 * directly the values of the CDF, but they are transformed to ease
 * the use of the results by the higher levels of fast-dm.  To get the
 * actual values of the CDF, the following transform has to be applied
 * to the function results:
 *
 *    for b_upper: CDF = return value - F^-(\infty)
 *    for b_lower: CDF = F^-(\infty) - return value
 *
 * When all variabilities are zero, F^-(\infty) can be computed using
 * the function 'F_limit'.
 */

struct F_calculator {
	int  N;
	int  plus;
	void *data;

	void (*start) (struct F_calculator *, enum boundary);
	void (*delete) (struct F_calculator *);
	const double *(*get_F) (struct F_calculator *, double t);
	double (*get_z) (const struct F_calculator *, int i);
};

/**********************************************************************
 * parameters to control the precision of the CDF computation
 */

double  TUNE_PDE_DT_MIN;
double  TUNE_PDE_DT_MAX;
double  TUNE_PDE_DT_SCALE;
double  TUNE_DZ;
double  TUNE_DV;
double  TUNE_DT0;

static  int  precision_set = 0;

void
set_precision (double p)
/* Try to achieve an accuracy of approximately 10^{-p} for the CDF.  */
{
	TUNE_PDE_DT_MIN = pow(10, -0.400825*p-1.422813);
	TUNE_PDE_DT_MAX = pow(10, -0.627224*p+0.492689);
	TUNE_PDE_DT_SCALE = pow(10, -1.012677*p+2.261668);
	TUNE_DZ = pow(10, -0.5*p-0.033403);
	TUNE_DV = pow(10, -1.0*p+1.4);
	TUNE_DT0 = pow(10, -0.5*p-0.323859);

	precision_set = 1;
}

static double
F_limit(double a, double z, double v)
{
	if (fabs(v) < 1e-8) {
		return 1 - z/a;
	} else {
		return (exp(-2*v*z)-exp(-2*v*a)) / (1-exp(-2*v*a));
	}
}

/**********************************************************************
 * plain: no variability
 */

struct F_plain_data {
	double  a, v, t0;	/* parameters (except z) */
	double  dz;		/* z step-size */
	double  t;		/* time corresponding to the vector F */
	double *F;		/* state at time t */
};

static void
F_plain_start (struct F_calculator *fc, enum boundary b)
{
	struct F_plain_data *data = fc->data;
	double  a = data->a;
	double  v = data->v;
	int  N = fc->N;
	int  i;

	fc->plus = b;
	data->t = 0;

	data->F[0] = (b == b_upper) ? 1 : 0;
	for (i=1; i<N; i++) {
		double  z = F_get_z (fc, i);
		data->F[i] = F_limit(a, z, v);
	}
	data->F[N] = (b == b_upper) ? 1 : 0;
}

static void
F_plain_delete (struct F_calculator *fc)
{
	struct F_plain_data *data = fc->data;

	xfree (data->F);
	xfree (data);
	xfree (fc);
}

static const double *
F_plain_get_F (struct F_calculator *fc, double t)
{
	struct F_plain_data *data = fc->data;
	if (t > data->t) {
		double  t0 = data->t0;
		if (t > t0) {
			double  from = data->t - t0;
			double  to = t - t0;
			if (from < 0)  from = 0;
			advance_to (fc->N, data->F,
				    from, to, data->dz, data->v);
		}
		data->t = t;
	}
	return  data->F;
}

static double
F_plain_get_z (const struct F_calculator *fc, int i)
{
	struct F_plain_data *data = fc->data;
	return  i * data->dz;
}

static struct F_calculator *
F_plain_new (const double *para)
/* Allocate a new 'struct F_calculator' (without variabilities).  */
{
	struct F_calculator *fc;
	struct F_plain_data *data;
	int  N;

	N = 2*(int)(para[p_a]*0.5/TUNE_DZ+0.5);
	if (N<4)  N = 4;
	/* N must be even, otherwise the case sz == a fails */

	fc = xnew (struct F_calculator, 1);
	fc->N = N;
	fc->plus = -1;
	data = xnew (struct F_plain_data, 1);
	data->a = para[p_a];
	data->v = para[p_v];
	data->t0 = para[p_t0];
	data->dz = para[p_a]/N;
	data->t = 0;
	data->F = xnew (double, N+1);
	fc->data = data;

	fc->start = F_plain_start;
	fc->delete = F_plain_delete;
	fc->get_F = F_plain_get_F;
	fc->get_z = F_plain_get_z;

	return  fc;
}

/**********************************************************************
 * sz: variability in z
 */

struct F_sz_data {
	struct F_calculator *base_fc; /* gives the values we average over */
	double *avg;		      /* the computed averages */
	int  k;			      /* the average involves 2*k+1 cells */
	double  q;		      /* unused part of the outermost cells */
	double  f;		      /* scale factor for the integration */
};

static void
F_sz_start (struct F_calculator *fc, enum boundary plus)
{
	struct F_sz_data *data = fc->data;
	fc->plus = plus;
	F_start (data->base_fc, plus);
}

static void
F_sz_delete (struct F_calculator *fc)
{
	struct F_sz_data *data = fc->data;

	F_delete (data->base_fc);
	xfree (data->avg);
	xfree (data);
	xfree (fc);
}

static const double *
F_sz_get_F (struct F_calculator *fc, double t)
{
	struct F_sz_data *data = fc->data;
	const double *F;
	double  tmp, q, f;
	int  i, j, m;

	F = F_get_F (data->base_fc, t);
	m = 2*data->k;
	q = data->q;
	f = data->f;
	if (m >= 3) {
		for (i=0; i<=fc->N; ++i) {
			tmp = F[i] * 0.5*(1-q)*(1-q);
			tmp += F[i+1] * (1-0.5*q*q);
			for (j=i+2; j<i+m-1; ++j)  tmp += F[j];
			tmp += F[i+m-1] * (1-0.5*q*q);
			tmp += F[i+m] * 0.5*(1-q)*(1-q);
			data->avg[i] = tmp * f;
		}
	} else {		/* m == 2 */
		for (i=0; i<=fc->N; ++i) {
			tmp = F[i] * 0.5*(1-q)*(1-q);
			tmp += F[i+1] * (1-q*q);
			tmp += F[i+2] * 0.5*(1-q)*(1-q);
			data->avg[i] = tmp * f;
		}
	}
	/* m == 1 is impossible here */

	return  data->avg;
}

static double
F_sz_get_z (const struct F_calculator *fc, int i)
{
	struct F_sz_data *data = fc->data;
	return  F_get_z (data->base_fc, i+data->k);
}

static struct F_calculator *
F_sz_new (const double *para)
/* Allocate a new 'struct F_calculator' (with sv == 0 and st == 0).
 *
 * This function can deal with variabilities in z.
 * If 'sz == 0', it just returns the result of 'F_plain_new'.
 */
{
	struct F_calculator *base_fc;
	struct F_calculator *fc;
	struct F_sz_data *data;
	double  tmp, dz;
	int  N, k;

	base_fc = F_plain_new (para);
	if (para[p_sz] == 0)  return base_fc;

	N = base_fc->N;
	dz = F_get_z (base_fc, 1) - F_get_z (base_fc, 0);
	tmp = para[p_sz]/(2*dz);
	k = ceil(tmp) + 0.5;
	assert (2*k <= N);

	fc = xnew (struct F_calculator, 1);
	fc->N = N-2*k;
	fc->plus = -1;
	data = xnew (struct F_sz_data, 1);
	data->base_fc = base_fc;
	data->avg = xnew (double, fc->N+1);
	data->k = k;
	data->q = k - tmp;
	data->f = dz/para[p_sz];
	fc->data = data;

	fc->start = F_sz_start;
	fc->delete = F_sz_delete;
	fc->get_F = F_sz_get_F;
	fc->get_z = F_sz_get_z;

	return  fc;
}

/**********************************************************************
 * sv: variability in v
 */

struct F_sv_data {
	int  nv;		       /* number of points in integration */
	struct F_calculator **base_fc; /* F_calculators for different v */
	double *avg;
};

static void
F_sv_start (struct F_calculator *fc, enum boundary plus)
{
	struct F_sv_data *data = fc->data;
	int  j;

	fc->plus = plus;
	for (j=0; j<data->nv; ++j)
		F_start (data->base_fc[j], plus);
}

static void
F_sv_delete (struct F_calculator *fc)
{
	struct F_sv_data *data = fc->data;
	int  j;

	for (j=0; j<data->nv; ++j)
		F_delete (data->base_fc[j]);
	xfree (data->base_fc);
	xfree (data->avg);
	xfree (data);
	xfree (fc);
}

static const double *
F_sv_get_F (struct F_calculator *fc, double t)
{
	struct F_sv_data *data = fc->data;
	const double *F;
	double *avg = data->avg;
	int  i, j;

	F = F_get_F(data->base_fc[0], t);
	for (i=0; i<=fc->N; ++i)  avg[i] = F[i];
	for (j=1; j<data->nv; ++j) {
		F = F_get_F(data->base_fc[j], t);
		for (i=0; i<=fc->N; ++i)  avg[i] += F[i];
	}
	for (i=0; i<=fc->N; ++i)  avg[i] /= data->nv;

	return  avg;
}

static double
F_sv_get_z (const struct F_calculator *fc, int i)
{
	struct F_sv_data *data = fc->data;
	return  F_get_z (data->base_fc[0], i);
}

static struct F_calculator *
F_sv_new (const double *para)
/* Allocate a new 'struct F_calculator'.
 *
 * This initialises the PDE and prepares most things for the
 * calculation.  The initial condition for the returned 'struct
 * F_calculator' has to be set using 'F_start'.
 *
 * This function can deal with variabilities in all parameters.
 * If 'sv == 0', it just return the result of 'F_st_new'.
 */
{
	struct F_calculator **base_fc;
	struct F_calculator *fc;
	struct F_sv_data *data;
	double para2 [p_count];
	int  nv, j;

	if (para[p_sv] == 0)  return  F_sz_new (para);

	nv = para[p_sv]/TUNE_DV + 0.5;
	if (nv < 3)  nv = 3;
	memcpy (para2, para, p_count*sizeof(double));
	para2[p_sv] = 0;
	base_fc = xnew (struct F_calculator *, nv);
	for (j=0; j<nv; ++j) {
		double  x = Phi_inverse ((0.5+j)/nv);
		para2[p_v] = para[p_sv]*x + para[p_v];
		base_fc[j] = F_sz_new (para2);
	}

	fc = xnew (struct F_calculator, 1);
	fc->N = base_fc[0]->N;
	fc->plus = -1;
	data = xnew (struct F_sv_data, 1);
	data->nv = nv;
	data->base_fc = base_fc;
	data->avg = xnew (double, fc->N+1);
	fc->data = data;

	fc->start = F_sv_start;
	fc->delete = F_sv_delete;
	fc->get_F = F_sv_get_F;
	fc->get_z = F_sv_get_z;

	return  fc;
}

/**********************************************************************
 * st0: variability in t0
 *
 * This implements numerical integration in t-direction, using the
 * trapez rule.  Since computing the CDF is slow and since we can
 * solve the PDE only forward in time, we cache old values of the CDF.
 *
 * The cached values form a grid of M different t-values such that the
 * smallest cached t-value is smaller or equal than t-0.5*st0, the
 * biggest cached t-value is bigger or equal than t+0.5*st0.  The
 * total length of the cached time interval is (M-1)*dt where M and dt
 * are chosen such that st0 = (M-2)*dt.
 */

struct F_st0_data {
	struct F_calculator *base_fc;
	double  st0;		/* variability of t0 */
	int  M;			/* number of stored grid lines */
	double  start;		/* t-value of first stored grid line */
	double  dt;		/* t-spacing of stored grid lines */
	double *values;		/* array: stored grid lines (length M*(N+1)) */
	char *valid;		/* which lines in 'values' are valid */
	int  base;		/* first grid line starts at pos. base*(N+1) */
	double *avg;		/* the computed average (size N+1) */
};

static void
F_st0_start (struct F_calculator *fc, enum boundary plus)
{
	struct F_st0_data *data = fc->data;
	int j;

	fc->plus = plus;
	F_start (data->base_fc, plus);
	data->start = -0.5*data->st0;

	/* initially mark all of the cache as invalid */
	for (j=0; j<data->M; ++j)
		data->valid[j] = 0;
}

static void
F_st0_delete (struct F_calculator *fc)
{
	struct F_st0_data *data = fc->data;

	F_delete (data->base_fc);
	xfree (data->valid);
	xfree (data->values);
	xfree (data->avg);
	xfree (data);
	xfree (fc);
}

static const double *
F_st0_get_row(const struct F_calculator *fc, int j)
/* Get a pointer to one of the stored grid lines.
 * The value j is the grid line index (range 0, ..., M-1).
 * The returned value is a pointer to an array of length N+1.  */
{
	const struct F_st0_data *data = fc->data;
	int  M, N, idx;
	double *row;

	M = data->M;
	N = fc->N;
	assert(0 <= j && j < M);
	idx = (data->base + j)%M;
	row = data->values + idx*(N+1);

	if (! data->valid[idx]) {
		double t;
		const double *F;

		t = data->start + j*data->dt;
		F = F_get_F(data->base_fc, t);
		memcpy(row, F, (N+1)*sizeof(double));
		data->valid[idx] = 1;
	}

	return row;
}

static void
add_vec(long n, double a, const double *x, double *y)
/* the vector operation y += a*x */
{
#ifdef HAVE_LIBBLAS
	extern void daxpy_(long *Np, double *DAp, const double *X, long *INCXp,
			   double *Y, long *INCYp);
	long inc = 1;
	daxpy_(&n, &a, x, &inc, y, &inc);
#else /* ! HAVE_LIBBLAS */
	int i;
	if (a == 1) {
		for (i=0; i<n; ++i) y[i] += x[i];
	} else {
		for (i=0; i<n; ++i) y[i] += a*x[i];
	}
#endif /* ! HAVE_LIBBLAS */
}

static const double *
F_st0_get_F (struct F_calculator *fc, double t)
{
	struct F_st0_data *data = fc->data;
	double  a, b, dt;
	const double *row;
	double  q, r, *avg;
	int  M, N, shift;
	int  i, j, m;

	a = t - 0.5*data->st0;
	b = t + 0.5*data->st0;
	dt = data->dt;
	M = data->M;
	N = fc->N;

	/* how many of the precalculated rows can we keep? */
	if (a - data->start >= M*dt) {
		/* beware of integer overflows for small dt */
		shift = M;
	} else {
		shift = (a - data->start)/dt;
		assert (shift >= 0);
	}

	for (j=0; j<shift; ++j)
		data->valid[(data->base+j)%M] = 0;

	if (shift < M) {
		data->start += shift*dt;
		data->base = (data->base+shift)%M;
	} else {
		data->start = a;
	}

	/* compute the average over the rows from a to b */
	avg = data->avg;
	for (i=0; i<=N; ++i)  avg[i] = 0;
	{
		double  tmp = (b - data->start)/dt;
		m = ceil (tmp) + 0.5;
		if (m >= M)  m = M-1; /* protect against rounding errors */
		q = (a - data->start)/dt;
		r = m - tmp;
	}
	if (m >= 3) {
		row = F_st0_get_row(fc, 0);
		add_vec(N+1, 0.5*(1-q)*(1-q), row, avg);

		row = F_st0_get_row(fc, 1);
		add_vec(N+1, 1-0.5*q*q, row, avg);

		for (j=2; j<m-1; ++j) {
			row = F_st0_get_row(fc, j);
			add_vec(N+1, 1, row, avg);
		}

		row = F_st0_get_row(fc, m-1);
		add_vec(N+1, 1-0.5*r*r, row, avg);

		row = F_st0_get_row(fc, m);
		add_vec(N+1, 0.5*(1-r)*(1-r), row, avg);
	} else if (m == 2) {
		row = F_st0_get_row(fc, 0);
		add_vec(N+1, 0.5*(1-q)*(1-q), row, avg);

		row = F_st0_get_row(fc, 1);
		add_vec(N+1, 1-0.5*(q*q+r*r), row, avg);

		row = F_st0_get_row(fc, 2);
		add_vec(N+1, 0.5*(1-r)*(1-r), row, avg);
	} else if (m == 1) {
		row = F_st0_get_row(fc, 0);
		add_vec(N+1, 0.5*((1-q)*(1-q)-r*r), row, avg);

		row = F_st0_get_row(fc, 1);
		add_vec(N+1, 0.5*((1-r)*(1-r)-q*q), row, avg);
	}

	for (i=0; i<=N; ++i)  avg[i] *= dt/(b-a);
	return  avg;
}

static double
F_st0_get_z (const struct F_calculator *fc, int i)
{
	struct F_st0_data *data = fc->data;
	return  F_get_z (data->base_fc, i);
}

static struct F_calculator *
F_st0_new (const double *para)
/* Allocate a new 'struct F_calculator' (with sv == 0).
 *
 * This function can deal with variabilities in z and t.
 * If 'st0 == 0', it just returns the result of 'F_sz_new'.
 */
{
	struct F_calculator *base_fc;
	struct F_calculator *fc;
	struct F_st0_data *data;
	double  dt;
	int  M, N;

	base_fc = F_sv_new (para);
	if (para[p_st0] <= TUNE_DT0*1e-6)  return base_fc;

	M = para[p_st0]/TUNE_DT0 + 1.5;
	if (M < 3)  M = 3;
	dt = para[p_st0]/(M-2);
	N = base_fc->N;

	fc = xnew (struct F_calculator, 1);
	fc->N = N;
	fc->plus = -1;
	data = xnew (struct F_st0_data, 1);
	data->st0 = para[p_st0];
	data->base_fc = base_fc;
	data->M = M;
	/* data->start is set in F_st0_start */
	data->dt = dt;
	data->values = xnew (double, M*(N+1));
	data->valid = xnew (char, M);
	data->base = 0;
	data->avg = xnew (double, N+1);
	fc->data = data;

	fc->start = F_st0_start;
	fc->delete = F_st0_delete;
	fc->get_F = F_st0_get_F;
	fc->get_z = F_st0_get_z;

	return  fc;
}

/**********************************************************************
 * externally visible routines
 */

struct F_calculator *
F_new (const double *para)
/* Allocate data required to compute a CDF.
 *
 * The array 'para' contains the model parameters.  The 'enum
 * parameter_index' tells which parameter is stored in which entry.
 * The array must be of size 'p_count'.
 *
 * The returned structure must be initialised with 'F_start' and must
 * be freed with 'F_delete' after use.
 */
{
	assert (precision_set);
	return  F_st0_new (para);
}

void
F_delete (struct F_calculator *fc)
/* Free a 'struct F_calculator' and all associated resources.
 * 'fc' must have been allocated by 'F_new'.  After 'F_delete' is
 * called, 'fc' cannot be used any longer.  */
{
	fc->delete (fc);
}

void
F_start (struct F_calculator *fc, enum boundary b)
/* Set the initial condition for the PDE.
 *
 * If 'b' is 'b_upper', prepare to calculate the CDF for hitting a
 * before 0, otherwise prepare to calculate the CDF for hitting 0
 * before a.
 */
{
	fc->start (fc, b);
}

int
F_get_N (const struct F_calculator *fc)
{
	return  fc->N;
}

double
F_get_z (const struct F_calculator *fc, int i)
/* Get the z-value corresponding to index i.  */
{
	return  fc->get_z (fc, i);
}

const double *
F_get_F (struct F_calculator *fc, double t)
/* Get the array of CDF values at time t for all grid points z.
 *
 * 'F_start' must be used for initialisation before calling 'F_get_F'.
 * Between calls of 'F_start' the calls to 'F_get_F' must have
 * increasing values of 't'.
 *
 * The returned array is owned by the 'struct F_calculator' and must
 * not be changed or freed.
 */
{
	return  fc->get_F (fc, t);
}

double
F_get_val (struct F_calculator *fc, double t, double z)
/* Get the value of the CDF for the parameters given when creating fc.
 * The function Uses linear interpolation for z-values between the grid points.
 * Don't use this function for parameter fitting, since it is not very fast
 * (use 'F_get_F' instead).
 */
{
	const double *F;
	double  z0, z1;
	double  p, x;
	int  N = fc->N;
	int  i;

	F = F_get_F (fc, t);
	if (N == 0) {
		x = F[0];
	} else {
		z0 = F_get_z (fc, 0);
		z1 = F_get_z (fc, N);
		i = N*(z-z0)/(z1-z0);
		if (i < N) {
			z0 = F_get_z (fc, i);
			z1 = F_get_z (fc, i+1);
			p = (z1-z) / (z1-z0);
			x = p*F[i] + (1-p)*F[i+1];
		} else {
			x = F[N];
		}
	}
	return  x;
}

/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
