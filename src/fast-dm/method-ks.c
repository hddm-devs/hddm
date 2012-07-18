/* method-ks.c - Kolmogorov-Smirnov method for fast-dm
 *
 * Copyright (C) 2007  Jochen Voss, Andreas Voss.
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
#include <math.h>

#include "fast-dm.h"


struct distance *
KS_get_distance(const struct samples *samples, const double *para)
/* Compute the KS test statistic
 *
 * The function returns a 'struct distance' which the caller must free
 * after use.
 */
{
	struct F_calculator *fc;
	struct distance *result;
	double  p_mid, dp;
	double *T;
	int  i, j, N;

	fc = F_new(para);
	N = F_get_N(fc);

	result = new_distance(N, F_get_z(fc, 0), F_get_z(fc, N));
	T = result->T;
	for (i=0; i<=N; ++i)  T[i] = 0;

	dp = 1.0/(samples->plus_used+samples->minus_used);
	p_mid = samples->minus_used*dp;

	F_start (fc, b_upper);
	for (j=0; j<samples->plus_used; ++j) {
		const double *F = F_get_F(fc, samples->plus_data[j]);
		for (i=0; i<=N; ++i) {
			double p_theo = F[i];
			double dist;

			dist = fabs(p_mid+j*dp-p_theo);
			if (dist > T[i])  T[i] = dist;
			dist = fabs(p_mid+(j+1)*dp-p_theo);
			if (dist > T[i])  T[i] = dist;
		}
	}

	F_start(fc, b_lower);
	for (j=0; j<samples->minus_used; ++j) {
		const double *F = F_get_F(fc, samples->minus_data[j]);
		for (i=0; i<=N; ++i) {
			double p_theo = F[i];
			double dist;

			dist = fabs(p_mid-j*dp-p_theo);
			if (dist > T[i])  T[i] = dist;
			dist = fabs(p_mid-(j+1)*dp-p_theo);
			if (dist > T[i])  T[i] = dist;
		}
	}

	F_delete(fc);
	return  result;
}

static double
Q_KS(double lambda)
/* Implement equation (14.3.7) from numerical recipes.  */
{
	double accuracy = 0.001;
	double  c, d, limit, sum;
	int  n, nn;

	/* We want to avoid the random fluctuations (caused by
	 * truncating the sum) for small values of lambda.  The exact
	 * value at lambda=0 is 1.  The code below returns
	 * 1-1.02072e-05 for lambda=0.3 with accuracy=0.0001.  It
	 * returns 1-0.000314775 for lambda=0.35 with accuracy=0.001.
	 * We use linear interpolation between these points.  */
	if (lambda <= 0.3) {
		double a = 1.02072e-05;
		return 1.0 - a*lambda/0.3;
	} else if (lambda <= 0.35) {
		double a = 1.02072e-05;
		double b = 0.000314775;
		return 1-a - (b-a)*(lambda-0.3)/(0.35-0.3);
	}

	c = 2*lambda*lambda;
	d = 1/c;
	limit = d*log(d/(6*accuracy));
	n = 0;
	sum = 0;
	do {
		n += 1;
		nn = n*n;
		sum -= exp(-c*nn);
		n += 1;
		nn = n*n;
		sum += exp(-c*nn);
	} while (n<6 || nn<=limit);
	return  -2*sum;
}

double
KS_T_to_p(double T, const struct samples *s)
/* Implement equation (14.3.9) from numerical recipes.  */
{
	int n = s->plus_used + s->minus_used;
	double sqrtn = sqrt(n);

	return  Q_KS((sqrtn + 0.12 + 0.11/sqrtn)*T);
}

/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
