/* simplex2.c - downhill simplex method of Nelder and Mead
 *              (2 valued objective function, lexicographical order)
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
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "fast-dm.h"


/* The following parameters refer to the description of the method in
 * Jeffrey C. Lagarias, James A. Reeds, Margaret H. Wright, and Paul
 * E. Wright: Convergence Properties of the Nelder-Mead Simplex Method
 * In Low Dimensions.  SIAM J. Optim, Vol. 9 (1998), No. 1,
 * pp. 112-147.
 *
 * http://citeseer.ist.psu.edu/3996.html
 */

#define RHO 1.0			/* reflection */
#define CHI 2.0			/* expansion */
#define GAMMA 0.5		/* contraction */
#define SIGMA 0.5		/* shrink */


static void
move_point(int n, double p, const double *x, const double *y, double *z)
/* Move a point in direction of another point.
 * x, y and z must be vectors of length n.
 * The function reads x and y and stores x + p*(y-x) in z.
 * For p=2 this calculates the reflection of x at y.  */
{
	int  i;
	for (i=0; i<n; ++i)  z[i] = x[i] + p*(y[i]-x[i]);
}

static int
compare_points(const void *a, const void *b)
/* lexicographical comparison of pairs of values */
{
	const double *da = a;
	const double *db = b;
	if (da[0]<db[0])  return -1;
	if (da[0]>db[0])  return +1;
	if (da[1]<db[1])  return -1;
	if (da[1]>db[1])  return +1;
	return 0;
}

void
simplex2(int n, const double *eps, double size_limit,
	 void (*fn)(const double *x, double res[2], void *data),
	 double *x, double *fn_ret, void *data)
/* Multidimensional minimisation of the function 'fn' using the
 * downhill simplex algorithm of Nelder and Mead.  This version of the
 * function uses a two-valued objective function.
 *
 * 'fn' is the function to minimise.  The first argument of 'fn' is a
 * vector x which consists of 'n' doubles.  With the 'data' pointer
 * additional information may be forwarded to 'fn'.  On input the
 * vector 'x' defines one vertex of the starting simplex.  For the
 * other vertices, one of the n components of x is incremented by the
 * according value of 'eps'.  'size_limit' defines the precision of
 * the search.  The algorithm is terminated as soon as all edges of
 * the simplex are smaller than 'size_limit'.
 *
 * On return the vector 'x' contains the best found solution, that
 * is, the parameters with the smallest result of 'fn'.  If 'fn_ret'
 * is not NULL, the result of 'fn' applied to 'x' is copied into this
 * vector.
 */
{
	double *points;
	double *mid, *tmp1, *tmp2;
	int  i, j, k;

	assert (n >= 2);

	/* Allocate an array for the vertices of the simplex.  Vertex
	 * k is stored in entries k*(n+2)+2, ..., k*(n+2)+n+1, the
	 * function value at this vertex is stored in entries k*(n+2)
	 * and k*(n+2)+1.  Vertices are stored in order of increasing
	 * function values.  */
	points = xnew(double, (n+1)*(n+2));
	mid = xnew(double, n+2);
	tmp1 = xnew(double, n+2);
	tmp2 = xnew(double, n+2);

	/* initialise the simplex */
	for (j=0; j<=n; ++j) {
		for (i=0; i<n; ++i) {
			points[j*(n+2)+2+i] = x[i] + ((i+1)==j ? eps[i] : 0);
		}
		fn(points+j*(n+2)+2, points+j*(n+2), data);
	}

	for (k=0; ; ++k) {
		double  size;

		qsort(points, n+1, (n+2)*sizeof(double), compare_points);

		/* stopping criterion */
		size = 0;
		for (i=0; i<n; ++i) {
			double  min, max;

			min = max = points[2+i];
			for (j=1; j<=n; ++j) {
				double  x = points[j*(n+2)+2+i];
				if (x < min) {
					min = x;
				} else if (x > max) {
					max = x;
				}
			}
			if (max-min > size)  size = max-min;
			if (size > size_limit)  break;
		}
		if (size < size_limit)  break;

		/* calculate the reflection point */
		for (i=2; i<=n+1; ++i)  mid[i] = points[i];
		for (j=1; j<n; ++j) {
			for (i=2; i<=n+1; ++i)  mid[i] += points[j*(n+2)+i];
		}
		for (i=2; i<=n+1; ++i)  mid[i] /= n;

		/* try to reflect */
		move_point(n, 1+RHO, points+n*(n+2)+2, mid+2, tmp1+2);
		fn(tmp1+2, tmp1, data);
		if (compare_points(points+0*(n+2), tmp1) <= 0
		    && compare_points(tmp1, points+(n-1)*(n+2)) < 0) {
			/* accept the reflection */
			memcpy(points+n*(n+2), tmp1, (n+2)*sizeof(double));
			continue;
		}

		if (compare_points(tmp1, points+0*(n+2)) < 0) {
			/* try the extended reflection */
			move_point(n, 1+CHI,
				   points+n*(n+2)+2, mid+2, tmp2+2);
			fn(tmp2+2, tmp2, data);
			if (compare_points(tmp2, tmp1) < 0) {
				/* accept the extended reflection */
				memcpy(points+n*(n+2), tmp2,
				       (n+2)*sizeof(double));
			} else {
				/* accept the original reflection */
				memcpy(points+n*(n+2), tmp1,
				       (n+2)*sizeof(double));
			}
			continue;
		}

		if (compare_points(tmp1, points+n*(n+2)) < 0) {
			/* try an outside contraction */
			move_point(n, 1-GAMMA,
				   tmp1+2, mid+2, tmp2+2);
			fn(tmp2+2, tmp2, data);
			if (compare_points(tmp2, tmp1) <= 0) {
				/* accept the outside contraction */
				memcpy (points+n*(n+2), tmp2,
					(n+2)*sizeof(double));
				continue;
			}
		} else {
			/* try an inside contraction */
			move_point(n, 1-GAMMA,
				   points+n*(n+2)+2, mid+2, tmp2+2);
			fn(tmp2+2, tmp2, data);
			if (compare_points(tmp2, points+n*(n+2)) < 0) {
				/* accept the inside contraction */
				memcpy (points+n*(n+2), tmp2,
					(n+2)*sizeof(double));
				continue;
			}
		}

		/* if everything else fails, use shrinking */
		for (j=1; j<=n; ++j) {
			/* move point j towards the best point */
			move_point(n, 1-SIGMA, points+j*(n+2)+2, points+2,
				   points+j*(n+2)+2);
			fn(points+j*(n+2)+2, points+j*(n+2), data);
		}
	}

	if (fn_ret)
		memcpy(fn_ret, points, 2*sizeof(double));
	memcpy(x, points+2, n*sizeof(double));

	xfree(tmp1);
	xfree(tmp2);
	xfree(mid);
	xfree(points);
}

/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
