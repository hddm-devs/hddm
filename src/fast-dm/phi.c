/* phi.c - the CDF and inverse CDF of the standard normal distribution
 *
 * Copyright (C) 2006  Jochen Voss.
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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <math.h>

#include "fast-dm.h"

double
Phi (double x)
/* The distribution function of the standard normal distribution.  */
{
	return  0.5*(1+erf (x/M_SQRT2));
}

double
Phi_inverse (double y)
/* The inverse of Phi, calculated using the bisection method */
{
	double  l, r;

	if (y<=0.5) {
		l = -1;
		while (Phi(l)>=y)  l -= 1;
		r = l+1;
	} else {
		r = 0;
		while (Phi(r)<y)  r += 1;
		l = r-1;
	}

	do {
		double m = 0.5*(l+r);
		if (Phi(m) < y) {
			l = m;
		} else {
			r = m;
		}
	} while (r-l > 1e-8);
	return  0.5*(l+r);
}

/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
