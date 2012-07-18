/* pde.c - numerically solve the Fokker-Planck equation
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

#include "fast-dm.h"


static void
solve_tridiag(int n, const double *rhs, double *res, double left,
	      double mid, double right)
/* Solve an n by n tridiagonal system of linear equations.
 *
 * The matrix has 'mid' on the diagonal, 'left' on the subdiagonal and
 * 'right' on the superdiagonal.
 */
{
	double *tmp, p, old_res, old_tmp;
	int  i;

	tmp = xnew(double, n-1);

	/* step 1: solving forward */
	tmp[0] = old_tmp = right / mid;
	res[0] = old_res = rhs[0] / mid;
	for (i=1; i<n-1; ++i) {
		p = 1.0/(mid-left*old_tmp);
		res[i] = old_res = (rhs[i] - left*old_res)*p;
		tmp[i] = old_tmp = right*p;
	}
	p = 1.0/(mid-left*old_tmp);
	res[n-1] = (rhs[n-1] - left*old_res)*p;

	/* step 2: solving backward */
	for (i=n-1; i>0; --i)  res[i-1] -= tmp[i-1]*res[i];
	xfree(tmp);
}

static void
make_step (int N, double *vector, double dt, double dz, double v)
/* Advance the numerical solution of the PDE by one step in time,
 * using the Crank-Nicolson scheme.  The time step size is 'dt', the
 * space grid size is 'dz'.  */
{
	double	*tmp_vector;
	double  left, mid, right;
	int	i;

	tmp_vector = xnew (double, N+1);

	left  =  (1-dz*v) / (2*dz*dz);
	mid   =  -1 / (dz*dz);
	right =  (1+dz*v) / (2*dz*dz);

	tmp_vector[1] = (dt*left * vector[0] +
			 (1+0.5*dt*mid) * vector[1] +
			 0.5*dt*right * vector[2]);
	for (i=2; i<N-1; i++) {
		tmp_vector[i] = (0.5*dt*left * vector[i-1] +
				 (1+0.5*dt*mid) * vector[i] +
				 0.5*dt*right * vector[i+1]);
	}
	tmp_vector[N-1] = (0.5*dt*left * vector[N-2] +
			   (1+0.5*dt*mid) * vector[N-1] +
			   dt*right * vector[N]);

	solve_tridiag(N-1, tmp_vector+1, vector+1,
		      -0.5*dt*left, 1-0.5*dt*mid, -0.5*dt*right);

	xfree (tmp_vector);
}

void
advance_to (int N, double *vector, double t0, double t1, double dz, double v)
/* Advance the state 'vector' of the PDE from time 't0' to time 't1' */
{
	int  done = 0;

	do {
		double  dt = TUNE_PDE_DT_MIN + TUNE_PDE_DT_SCALE*t0;
		if (dt > TUNE_PDE_DT_MAX)  dt = TUNE_PDE_DT_MAX;
		if (t0 + dt >= t1) {
			dt = t1 - t0;
			done = 1;
		}
		make_step (N, vector, dt, dz, v);
		t0 += dt;
	} while (! done);
}

/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
