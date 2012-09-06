/* EZ-diff.c - estimate some parameters using the EZ-model
 *
 * This implements the EZ-model of Wagenmakers, E.-J., van der Maas,
 * H. L. J., & Grasman, R. P. P. P. (2006)
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
#include <math.h>

#include "fast-dm.h"


int
sign(double value)
{
	return (value >= 0) ? 1 : -1;
}

double
logit(double value)
{
	return log(value / (1 - value));
}

int
make_stat(const struct samples *data, double *mean, double *var, double *pc)
{
	double sum = 0;
	int i;

	if (data->plus_used+data->minus_used <= 0)
		return -1;

	for (i = 0; i < data->plus_used; i++)
		sum += data->plus_data[i];
	for (i = 0; i < data->minus_used; i++)
		sum += data->minus_data[i];
	*mean = sum / (data->plus_used+data->minus_used);

	sum = 0;
	for (i = 0; i < data->plus_used; i++)
		sum += (data->plus_data[i]-*mean)*(data->plus_data[i]-*mean);
	for (i = 0; i < data->minus_used; i++)
		sum += (data->minus_data[i]-*mean)*(data->minus_data[i]-*mean);
	*var = sum / (data->plus_used+data->minus_used);

	*pc = (double)data->plus_used / (data->plus_used+data->minus_used);

	return 1;
}

int
EZ_par(const struct samples *data, double *a, double *v, double *t0)
/* Calculate estimates for some parameters of the diffusion  modell.
 *
 * Based on the mean and variance of the reaction times and the
 * percentage of "correct" responses (i.e. of 'plus_data'), estimates
 * for the threshold separation (a), the drift (v), and the
 * response-time constant (t0) are calculated.  The algorithm is
 * derived from Wagenmakers et al. (2006).
 */
{
	double RT_mean, RT_var, PC, L;

	if (make_stat(data, &RT_mean, &RT_var, &PC) < 0)
		return -1;

	if ((PC>0) && (PC<1))
		L = logit(PC);
	else if (PC==0)
		L =logit(PC + 1.0/(data->minus_used+data->plus_used));
	else if (PC==1)
		L =logit(PC - 1.0/(data->minus_used+data->plus_used));



	*v = sign(PC-0.5) * pow(L*(PC*PC*L-PC*L+PC-0.5) / RT_var, 0.25);
	if (fabs(PC-0.5) > 1e-4) {
		*a = L / *v;
		*t0 = RT_mean - (((*a)/(2*(*v)))*(1-exp(-(*v)*(*a)))
				    / (1+exp(-(*v)*(*a))));
	} else {
		*a = pow (24 * RT_var, 0.25);
		*t0 = RT_mean - 0.25*(*a)*(*a);
	}

	return 1;
}

/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
