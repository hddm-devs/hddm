/* dataset.c - represent all the data for one run of the experiment
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
#include <string.h>
#include <errno.h>

#include "fast-dm.h"


/**********************************************************************
 * auxiliary functions
 */

static int
compare_doubles (const void *a, const void *b)
{
	const double *da = a;
	const double *db = b;

	if (*da < *db)  return -1;
	if (*da > *db)  return 1;
	return  0;
}

static void
str_add (char **s1, const char *s2, int *used, int *alloc)
{
	int  l = strlen(s2);

	if (*used+l+1 >= *alloc) {
		*alloc += 64;
		*s1 = xrenew(char, *s1, *alloc);
	}
	memcpy (*s1+*used, s2, l);
	*used += l;
	(*s1)[*used] = '\0';
}

/**********************************************************************
 * samples
 */

struct samples *
new_samples(const char *name)
{
	struct samples *s;

	s = xnew(struct samples, 1);
	s->name = xstrdup(name);
	s->plus_alloc = s->minus_alloc = 32;
	s->plus_data = xnew(double, s->plus_alloc);
	s->minus_data = xnew(double, s->minus_alloc);
	s->plus_used = s->minus_used = 0;

	return  s;
}

void
delete_samples(struct samples *s)
{
	xfree(s->plus_data);
	xfree(s->minus_data);
	xfree(s->name);
	xfree(s);
}

void
samples_add_sample (struct samples *samples, double t, int response)
{
	if (response) {
		if (samples->plus_used >= samples->plus_alloc) {
			samples->plus_alloc += 32;
			samples->plus_data = xrenew(double,
						    samples->plus_data,
						    samples->plus_alloc);
		}
		samples->plus_data [samples->plus_used++] = t;
	} else {
		if (samples->minus_used >= samples->minus_alloc) {
			samples->minus_alloc += 32;
			samples->minus_data = xrenew(double,
						     samples->minus_data,
						     samples->minus_alloc);
		}
		samples->minus_data [samples->minus_used++] = t;
	}
}

void
samples_sort (struct samples *samples)
{
	qsort (samples->plus_data, samples->plus_used,
	       sizeof(double), compare_doubles);
	qsort (samples->minus_data, samples->minus_used,
	       sizeof(double), compare_doubles);
}

/**********************************************************************
 * datasets
 */

struct dataset *
new_dataset (void)
{
	struct dataset *ds;

	ds = xnew(struct dataset, 1);

	ds->fname = NULL;
	ds->logname = NULL;
	ds->key = NULL;
	ds->precision = -1;

	ds->cmds_used = 0;
	ds->cmds_alloc = 16;
	ds->cmds = xnew(struct cmds, ds->cmds_alloc);

	ds->param = new_array();
	ds->z = new_array();

	ds->consts_used = 0;
	ds->consts_alloc = 8;
	ds->consts = xnew(double, ds->consts_alloc);

	ds->samples_used = 0;
	ds->samples_alloc = 8;
	ds->samples = xnew(struct samples *, ds->samples_alloc);

	return  ds;
}

void
delete_dataset (struct dataset *ds)
{
	int  i;

	for (i=0; i<ds->samples_used; ++i)
		delete_samples(ds->samples[i]);
	xfree(ds->samples);
	xfree(ds->consts);
	delete_array(ds->z);
	delete_array(ds->param);
	xfree(ds->cmds);
	xfree(ds->key);
	xfree(ds->logname);
	xfree(ds->fname);
	xfree(ds);
}

void
dataset_print (const struct dataset *ds)
/* Print details of the data set to stdout.
 *
 * Parameters are indexed for each experimental condition and sample
 * sizes per result RESULT (i.e., per response) are given for each
 * experimental condition.
 */
{
	char  buffer[64];
	int  pused, palloc;
	char *pstr;
	int  i;

	pused = 0;
	palloc = 80;
	pstr = xnew (char, palloc);
	*pstr = '\0';

	printf ("dataset %s:\n", ds->fname);
	for (i=0; i<ds->cmds_used; ++i) {
		int  arg1 = ds->cmds[i].arg1;
		int  arg2 = ds->cmds[i].arg2;
		switch (ds->cmds[i].cmd) {
		case c_copy_param:
			if (*pstr) str_add (&pstr, ", ", &pused, &palloc);
			if (arg1>=0)
				str_add (&pstr, ds->param->entry[arg2],
					 &pused, &palloc);
			else
				str_add (&pstr, ds->z->entry[arg2],
					 &pused, &palloc);
			break;
		case c_copy_const:
			snprintf (buffer, 64, "%g", ds->consts[arg2]);
			if (*pstr) str_add (&pstr, ", ", &pused, &palloc);
			str_add (&pstr, buffer, &pused, &palloc);
			break;
		case c_run:
			printf ("  %s (%d+%d samples)\n", pstr,
				ds->samples[arg1]->plus_used,
				ds->samples[arg1]->minus_used);
			pused = 0;
			*pstr = '\0';
			break;
		}
	}

	xfree (pstr);
}

void
dataset_print_commands (const struct dataset *ds)
{
	const char *pname[p_count];
	int  i;

	pname[p_a] = "a";
	pname[p_v] = "v";
	pname[p_t0] = "t0";
	pname[p_sz] = "sz";
	pname[p_sv] = "sv";
	pname[p_st0] = "st0";

	for (i=0; i<ds->cmds_used; ++i) {
		int  arg1 = ds->cmds[i].arg1;
		int  arg2 = ds->cmds[i].arg2;
		switch (ds->cmds[i].cmd) {
		case c_copy_param:
			if (arg1 >= 0) {
				printf ("  c%d: use %s as %s\n", i,
					ds->param->entry[arg2],
					pname[arg1]);
			} else {
				printf ("  c%d: use %s as z\n", i,
					ds->z->entry[arg2]);
			}
			break;
		case c_copy_const:
			if (arg1 >= 0) {
				printf ("  c%d: use %g as %s\n", i,
					ds->consts[arg2],
					pname[arg1]);
			} else {
				printf ("  c%d: use %g*a as z\n", i,
					ds->consts[arg2]);
			}
			break;
		case c_run:
			printf ("  c%d: analyse sample %s\n", i,
				ds->samples[arg1]->name);
			break;
		}
	}
}


int
dataset_samples_idx (struct dataset *ds, const char *name, int add)
/* Find and add sample sets.
 *
 * This returns the index of the sample set with name 'name' in the
 * dataset 'ds'.  If there is no such sample, a new one is created.
 */
{
	int  i;

	for (i=0; i<ds->samples_used; ++i) {
		if (strcmp (ds->samples[i]->name, name) == 0)  return i;
	}
	if (! add)  return -1;
	if (ds->samples_used >= ds->samples_alloc) {
		ds->samples_alloc += 8;
		ds->samples = xrenew(struct samples *, ds->samples,
				     ds->samples_alloc);
	}
	ds->samples[ds->samples_used] = new_samples(name);
	return  ds->samples_used++;
}

int
dataset_add_const (struct dataset *ds, double x)
{
	int  i;

	for (i=0; i<ds->consts_used; ++i) {
		if (ds->consts[i] == x)  return i;
	}
	if (ds->consts_used >= ds->consts_alloc) {
		ds->consts_alloc += 8;
		ds->consts = xrenew(double, ds->consts, ds->consts_alloc);
	}
	ds->consts[ds->consts_used] = x;
	return  ds->consts_used++;
}

int
dataset_add_param (struct dataset *ds, const char *name)
{
	int  i;

	for (i=0; i<ds->param->used; ++i) {
		if (strcmp (ds->param->entry[i], name) == 0)  return i;
	}
	if (ds->param->used >= ds->param->alloc) {
		ds->param->alloc += 16;
		ds->param->entry = xrenew(char *, ds->param->entry,
					   ds->param->alloc);
	}
	ds->param->entry[ds->param->used] = xstrdup(name);
	return  ds->param->used++;
}

int
dataset_add_z (struct dataset *ds, const char *name)
{
	int  i;

	for (i=0; i<ds->z->used; ++i) {
		if (strcmp (ds->z->entry[i], name) == 0)  return i;
	}
	if (ds->z->used >= ds->z->alloc) {
		ds->z->alloc += 4;
		ds->z->entry = xrenew(char *, ds->z->entry, ds->z->alloc);
	}
	ds->z->entry[ds->z->used] = xstrdup(name);
	return  ds->z->used++;
}

void
dataset_add_cmd (struct dataset *ds, enum cmd cmd, int arg1, int arg2)
{
	if (ds->cmds_used >= ds->cmds_alloc) {
		ds->cmds_alloc += 16;
		ds->cmds = xrenew(struct cmds, ds->cmds, ds->cmds_alloc);
	}
	ds->cmds[ds->cmds_used].cmd = cmd;
	ds->cmds[ds->cmds_used].arg1 = arg1;
	ds->cmds[ds->cmds_used].arg2 = arg2;
	ds->cmds_used++;
}

void
dataset_save_result (const struct dataset *ds,
		     const double *x, const double *z, double p, double time)
{
	FILE *fd;
	int  i;

	if (! ds->logname)  return;

	fd = fopen(ds->logname, "w");
	if (! fd) {
		const char *msg = strerror(errno);
		fprintf(stderr,
			"error: cannot open save file \"%s\": %s\n",
			ds->logname, msg);
		exit(1);
	}

	for (i=0; i<ds->z->used; ++i)
		fprintf (fd, "%s = %f\n",
			 ds->z->entry[i], z[i]);

	for (i=0; i<ds->param->used; ++i)
		fprintf (fd, "%s = %f\n",
			 ds->param->entry[i], x[i]);

	fprintf (fd, "precision = %f\n", ds->precision);
	fprintf (fd, "p = %f\n", p);
	fprintf (fd, "time = %f\n", time);
	fclose (fd);
}


/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
