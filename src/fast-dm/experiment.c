/* experiment.c - read the experiment control files
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

#include "fast-dm.h"


/**********************************************************************
 * auxiliary functions
 */

static int
string_to_int (const char *str, long int *res)
/* return 1 on success and 0 on error */
{
	char *tail;
	long int  x;

	x = strtol (str, &tail, 0);
	if (tail == str)  return 0;
	if (*tail != '\0')  return 0;
	*res = x;
	return 1;
}

static int
string_to_double (const char *str, double *res)
/* return 1 on success and 0 on error */
{
	char *tail;
	double  x;

	x = strtod (str, &tail);
	if (tail == str)  return 0;
	if (*tail != '\0')  return 0;
	*res = x;
	return 1;
}

static void
check_templates (const char *load_template, const char *save_template)
{
	int  has_load_star, has_save_star;

	if (! save_template)  return;

	has_load_star = (strchr(load_template, '*') != NULL);
	has_save_star = (strchr(save_template, '*') != NULL);

	if (has_load_star && has_save_star)  return;
	if (has_load_star && ! save_template)  return;
	if (! has_load_star && ! has_save_star)  return;

	fprintf (stderr,
		 "incompatible templates '%s' and '%s', aborting\n",
		 load_template, save_template);
	exit(1);
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
 * information about one experiment
 */

struct param_info {
	const char *name;
	int  idx;
	struct set *depends;
	char *value;
};

static const  struct param_info  default_params[] = {
	{ "a", p_a, NULL, NULL},
	{ "z", -1, NULL, NULL},
	{ "v", p_v, NULL, NULL},
	{ "t0", p_t0, NULL, NULL},
	{ "sz", p_sz, NULL, NULL},
	{ "sv", p_sv, NULL, NULL},
	{ "st0", p_st0, NULL, NULL},
	{ NULL, 0, NULL, NULL}
};

struct experiment {
	char *fname;

	/* precision for computing the CDF */
	double  precision;

	/* model parameters */
	struct param_info *params;

	/* experimental conditions */
	struct set *cond;

	/* data files */
	char *data_template;
	struct array *files;
	int  current_file;

	/* data file fields */
	int nf;
	struct fi {
		enum fi_type { t_time, t_response, t_ignore, t_cond } type;
		int cond_idx;
	} *fi;

	/* per data-set log files */
	char *save_template;

	/* common log file */
	char *log_file_name;
	struct array *log_fields;
};

struct experiment *
new_experiment (const char *fname)
/* Read information for a new experiment from the input file 'fname'
 * and return it as a structure.  The format of experiment control
 * files is described in the users' guide (file "MANUAL" of the source
 * code distribution)
 */
{
	struct file *file;
	struct experiment *ex;
	struct param_info *params;
	const char *const *tokens;
	int depends_seen = 0;
	int format_seen = 0;
	int i, j, n;

	file = new_file (fname);
	if (!file)
		return NULL;

	ex = xnew(struct experiment, 1);
	ex->fname = xstrdup(fname);
	ex->precision = 3;
	ex->params = params = xmalloc(sizeof (default_params));
	memcpy (params, default_params, sizeof (default_params));
	for (i = 0; params[i].name; ++i) {
		params[i].depends = new_set ();
	}
	ex->cond = new_set();
	ex->data_template = NULL;
	ex->save_template = NULL;
	ex->log_file_name = NULL;
	ex->log_fields = NULL;

	while (file_read (file, &tokens, &n)) {
		if (strcmp (tokens[0], "set") == 0) {
			if (depends_seen)
				file_error (file, "'set' after 'depends'");
			if (n != 3)
				file_error (file, "syntax error");

			for (i = 0; params[i].name; ++i) {
				if (strcmp (params[i].name, tokens[1]) == 0)
					break;
			}
			if (!params[i].name)
				file_error (file, "invalid parameter '%s'",
					    tokens[1]);
			if (params[i].value)
				file_error (file, "parameter '%s' set twice",
					    tokens[1]);
			params[i].value = xstrdup(tokens[2]);
		} else if (strcmp (tokens[0], "depends") == 0) {
			if (format_seen)
				file_error (file, "'depends' after 'format'");
			if (n < 3)
				file_error (file, "not enough arguments");

			for (i = 0; params[i].name; ++i) {
				if (strcmp (params[i].name, tokens[1]) == 0)
					break;
			}
			if (!params[i].name)
				file_error (file, "invalid parameter '%s'",
					    tokens[1]);
			if (params[i].value)
				file_error (file, "parameter '%s' already set",
					    tokens[1]);
			for (j = 2; j < n; ++j) {
				if (strcmp(tokens[j],"RESPONSE") == 0
				    || strcmp(tokens[j],"TIME") == 0
				    || strcmp(tokens[j],"*") == 0) {
					file_error (file,
						    "invalid condition '%s'",
						    tokens[j]);
				}
				set_item (params[i].depends, tokens[j], 1);
				set_item (ex->cond, tokens[j], 1);
			}
		} else if (strcmp (tokens[0], "format") == 0) {
			struct set *seen = new_set ();

			if (ex->data_template)
				file_error (file, "'format' after 'load'");
			if (format_seen)
				file_error (file, "more than one 'format'");
			format_seen = 1;

			ex->nf = n-1;
			ex->fi = xnew(struct fi, ex->nf);
			for (i = 1; i < n; ++i) {
				const char *field = tokens[i];
				enum fi_type  ft = t_ignore;

				j = set_item (ex->cond, field, 0);
				if (strcmp (field, "TIME") == 0) {
					ft = t_time;
				} else if (strcmp (field, "RESPONSE") == 0) {
					ft = t_response;
				} else if (j>=0) {
					ft = t_cond;
				}
				ex->fi[i-1].type = ft;
				if (ft == t_ignore)  continue;
				if (set_item (seen, field, 0) >= 0)
					file_error (file, "duplicate field %s",
						    field);
				set_item (seen, field, 1);
				if (ft == t_cond)
					ex->fi[i-1].cond_idx = j;
			}
			if (set_item (seen, "TIME", 0) < 0)
				file_error (file, "missing field 'TIME'");
			if (set_item (seen, "RESPONSE", 0) < 0)
				file_error (file, "missing field 'RESPONSE'");
			for (i=0; i<ex->cond->used; ++i) {
				if (set_item (seen, ex->cond->item[i], 0) < 0)
					file_error (file,
						    "unknown condition '%s'",
						    ex->cond->item[i]);
			}
			delete_set (seen);
		} else if (strcmp (tokens[0], "load") == 0) {
			if (! format_seen)
				file_error (file,
					    "data format not specified");
			if (ex->data_template)
				file_error (file, "more than one 'load'");
			if (n != 2)
				file_error (file, "wrong number of arguments");
			ex->data_template = xstrdup(tokens[1]);
		} else if (strcmp (tokens[0], "save") == 0) {
			if (n != 2)
				file_error (file, "wrong number of arguments");
			ex->save_template = xstrdup(tokens[1]);
		} else if (strcmp (tokens[0], "log") == 0) {
			if (n != 2)
				file_error (file, "wrong number of arguments");
			ex->log_file_name = xstrdup(tokens[1]);
		} else if (strcmp (tokens[0], "precision") == 0) {
			if (n != 2)
				file_error (file, "wrong number of arguments");
			if (! string_to_double (tokens[1], &ex->precision)
			    || ex->precision < 1)
				file_error (file, "invalid precision '%s'",
					    tokens[1]);
		} else {
			file_error (file, "unknown command '%s'", tokens[0]);
		}
	}
	if (! ex->data_template)
		file_error (file, "missing 'load'");
	if (! (ex->save_template || ex->log_file_name)) {
		file_error (file, "missing 'save' or 'log'");
	}
	check_templates (ex->data_template, ex->save_template);
	ex->files = file_names_find(ex->data_template);
	ex->current_file = -1;

	delete_file (file);

	return ex;
}

void
delete_experiment (struct experiment *ex)
{
	int i;

	if (ex->log_fields)
		delete_array(ex->log_fields);
	xfree(ex->log_file_name);
	xfree(ex->save_template);
	xfree(ex->fi);
	delete_array(ex->files);
	xfree(ex->data_template);
	delete_set(ex->cond);
	for (i = 0; ex->params[i].name; ++i) {
		delete_set(ex->params[i].depends);
		xfree(ex->params[i].value);
	}
	xfree(ex->params);
	xfree(ex->fname);
	xfree(ex);
}

void
experiment_print (const struct experiment *ex)
/* Print all information about the experiment "ex" to stdout.
 *
 * In detail, the names and the format of the data files are printed.
 * Additionally, it is specified for each parameter of the diffusion
 * model whether it is being optimised, or whether it is fixed to a
 * given value, and--where necessary--on which condition-variables of
 * the data file it depends.
 */
{
	int  i, j, first;

	printf ("experiment %s (%d data sets):\n",
		ex->fname, ex->files->used);

	printf ("  precision: %g\n", ex->precision);

	printf ("  format of \"%s\":", ex->data_template);
	for (i=0; i<ex->nf; ++i) {
		switch (ex->fi[i].type) {
		case t_time:
			printf (" TIME");
			break;
		case t_response:
			printf (" RESPONSE");
			break;
		case t_ignore:
			printf (" *");
			break;
		case t_cond:
			printf (" %s", ex->cond->item[ex->fi[i].cond_idx]);
			break;
		}
	}
	putchar ('\n');

	first = 1;
	for (i=0; ex->params[i].name; ++i) {
		if (ex->params[i].value)  continue;
		if (first) {
			printf ("  optimised parameters: ");
			first = 0;
		} else {
			printf (", ");
		}
		printf ("%s", ex->params[i].name);
		for (j=0; j<ex->params[i].depends->used; ++j) {
			printf ("_%s",ex->params[i].depends->item[j]);
		}
	}
	if (! first)  putchar ('\n');

	first = 1;
	for (i=0; ex->params[i].name; ++i) {
		if (! ex->params[i].value)  continue;
		if (first) {
			printf ("  fixed parameters: ");
			first = 0;
		} else {
			printf (", ");
		}
		printf ("%s=%s", ex->params[i].name, ex->params[i].value);
	}
	if (! first)  putchar ('\n');
}

static char *
experiment_get_data_name(const struct experiment *ex)
{
	const char *key;

	key = ex->files->entry[ex->current_file];
	return  file_names_replace_star(ex->data_template, key);
}

static char *
experiment_get_save_name(const struct experiment *ex)
{
	const char *key = ex->files->entry[ex->current_file];
	if (! ex->save_template)  return 0;

	return file_names_replace_star(ex->save_template, key);
}

static struct file *
experiment_open_dataset(struct experiment *ex)
{
	char *fname;
	struct file *file;

 retry:
	ex->current_file += 1;
	if (ex->current_file >= ex->files->used)  return NULL;

	fname = experiment_get_data_name(ex);
	file = new_file (fname);
	if (! file) {
		fprintf(stderr, "warning: failed to open file %s\n", fname);
		xfree(fname);
		goto retry;
	}
	xfree(fname);

	return  file;
}

static void
dataset_init_param (struct dataset *ds, struct param_info *param,
		    const struct dict *condv)
/* Add the necessary commands to 'ds' to initalise parameter 'param'.
 *
 * The dictionary 'condv' maps parameter names to experimental conditions
 * where required.
 */
{
	int  name_used, name_alloc;
	char *name;
	int  j;

	name_alloc = 64;
	name = xnew(char, name_alloc);
	name_used = 0;
	*name = '\0';

	str_add (&name, param->name, &name_used, &name_alloc);
	for (j=0; j<param->depends->used; ++j) {
		const char *cv;

		cv = dict_lookup (condv, param->depends->item[j]);
		str_add (&name, "_", &name_used, &name_alloc);
		str_add (&name, cv, &name_used, &name_alloc);
	}

	if (param->value) {
		double  x;
		if (! string_to_double (param->value, &x)) {
			fprintf (stderr, "invalid value '%s' for '%s'\n",
				 param->value, name);
			exit (1);
		}
		dataset_add_cmd (ds, c_copy_const, param->idx,
				 dataset_add_const (ds, x));
	} else {
		if (param->idx >= 0) {
			dataset_add_cmd (ds, c_copy_param, param->idx,
					 dataset_add_param (ds, name));
		} else {
			dataset_add_cmd (ds, c_copy_param, param->idx,
					 dataset_add_z (ds, name));
		}
	}
	xfree(name);
}

struct dataset *
experiment_get_dataset (struct experiment *ex, int continue_flag)
/* Read the next dataset.
 *
 * The returned data-set structure contains the parameters to be
 * optimised ('param'), values for the parameters fixed to a given
 * constant ('consts'), and arrays of responses and reaction times
 * ('samples').  If different experimental conditions are defined for
 * the experiment, different samples are allocated within the returned
 * structure.
 *
 * If 'continue_flag' is non-zero and save files are used, skip data
 * sets where a save file is already present.
 *
 * If the data file does not fit the format specified in 'ex', an
 * error message is printed to stderr and the next valid file is
 * returned instead.  If no more data sets are available, return NULL.
 */
{
	struct dataset *ds;
	struct file *file;
	const char *key;
	const char *const* words;
	struct dict *condv;
	int  sample_name_used, sample_name_alloc;
	char *sample_name;
	int  valid;
	int  i, n;

 retry:
	valid = 0;
	file = experiment_open_dataset (ex);
	if (! file)  return NULL;

	ds = new_dataset ();
	ds->fname = xstrdup(file_name (file));
	ds->logname = experiment_get_save_name (ex);
	key = ex->files->entry[ex->current_file];
	ds->key = xstrdup(*key ? key : ds->fname);
	ds->precision = ex->precision;

	if (continue_flag && ds->logname) {
		FILE *test = fopen(ds->logname, "r");
		if (test) {
			fclose(test);
			file_message(file, "skipping since %s is present",
				     ds->logname);
			goto abort2;
		}
	}

	condv = new_dict ();

	sample_name_alloc = 80;
	sample_name = xnew(char, sample_name_alloc);
	sample_name_used = 1;
	sample_name[0] = '\0';

	valid = 1;
	while (file_read (file, &words, &n) && valid) {
		struct samples *samples;
		double  t;
		long int  resp;
		int  idx;

		/* analyse a line from the data file */
		if (n != ex->nf) {
			file_message (file, "wrong number of fields");
			valid = 0;
			goto abort;
		}
		dict_clear (condv);
		for (i=0; i<n; ++i) {
			switch (ex->fi[i].type) {
			case t_time:
				if (! string_to_double(words[i], &t)) {
					file_message (file,
						      "invalid number '%s'",
						      words[i]);
					valid = 0;
					goto abort;
				}
				break;
			case t_response:
				if (! string_to_int(words[i], &resp)
				    || (resp != 0 && resp != 1)) {
					file_message (file,
						      "invalid response '%s'",
						      words[i]);
					valid = 0;
					goto abort;
				}
				break;
			case t_ignore:
				break;
			case t_cond:
				idx = ex->fi[i].cond_idx;
				dict_add (condv, ex->cond->item[idx],
					  words[i]);
				break;
			}
		}

		/* get the sample set name */
		sample_name_used = 0;
		sample_name[0] = '\0';
		str_add (&sample_name, ds->fname,
			 &sample_name_used, &sample_name_alloc);
		for (i=0; i<ex->cond->used; ++i) {
			const char *val;

			val = dict_lookup (condv, ex->cond->item[i]);
			str_add (&sample_name, "_",
				 &sample_name_used, &sample_name_alloc);
			str_add (&sample_name, val,
				 &sample_name_used, &sample_name_alloc);
		}

		/* create new sample sets as needed */
		idx = dataset_samples_idx (ds, sample_name, 0);
		if (idx < 0) {
			idx = dataset_samples_idx (ds, sample_name, 1);
			for (i=0; ex->params[i].name; ++i) {
				dataset_init_param (ds, ex->params+i, condv);
			}
			dataset_add_cmd (ds, c_run, idx, 0);
		}
		samples = ds->samples[idx];

		/* register the sample data */
		samples_add_sample (samples, t, resp);
	}

 abort:
	xfree(sample_name);
	delete_dict(condv);
 abort2:
	delete_file(file);
	if (! valid) {
		delete_dataset(ds);
		goto retry;
	}

	for (i=0; i<ds->samples_used; ++i)
		samples_sort (ds->samples[i]);

	return  ds;
}

static int
file_find_headers (char *fname, struct array *fields)
/* Search through the file 'fname' for the last header line.
 *
 * The header fields are stored in the array 'fields'.
 *
 * Return '-1' if the file does not exist, '0' if no (correct) header is
 * found, and '1', if everything is ok.
 */
{
	struct file *save;
	const char *const* words;
	int  n, found;

	if (!(save = new_file(fname)))
		return -1;

	found = 0;
	while (file_read (save, &words, &n)) {
		int  i;

		if (strcmp(words[0], "dataset") != 0 || n<2)  continue;

		found = 1;
		array_clear (fields);
		for (i=0; i<n; ++i)
			array_append (fields, words[i]);
	}
	delete_file(save);

	return  found;
}

static void
dataset_default_headers (const struct dataset *ds, struct array *fields)
{
	int i;

	array_clear(fields);
	array_append(fields, "dataset");
	for (i=0; i<ds->z->used; ++i){
		array_append(fields, ds->z->entry[i]);
	}
	for (i=0; i<ds->param->used; ++i) {
		array_append(fields, ds->param->entry[i]);
	}
	array_append(fields, "p");
	array_append(fields, "time");
}

static int
dataset_get_value (const struct dataset *ds, const char *name,
		   const double *param, const double *z,
		   double p, double time, double *value_ret)
{
	int  i;

	if (strcmp(name, "p") == 0) {
		*value_ret = p;
		return  1;
	} else if (strcmp(name, "time") ==0 ) {
		*value_ret = time;
		return  1;
	}

	for (i=0; i<ds->param->used; i++) {
		if (strcmp(name, ds->param->entry[i])==0) {
			*value_ret = param[i];
			return  1;
		}
	}

	for (i=0; i<ds->z->used; i++) {
		if (strcmp(name, ds->z->entry[i])==0) {
			*value_ret = z[i];
			return  1;
		}
	}

	return  0;
}

void
experiment_log (struct experiment *ex, const struct dataset *ds,
		double *values, double  *z, double p, double time)
/* Save the results of the parameter estimation to the global log file.
 *
 * If 'ex->log_file_name' is defined, values for all Parameters are
 * appended to the output file.
 */
{
	struct array *fields;
	int  needs_header_check = 1;
	int  needs_header_line = 0;
	FILE *fd;
	double  value;
	int  i;

	if (! ex->log_file_name)  return;

	if (! ex->log_fields) {
		ex->log_fields = new_array();
		if (! file_find_headers(ex->log_file_name,
					ex->log_fields)) {
			dataset_default_headers (ds, ex->log_fields);
			needs_header_check = 0;
			needs_header_line = 1;
		}
	}

	fields = ex->log_fields;

	if (needs_header_check) {
		int  bad_header = 0;
		struct array *target;

		target = new_array ();
		dataset_default_headers (ds, target);
		for (i=0; i<target->used; ++i) {
			if (array_find(fields,target->entry[i]) == -1) {
				bad_header = 1;
				break;
			}
		}

		if (bad_header) {
			delete_array (ex->log_fields);
			ex->log_fields = target;
			fields = target;
			needs_header_line = 1;
		} else {
			delete_array (target);
		}
	}

	fd = fopen (ex->log_file_name, "a");
	if (! fd) {
		fprintf(stderr, "error: cannot open log file %s\n",
			ex->log_file_name);
		exit(1);
	}
	if (needs_header_line) {
		int i;

		for (i=0; i<fields->used; ++i)
			fprintf (fd, " %7s", fields->entry[i]);
		fputc ('\n', fd);
	}

	for (i=0; i<fields->used; i++) {
		if (strcmp(fields->entry[i], "dataset") == 0) {
			fprintf (fd, " %7s", ds->key);
		} else {
			int  found;
			found = dataset_get_value (ds, fields->entry[i],
						   values, z, p, time, &value);
			if (found)
				fprintf (fd, " %7.3f", value);
			else
				fprintf (fd, " %7s", "-");
		}
	}
	fputc ('\n', fd);

	fclose (fd);
}

/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
