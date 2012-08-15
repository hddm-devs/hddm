/* fast-dm.h - global header file for the PDE project
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

#include <stddef.h>


#if __GNUC__ >= 3
#define  jv_pure  __attribute__ ((pure))
#define  jv_const  __attribute__ ((const))
#define  jv_noreturn  __attribute__ ((noreturn))
#define  jv_malloc  __attribute__ ((malloc))
#define  jv_printf  __attribute__ ((format (printf, 1, 2)))
#define  jv_fprintf  __attribute__ ((format (printf, 2, 3)))
#define  jv_must_check  __attribute__ ((warn_unused_result))
#define  jv_unused  __attribute__ ((unused))
#else
#define  jv_pure  /* no pure */
#define  jv_const  /* no const */
#define  jv_noreturn  /* no noreturn */
#define  jv_malloc  /* no malloc */
#define  jv_printf  /* no printf */
#define  jv_fprintf  /* no printf */
#define  jv_must_check  /* no warn_unused_result */
#define  jv_unused  /* no unused */
#endif


/* windows compatibility hacks */

#ifdef _WIN32

/* provided in "win32erf.c" */
extern  double  erf (double x);

#define snprintf _snprintf

#endif	/* _WIN32 */


/* from "xmalloc.c" */

extern  void *xmalloc (size_t size) jv_malloc;
#define xnew(T,N) ((T *)xmalloc((N)*sizeof(T)))
extern  void *xrealloc (void *ptr, size_t newsize);
#define xrenew(T,OLD,N) ((T *)xrealloc(OLD,(N)*sizeof(T)))
extern  void  xfree (void *ptr);
extern  char *xstrdup (const char *s) jv_malloc;
extern  char *xstrndup (const char *s, size_t n) jv_malloc;

/* from "container.c" */

struct set {
	int  used, alloc;
	char **item;
};

extern  struct set *new_set (void);
extern  void  delete_set (struct set *set);
extern  int  set_item (struct set *set, const char *item, int add);

struct dict;

extern  struct dict *new_dict (void);
extern  void  delete_dict (struct dict *dict);
extern  void  dict_add (struct dict *dict, const char *key, const char *value);
extern  const char *dict_lookup (const struct dict *dict, const char *key);
extern  void  dict_clear (struct dict *dict);

struct array {
	int  used, alloc;
	char **entry;
};
extern  struct array *new_array (void);
extern  void  delete_array (struct array *array);
extern  void  array_clear (struct array *array);
extern  int  array_find (struct array *array, const char *str);
extern  void  array_append (struct array *array, const char *str);
extern  void  array_sort (struct array *array);

/* from "file.c" */

extern  struct array *file_names_find (const char *pattern);
extern  char *file_names_replace_star(const char *template, const char *key);

struct file;

extern  struct file *new_file (const char *fname);
extern  void  delete_file (struct file *file);
extern  void  file_error (struct file *file, const char *format, ...)
	jv_fprintf jv_noreturn;
extern  void  file_message (struct file *file, const char *format, ...)
	jv_fprintf;
extern  const char *file_name (const struct file *file);
extern  int  file_read (struct file *file,
			const char *const**w_ptr, int *n_ptr);

/* from "dataset.c" */

enum parameter_index {
	p_a,
	p_v,
	p_t0,
	p_sz,
	p_sv,
	p_st0,
	p_count
};

struct samples {
	char *name;

	int  plus_alloc, minus_alloc;
	int  plus_used, minus_used;
	double *plus_data, *minus_data;
};

extern  struct samples *new_samples (const char *name);
extern  void  delete_samples (struct samples *s);
extern  void  samples_add_sample (struct samples *samples, double t, int res);
extern  void  samples_sort (struct samples *samples);

struct dataset {
	char *fname, *logname, *key;
	double  precision;

	/* commands for initialisation */
	int  cmds_used, cmds_alloc;
	struct cmds {
		enum cmd { c_copy_param, c_copy_const, c_run } cmd;
		int  arg1, arg2;
	} *cmds;

	/* names of the parameters optimised by the simplex algorithm */
	struct array *param;

	/* names of the z-parameters optimised by direct search */
	struct array *z;

	/* constants for initialisation of fixed parameters */
	int  consts_used, consts_alloc;
	double *consts;

	/* reaction time data, split by experimental condition */
	int  samples_used, samples_alloc;
	struct samples **samples;
};

extern  struct dataset *new_dataset (void);
extern  void  delete_dataset (struct dataset *dataset);
extern  void  dataset_print (const struct dataset *dataset);
extern  void  dataset_print_commands (const struct dataset *dataset);
extern  int  dataset_samples_idx (struct dataset *dataset,
				  const char *name, int add);
extern  int  dataset_add_const (struct dataset *dataset, double x);
extern  int  dataset_add_param (struct dataset *dataset, const char *name);
extern  int  dataset_add_z (struct dataset *dataset, const char *name);
extern  void  dataset_add_cmd (struct dataset *dataset,
			       enum cmd cmd, int arg1, int arg2);
extern  void  dataset_save_result (const struct dataset *datatset,
				   const double *x, const double *z,
				   double p, double time);

/* from "experiment.c" */

struct experiment;
extern  struct experiment *new_experiment(const char *fname);
extern  void  delete_experiment (struct experiment *ex);
extern  void  experiment_print (const struct experiment *ex);

extern  struct dataset *experiment_get_dataset (struct experiment *li,
						int continue_flag);
extern  void  experiment_log (struct experiment *ex, const struct dataset *ds,
			      double *values, double  *z, double p,
			      double time);

/* from "pde.c" */

extern  void  advance_to (int N, double *vector, double t0, double t1,
			  double dz, double v);

/* from "phi.c" */

extern  double  Phi (double x);
extern  double  Phi_inverse (double y);

/* from "cdf.c" */

extern double TUNE_PDE_DT_MIN;
extern double TUNE_PDE_DT_MAX;
extern double TUNE_PDE_DT_SCALE;
extern double TUNE_DZ;
extern double TUNE_DV;
extern double TUNE_DT0;

extern  void  set_precision (double p);


struct F_calculator;
enum boundary { b_lower=0, b_upper=1 };

extern  struct F_calculator *F_new (const double *para);
extern  void  F_delete (struct F_calculator *fc);
extern  void  F_start (struct F_calculator *fc, enum boundary);
extern  int  F_get_N (const struct F_calculator *fc);
extern  double  F_get_z (const struct F_calculator *fc, int i);
extern  const double *F_get_F (struct F_calculator *fc, double t);
extern  double  F_get_val (struct F_calculator *fc, double t, double z);

/* from "simplex.c" */

extern  double  simplex (int n, double *x, const double *eps,
			 double size_limit, void *data,
			 double (*fn)(const double *x, void *data));

/* from "simplex2.c" */

extern  void  simplex2(int n, const double *eps, double size_limit,
		       void (*fn)(const double *x, double res[2], void *data),
		       double *x_ret, double *fn_ret, void *data);

/* from "EZ-diff.c" */

extern  int  EZ_par (const struct samples *data,
		     double *a, double *v, double *t0);

/* from "main.c" */

struct distance {
	int  N;
	double *T;		/* array of distances (length N+1) */
	double  zmin, zmax;	/* z-values for T[0] and T[N] */
};

extern  struct distance *new_distance(int N, double zmin, double zmax);
extern  void  delete_distance(struct distance *d);

/* from "method-ks.c" */

extern struct distance *KS_get_distance(const struct samples *samples,
					const double *para);
extern double KS_T_to_p(double T, const struct samples *s);

/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
