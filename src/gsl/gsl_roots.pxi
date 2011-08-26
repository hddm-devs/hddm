cdef extern from "gsl/gsl_roots.h":
  
  ctypedef struct gsl_root_fsolver_type:
      char *name
      size_t size
      int (*set) (void *state, gsl_function * f, double * root, double x_lower, double x_upper) nogil
      int (*iterate) (void *state, gsl_function * f, double * root, double * x_lower, double * x_upper) nogil
  
  ctypedef struct gsl_root_fsolver:
      gsl_root_fsolver_type * type
      gsl_function * function 
      double root 
      double x_lower
      double x_upper
      void *state
  
  ctypedef struct gsl_root_fdfsolver_type:
      char *name
      size_t size
      int (*set) (void *state, gsl_function_fdf * f, double * root) nogil
      int (*iterate) (void *state, gsl_function_fdf * f, double * root) nogil
  
  ctypedef struct gsl_root_fdfsolver:
      gsl_root_fdfsolver_type * type
      gsl_function_fdf * fdf 
      double root 
      void *state
  
  gsl_root_fsolver * gsl_root_fsolver_alloc ( gsl_root_fsolver_type * T) nogil
  void gsl_root_fsolver_free (gsl_root_fsolver * s) nogil
  
  int gsl_root_fsolver_set (gsl_root_fsolver * s, gsl_function * f, 
                            double x_lower, double x_upper) nogil
  
  int gsl_root_fsolver_iterate (gsl_root_fsolver * s) nogil
  
  char * gsl_root_fsolver_name ( gsl_root_fsolver * s) nogil
  double gsl_root_fsolver_root ( gsl_root_fsolver * s) nogil
  double gsl_root_fsolver_x_lower ( gsl_root_fsolver * s) nogil
  double gsl_root_fsolver_x_upper ( gsl_root_fsolver * s) nogil
  
  
  gsl_root_fdfsolver * gsl_root_fdfsolver_alloc ( gsl_root_fdfsolver_type * T) nogil
  
  int gsl_root_fdfsolver_set (gsl_root_fdfsolver * s, 
                           gsl_function_fdf * fdf, double root) nogil
  
  int gsl_root_fdfsolver_iterate (gsl_root_fdfsolver * s) nogil
  
  void gsl_root_fdfsolver_free (gsl_root_fdfsolver * s) nogil
  
  char * gsl_root_fdfsolver_name ( gsl_root_fdfsolver * s) nogil
  double gsl_root_fdfsolver_root ( gsl_root_fdfsolver * s) nogil
  
  int gsl_root_test_interval (double x_lower, double x_upper, double epsabs, double epsrel) nogil
  
  int gsl_root_test_residual (double f, double epsabs) nogil
  
  int gsl_root_test_delta (double x1, double x0, double epsabs, double epsrel) nogil
  
  gsl_root_fsolver_type  * gsl_root_fsolver_bisection
  gsl_root_fsolver_type  * gsl_root_fsolver_brent
  gsl_root_fsolver_type  * gsl_root_fsolver_falsepos
  gsl_root_fdfsolver_type  * gsl_root_fdfsolver_newton
  gsl_root_fdfsolver_type  * gsl_root_fdfsolver_secant
  gsl_root_fdfsolver_type  * gsl_root_fdfsolver_steffenson
