cdef extern from "gsl/gsl_poly.h":

  #  Evaluate polynomial
  double  gsl_poly_eval(double c[], int len, double x) nogil

  # divided-difference polynomials
  int  gsl_poly_dd_init(double dd[], double xa[], double ya[], size_t size) nogil

  double  gsl_poly_dd_eval(double dd[], double xa[], size_t size, double x) nogil

  int  gsl_poly_dd_taylor(double c[], double xp, double dd[], double xa[], size_t size, double w[]) nogil

  # quadratic equation
  int  gsl_poly_solve_quadratic(double a, double b, double c, double *x0, double *x1) nogil

  int  gsl_poly_complex_solve_quadratic(double a, double b, double c, gsl_complex *z0, gsl_complex *z1) nogil

  # cubic equation
  int  gsl_poly_solve_cubic(double a, double b, double c, double *x0, double *x1, double *x2) nogil

  int  gsl_poly_complex_solve_cubic(double a, double b, double c, gsl_complex *z0, gsl_complex *z1, gsl_complex *z2) nogil

  # General Polynomial Equations
  ctypedef struct gsl_poly_complex_workspace:
    size_t nc
    double * matrix

  gsl_poly_complex_workspace *  gsl_poly_complex_workspace_alloc(size_t n) nogil

  void  gsl_poly_complex_workspace_free(gsl_poly_complex_workspace * w) nogil

  int  gsl_poly_complex_solve(double * a, size_t n, gsl_poly_complex_workspace * w, gsl_complex_packed_ptr z) nogil


