cdef extern from "gsl/gsl_sf_exp.h":

  double  gsl_sf_exp(double x) nogil

  int  gsl_sf_exp_e(double x, gsl_sf_result * result) nogil

  int  gsl_sf_exp_e10_e(double x, gsl_sf_result_e10 * result) nogil

  double  gsl_sf_exp_mult(double x, double y) nogil

  int  gsl_sf_exp_mult_e(double x, double y, gsl_sf_result * result) nogil

  int  gsl_sf_exp_mult_e10_e(double x, double y, gsl_sf_result_e10 * result) nogil

  double  gsl_sf_expm1(double x) nogil

  int  gsl_sf_expm1_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_exprel(double x) nogil

  int  gsl_sf_exprel_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_exprel_2(double x) nogil

  int  gsl_sf_exprel_2_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_exprel_n(int n, double x) nogil

  int  gsl_sf_exprel_n_e(int n, double x, gsl_sf_result * result) nogil

  int  gsl_sf_exp_err_e(double x, double dx, gsl_sf_result * result) nogil

  int  gsl_sf_exp_err_e10_e(double x, double dx, gsl_sf_result_e10 * result) nogil

  int  gsl_sf_exp_mult_err_e(double x, double dx, double y, double dy, gsl_sf_result * result) nogil

  int  gsl_sf_exp_mult_err_e10_e(double x, double dx, double y, double dy, gsl_sf_result_e10 * result) nogil

