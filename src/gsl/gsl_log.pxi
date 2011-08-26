cdef extern from "gsl/gsl_sf_log.h":

  double  gsl_sf_log(double x) nogil

  int  gsl_sf_log_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_log_abs(double x) nogil

  int  gsl_sf_log_abs_e(double x, gsl_sf_result * result) nogil

  int  gsl_sf_complex_log_e(double zr, double zi, gsl_sf_result * lnr, gsl_sf_result * theta) nogil

  double  gsl_sf_log_1plusx(double x) nogil

  int  gsl_sf_log_1plusx_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_log_1plusx_mx(double x) nogil

  int  gsl_sf_log_1plusx_mx_e(double x, gsl_sf_result * result) nogil

