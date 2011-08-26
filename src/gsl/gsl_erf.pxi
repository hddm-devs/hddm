cdef extern from "gsl/gsl_sf_erf.h":

  double  gsl_sf_erf(double x) nogil

  int  gsl_sf_erf_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_erfc(double x) nogil

  int  gsl_sf_erfc_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_log_erfc(double x) nogil

  int  gsl_sf_log_erfc_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_erf_Z(double x) nogil

  int  gsl_sf_erf_Z_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_erf_Q(double x) nogil

  int  gsl_sf_erf_Q_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_hazard(double x) nogil

  int  gsl_sf_hazard_e(double x, gsl_sf_result * result) nogil

