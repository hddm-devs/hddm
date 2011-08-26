cdef extern from "gsl/gsl_sf_synchrotron.h":

  double  gsl_sf_synchrotron_1(double x) nogil

  int  gsl_sf_synchrotron_1_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_synchrotron_2(double x) nogil

  int  gsl_sf_synchrotron_2_e(double x, gsl_sf_result * result) nogil

