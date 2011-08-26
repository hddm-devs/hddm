cdef extern from "gsl/gsl_sf_expint.h":

  double  gsl_sf_expint_E1(double x) nogil

  int  gsl_sf_expint_E1_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_expint_E2(double x) nogil

  int  gsl_sf_expint_E2_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_expint_Ei(double x) nogil

  int  gsl_sf_expint_Ei_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_Shi(double x) nogil

  int  gsl_sf_Shi_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_Chi(double x) nogil

  int  gsl_sf_Chi_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_expint_3(double x) nogil

  int  gsl_sf_expint_3_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_Si(double x) nogil

  int  gsl_sf_Si_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_Ci(double x) nogil

  int  gsl_sf_Ci_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_atanint(double x) nogil

  int  gsl_sf_atanint_e(double x, gsl_sf_result * result) nogil

