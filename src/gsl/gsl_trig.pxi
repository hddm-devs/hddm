cdef extern from "gsl/gsl_sf_trig.h":

  double  gsl_sf_sin(double x) nogil

  int  gsl_sf_sin_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_cos(double x) nogil

  int  gsl_sf_cos_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_hypot(double x, double y) nogil

  int  gsl_sf_hypot_e(double x, double y, gsl_sf_result * result) nogil

  double  gsl_sf_sinc(double x) nogil

  int  gsl_sf_sinc_e(double x, gsl_sf_result * result) nogil

  int  gsl_sf_complex_sin_e(double zr, double zi, gsl_sf_result * szr, gsl_sf_result * szi) nogil

  int  gsl_sf_complex_cos_e(double zr, double zi, gsl_sf_result * czr, gsl_sf_result * czi) nogil

  int  gsl_sf_complex_logsin_e(double zr, double zi, gsl_sf_result * lszr, gsl_sf_result * lszi) nogil

  double  gsl_sf_lnsinh(double x) nogil

  int  gsl_sf_lnsinh_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_lncosh(double x) nogil

  int  gsl_sf_lncosh_e(double x, gsl_sf_result * result) nogil

  int  gsl_sf_polar_to_rect(double r, double theta, gsl_sf_result * x, gsl_sf_result * y) nogil

  int  gsl_sf_rect_to_polar(double x, double y, gsl_sf_result * r, gsl_sf_result * theta) nogil

  double  gsl_sf_angle_restrict_symm(double theta) nogil

  int  gsl_sf_angle_restrict_symm_e(double * theta) nogil

  double  gsl_sf_angle_restrict_pos(double theta) nogil

  int  gsl_sf_angle_restrict_pos_e(double * theta) nogil

  double  gsl_sf_sin_err(double x, double dx) nogil

  int  gsl_sf_sin_err_e(double x, double dx, gsl_sf_result * result) nogil

  double  gsl_sf_cos_err(double x, double dx) nogil

  int  gsl_sf_cos_err_e(double x, double dx, gsl_sf_result * result) nogil

