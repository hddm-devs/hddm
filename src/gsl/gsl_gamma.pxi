cdef extern from "gsl/gsl_sf_gamma.h":

  double  gsl_sf_gamma(double x) nogil

  int  gsl_sf_gamma_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_lngamma(double x) nogil

  int  gsl_sf_lngamma_e(double x, gsl_sf_result * result) nogil

  int  gsl_sf_lngamma_sgn_e(double x, gsl_sf_result * result_lg, double * sgn) nogil

  double  gsl_sf_gammastar(double x) nogil

  int  gsl_sf_gammastar_e(double x, gsl_sf_result * result) nogil

  double  gsl_sf_gammainv(double x) nogil

  int  gsl_sf_gammainv_e(double x, gsl_sf_result * result) nogil

  int  gsl_sf_lngamma_complex_e(double zr, double zi, gsl_sf_result * lnr, gsl_sf_result * arg) nogil

  double  gsl_sf_taylorcoeff(int n, double x) nogil

  int  gsl_sf_taylorcoeff_e(int n, double x, gsl_sf_result * result) nogil

  double  gsl_sf_fact(unsigned int n) nogil

  int  gsl_sf_fact_e(unsigned int n, gsl_sf_result * result) nogil

  double  gsl_sf_doublefact(unsigned int n) nogil

  int  gsl_sf_doublefact_e(unsigned int n, gsl_sf_result * result) nogil

  double  gsl_sf_lnfact(unsigned int n) nogil

  int  gsl_sf_lnfact_e(unsigned int n, gsl_sf_result * result) nogil

  double  gsl_sf_lndoublefact(unsigned int n) nogil

  int  gsl_sf_lndoublefact_e(unsigned int n, gsl_sf_result * result) nogil

  double  gsl_sf_choose(unsigned int n, unsigned int m) nogil

  int  gsl_sf_choose_e(unsigned int n, unsigned int m, gsl_sf_result * result) nogil

  double  gsl_sf_lnchoose(unsigned int n, unsigned int m) nogil

  int  gsl_sf_lnchoose_e(unsigned int n, unsigned int m, gsl_sf_result * result) nogil

  double  gsl_sf_poch(double a, double x) nogil

  int  gsl_sf_poch_e(double a, double x, gsl_sf_result * result) nogil

  double  gsl_sf_lnpoch(double a, double x) nogil

  int  gsl_sf_lnpoch_e(double a, double x, gsl_sf_result * result) nogil

  int  gsl_sf_lnpoch_sgn_e(double a, double x, gsl_sf_result * result, double * sgn) nogil

  double  gsl_sf_pochrel(double a, double x) nogil

  int  gsl_sf_pochrel_e(double a, double x, gsl_sf_result * result) nogil

  double  gsl_sf_gamma_inc_Q(double a, double x) nogil

  int  gsl_sf_gamma_inc_Q_e(double a, double x, gsl_sf_result * result) nogil

  double  gsl_sf_gamma_inc_P(double a, double x) nogil

  int  gsl_sf_gamma_inc_P_e(double a, double x, gsl_sf_result * result) nogil

  double  gsl_sf_gamma_inc(double a, double x) nogil

  int  gsl_sf_gamma_inc_e(double a, double x, gsl_sf_result * result) nogil

  double  gsl_sf_beta(double a, double b) nogil

  int  gsl_sf_beta_e(double a, double b, gsl_sf_result * result) nogil

  double  gsl_sf_lnbeta(double a, double b) nogil

  int  gsl_sf_lnbeta_e(double a, double b, gsl_sf_result * result) nogil

  double  gsl_sf_beta_inc(double a, double b, double x) nogil

  int  gsl_sf_beta_inc_e(double a, double b, double x, gsl_sf_result * result) nogil

