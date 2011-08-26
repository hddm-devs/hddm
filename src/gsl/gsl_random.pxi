cdef extern from "gsl/gsl_randist.h":
  unsigned int gsl_ran_bernoulli ( gsl_rng * r, double p) nogil
  double gsl_ran_bernoulli_pdf ( unsigned int k, double p) nogil
  
  double gsl_ran_beta ( gsl_rng * r,  double a,  double b) nogil
  double gsl_ran_beta_pdf ( double x,  double a,  double b) nogil
  
  unsigned int gsl_ran_binomial ( gsl_rng * r, double p, unsigned int n) nogil
  unsigned int gsl_ran_binomial_tpe ( gsl_rng * r, double pp, unsigned int n) nogil
  double gsl_ran_binomial_pdf ( unsigned int k,  double p,  unsigned int n) nogil
  
  double gsl_ran_exponential ( gsl_rng * r,  double mu) nogil
  double gsl_ran_exponential_pdf ( double x,  double mu) nogil
  
  double gsl_ran_exppow ( gsl_rng * r,  double a,  double b) nogil
  double gsl_ran_exppow_pdf ( double x,  double a,  double b) nogil
  
  double gsl_ran_cauchy ( gsl_rng * r,  double a) nogil
  double gsl_ran_cauchy_pdf ( double x,  double a) nogil
  
  double gsl_ran_chisq ( gsl_rng * r,  double nu) nogil
  double gsl_ran_chisq_pdf ( double x,  double nu) nogil
  
  void gsl_ran_dirichlet ( gsl_rng * r,  size_t K,  double alpha[], double theta[]) nogil
  double gsl_ran_dirichlet_pdf ( size_t K,  double alpha[],  double theta[]) nogil
  double gsl_ran_dirichlet_lnpdf ( size_t K,  double alpha[],  double theta[]) nogil
  
  double gsl_ran_erlang ( gsl_rng * r,  double a,  double n) nogil
  double gsl_ran_erlang_pdf ( double x,  double a,  double n) nogil
  
  double gsl_ran_fdist ( gsl_rng * r,  double nu1,  double nu2) nogil
  double gsl_ran_fdist_pdf ( double x,  double nu1,  double nu2) nogil
  
  double gsl_ran_flat ( gsl_rng * r,  double a,  double b) nogil
  double gsl_ran_flat_pdf (double x,  double a,  double b) nogil
  
  double gsl_ran_gamma ( gsl_rng * r,  double a,  double b) nogil
  double gsl_ran_gamma_int ( gsl_rng * r,  unsigned int a) nogil
  double gsl_ran_gamma_pdf ( double x,  double a,  double b) nogil
  
  double gsl_ran_gaussian ( gsl_rng * r,  double sigma) nogil
  double gsl_ran_gaussian_ratio_method ( gsl_rng * r,  double sigma) nogil
  double gsl_ran_gaussian_pdf ( double x,  double sigma) nogil
  
  double gsl_ran_ugaussian ( gsl_rng * r) nogil
  double gsl_ran_ugaussian_ratio_method ( gsl_rng * r) nogil
  double gsl_ran_ugaussian_pdf ( double x) nogil
  
  double gsl_ran_gaussian_tail ( gsl_rng * r,  double a,  double sigma) nogil
  double gsl_ran_gaussian_tail_pdf ( double x,  double a,  double sigma) nogil
  
  double gsl_ran_ugaussian_tail ( gsl_rng * r,  double a) nogil
  double gsl_ran_ugaussian_tail_pdf ( double x,  double a) nogil
  
  void gsl_ran_bivariate_gaussian ( gsl_rng * r, double sigma_x, double sigma_y, double rho, double *x, double *y) nogil
  double gsl_ran_bivariate_gaussian_pdf ( double x,  double y,  double sigma_x,  double sigma_y,  double rho) nogil
  
  double gsl_ran_landau ( gsl_rng * r) nogil
  double gsl_ran_landau_pdf ( double x) nogil
  
  unsigned int gsl_ran_geometric ( gsl_rng * r,  double p) nogil
  double gsl_ran_geometric_pdf ( unsigned int k,  double p) nogil
  
  unsigned int gsl_ran_hypergeometric ( gsl_rng * r, unsigned int n1, unsigned int n2, unsigned int t) nogil
  double gsl_ran_hypergeometric_pdf ( unsigned int k,  unsigned int n1,  unsigned int n2, unsigned int t) nogil
  
  double gsl_ran_gumbel1 ( gsl_rng * r,  double a,  double b) nogil
  double gsl_ran_gumbel1_pdf ( double x,  double a,  double b) nogil
  
  double gsl_ran_gumbel2 ( gsl_rng * r,  double a,  double b) nogil
  double gsl_ran_gumbel2_pdf ( double x,  double a,  double b) nogil
  
  double gsl_ran_logistic ( gsl_rng * r,  double a) nogil
  double gsl_ran_logistic_pdf ( double x,  double a) nogil
  
  double gsl_ran_lognormal ( gsl_rng * r,  double zeta,  double sigma) nogil
  double gsl_ran_lognormal_pdf ( double x,  double zeta,  double sigma) nogil
  
  unsigned int gsl_ran_logarithmic ( gsl_rng * r,  double p) nogil
  double gsl_ran_logarithmic_pdf ( unsigned int k,  double p) nogil
  
  void gsl_ran_multinomial ( gsl_rng * r,  size_t K,
                             unsigned int N,  double p[],
                            unsigned int n[] ) nogil
  double gsl_ran_multinomial_pdf ( size_t K,
                                   double p[],  unsigned int n[] ) nogil
  double gsl_ran_multinomial_lnpdf ( size_t K,
                              double p[],  unsigned int n[] ) nogil
  
  
  unsigned int gsl_ran_negative_binomial ( gsl_rng * r, double p, double n) nogil
  double gsl_ran_negative_binomial_pdf ( unsigned int k,  double p, double n) nogil
  
  unsigned int gsl_ran_pascal ( gsl_rng * r, double p, unsigned int n) nogil
  double gsl_ran_pascal_pdf ( unsigned int k,  double p, unsigned int n) nogil
  
  double gsl_ran_pareto ( gsl_rng * r, double a,  double b) nogil
  double gsl_ran_pareto_pdf ( double x,  double a,  double b) nogil
  
  unsigned int gsl_ran_poisson ( gsl_rng * r, double mu) nogil
  void gsl_ran_poisson_array ( gsl_rng * r, size_t n, unsigned int array[],
                              double mu) nogil
  double gsl_ran_poisson_pdf ( unsigned int k,  double mu) nogil
  
  double gsl_ran_rayleigh ( gsl_rng * r,  double sigma) nogil
  double gsl_ran_rayleigh_pdf ( double x,  double sigma) nogil
  
  double gsl_ran_rayleigh_tail ( gsl_rng * r,  double a,  double sigma) nogil
  double gsl_ran_rayleigh_tail_pdf ( double x,  double a,  double sigma) nogil
  
  double gsl_ran_tdist ( gsl_rng * r,  double nu) nogil
  double gsl_ran_tdist_pdf ( double x,  double nu) nogil
  
  double gsl_ran_laplace ( gsl_rng * r,  double a) nogil
  double gsl_ran_laplace_pdf ( double x,  double a) nogil
  
  double gsl_ran_levy ( gsl_rng * r,  double c,  double alpha) nogil
  double gsl_ran_levy_skew ( gsl_rng * r,  double c,  double alpha,  double beta) nogil
  
  double gsl_ran_weibull ( gsl_rng * r,  double a,  double b) nogil
  double gsl_ran_weibull_pdf ( double x,  double a,  double b) nogil
  
  void gsl_ran_dir_2d ( gsl_rng * r, double * x, double * y) nogil
  void gsl_ran_dir_2d_trig_method ( gsl_rng * r, double * x, double * y) nogil
  void gsl_ran_dir_3d ( gsl_rng * r, double * x, double * y, double * z) nogil
  void gsl_ran_dir_nd ( gsl_rng * r, size_t n, double * x) nogil
  
  void gsl_ran_shuffle ( gsl_rng * r, void * base, size_t nmembm, size_t size) nogil
  int gsl_ran_choose ( gsl_rng * r, void * dest, size_t k, void * src, size_t n, size_t size) nogil 
  void gsl_ran_sample ( gsl_rng * r, void * dest, size_t k, void * src, size_t n, size_t size) nogil 
  
  
  ctypedef struct gsl_ran_discrete_t
  
  gsl_ran_discrete_t * gsl_ran_discrete_preproc (size_t K,  double *P) nogil
  void gsl_ran_discrete_free(gsl_ran_discrete_t *g) nogil
  size_t gsl_ran_discrete ( gsl_rng *r,  gsl_ran_discrete_t *g) nogil
  double gsl_ran_discrete_pdf (size_t k,  gsl_ran_discrete_t *g) nogil
  
cdef extern from "gsl/gsl_cdf.h": 
  double gsl_cdf_ugaussian_P ( double x) nogil
  double gsl_cdf_ugaussian_Q ( double x) nogil
  
  double gsl_cdf_ugaussian_Pinv ( double P) nogil
  double gsl_cdf_ugaussian_Qinv ( double Q) nogil
  
  double gsl_cdf_gaussian_P ( double x,  double sigma) nogil
  double gsl_cdf_gaussian_Q ( double x,  double sigma) nogil
  
  double gsl_cdf_gaussian_Pinv ( double P,  double sigma) nogil
  double gsl_cdf_gaussian_Qinv ( double Q,  double sigma) nogil
  
  double gsl_cdf_gamma_P ( double x,  double a,  double b) nogil
  double gsl_cdf_gamma_Q ( double x,  double a,  double b) nogil
  
  double gsl_cdf_gamma_Pinv ( double P,  double a,  double b) nogil
  double gsl_cdf_gamma_Qinv ( double Q,  double a,  double b) nogil
  
  double gsl_cdf_cauchy_P ( double x,  double a) nogil
  double gsl_cdf_cauchy_Q ( double x,  double a) nogil
  
  double gsl_cdf_cauchy_Pinv ( double P,  double a) nogil
  double gsl_cdf_cauchy_Qinv ( double Q,  double a) nogil
  
  double gsl_cdf_laplace_P ( double x,  double a) nogil
  double gsl_cdf_laplace_Q ( double x,  double a) nogil
  
  double gsl_cdf_laplace_Pinv ( double P,  double a) nogil
  double gsl_cdf_laplace_Qinv ( double Q,  double a) nogil
  
  double gsl_cdf_rayleigh_P ( double x,  double sigma) nogil
  double gsl_cdf_rayleigh_Q ( double x,  double sigma) nogil
  
  double gsl_cdf_rayleigh_Pinv ( double P,  double sigma) nogil
  double gsl_cdf_rayleigh_Qinv ( double Q,  double sigma) nogil
  
  double gsl_cdf_chisq_P ( double x,  double nu) nogil
  double gsl_cdf_chisq_Q ( double x,  double nu) nogil
  
  double gsl_cdf_chisq_Pinv ( double P,  double nu) nogil
  double gsl_cdf_chisq_Qinv ( double Q,  double nu) nogil
  
  double gsl_cdf_exponential_P ( double x,  double mu) nogil
  double gsl_cdf_exponential_Q ( double x,  double mu) nogil
  
  double gsl_cdf_exponential_Pinv ( double P,  double mu) nogil
  double gsl_cdf_exponential_Qinv ( double Q,  double mu) nogil
  
  double gsl_cdf_tdist_P ( double x,  double nu) nogil
  double gsl_cdf_tdist_Q ( double x,  double nu) nogil
  
  double gsl_cdf_tdist_Pinv ( double P,  double nu) nogil
  double gsl_cdf_tdist_Qinv ( double Q,  double nu) nogil
  
  double gsl_cdf_fdist_P ( double x,  double nu1,  double nu2) nogil
  double gsl_cdf_fdist_Q ( double x,  double nu1,  double nu2) nogil
  
  double gsl_cdf_beta_P ( double x,  double a,  double b) nogil
  double gsl_cdf_beta_Q ( double x,  double a,  double b) nogil
  
  double gsl_cdf_flat_P ( double x,  double a,  double b) nogil
  double gsl_cdf_flat_Q ( double x,  double a,  double b) nogil
  
  double gsl_cdf_flat_Pinv ( double P,  double a,  double b) nogil
  double gsl_cdf_flat_Qinv ( double Q,  double a,  double b) nogil
  
  double gsl_cdf_lognormal_P ( double x,  double zeta,  double sigma) nogil
  double gsl_cdf_lognormal_Q ( double x,  double zeta,  double sigma) nogil
  
  double gsl_cdf_lognormal_Pinv ( double P,  double zeta,  double sigma) nogil
  double gsl_cdf_lognormal_Qinv ( double Q,  double zeta,  double sigma) nogil
  
  double gsl_cdf_gumbel1_P ( double x,  double a,  double b) nogil
  double gsl_cdf_gumbel1_Q ( double x,  double a,  double b) nogil
  
  double gsl_cdf_gumbel1_Pinv ( double P,  double a,  double b) nogil
  double gsl_cdf_gumbel1_Qinv ( double Q,  double a,  double b) nogil
  
  double gsl_cdf_gumbel2_P ( double x,  double a,  double b) nogil
  double gsl_cdf_gumbel2_Q ( double x,  double a,  double b) nogil
  
  double gsl_cdf_gumbel2_Pinv ( double P,  double a,  double b) nogil
  double gsl_cdf_gumbel2_Qinv ( double Q,  double a,  double b) nogil
  
  double gsl_cdf_weibull_P ( double x,  double a,  double b) nogil
  double gsl_cdf_weibull_Q ( double x,  double a,  double b) nogil
  
  double gsl_cdf_weibull_Pinv ( double P,  double a,  double b) nogil
  double gsl_cdf_weibull_Qinv ( double Q,  double a,  double b) nogil
  
  double gsl_cdf_pareto_P ( double x,  double a,  double b) nogil
  double gsl_cdf_pareto_Q ( double x,  double a,  double b) nogil
  
  double gsl_cdf_pareto_Pinv ( double P,  double a,  double b) nogil
  double gsl_cdf_pareto_Qinv ( double Q,  double a,  double b) nogil
  
  double gsl_cdf_logistic_P ( double x,  double a) nogil
  double gsl_cdf_logistic_Q ( double x,  double a) nogil
  
  double gsl_cdf_logistic_Pinv ( double P,  double a) nogil
  double gsl_cdf_logistic_Qinv ( double Q,  double a) nogil
