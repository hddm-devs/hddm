cdef extern from "gsl/gsl_math.h":

  double M_E

  double M_LOG2E

  double M_LOG10E

  double M_SQRT2

  double M_SQRT1_2

  double M_SQRT3

  double M_PI

  double M_PI_2

  double M_PI_4

  double M_SQRTPI

  double M_2_SQRTPI

  double M_1_PI

  double M_2_PI

  double M_LN10

  double M_LN2

  double M_LNPI

  double M_EULER

  int  gsl_isnan(double x) nogil

  int  gsl_isinf(double x) nogil

  int  gsl_finite(double x) nogil

  double  gsl_log1p(double x) nogil

  double  gsl_expm1(double x) nogil

  double  gsl_hypot(double x, double y) nogil

  double  gsl_acosh(double x) nogil

  double  gsl_asinh(double x) nogil

  double  gsl_atanh(double x) nogil

  double  gsl_ldexp(double x, int e) nogil

  double  gsl_frexp(double x, int * e) nogil

  double  gsl_pow_int(double x, int n) nogil

  double  gsl_pow_2(double x) nogil

  double  gsl_pow_3(double x) nogil

  double  gsl_pow_4(double x) nogil

  double  gsl_pow_5(double x) nogil

  double  gsl_pow_6(double x) nogil

  double  gsl_pow_7(double x) nogil

  double  gsl_pow_8(double x) nogil

  double  gsl_pow_9(double x) nogil

  int GSL_SIGN(double x) nogil

  int GSL_IS_ODD(int n) nogil

  int GSL_IS_EVEN(int n) nogil

  double GSL_MAX(double a, double  b) nogil

  double GSL_MIN(double a, double  b) nogil

  double  GSL_MAX_DBL(double a, double b) nogil

  double  GSL_MIN_DBL(double a, double b) nogil

  int  GSL_MAX_INT(int a, int b) nogil

  int  GSL_MIN_INT(int a, int b) nogil

  long double  GSL_MAX_LDBL(long double a, long double b) nogil

  long double  GSL_MIN_LDBL(long double a, long double b) nogil

  int  gsl_fcmp(double x, double y, double epsilon) nogil

  # Definition of an arbitrary function with parameters
  ctypedef struct gsl_function:
    double (* function) (double x, void * params) nogil
    void * params

  double GSL_FN_EVAL(gsl_function * F, double x) nogil

  # Definition of an arbitrary function returning two values, r1, r2
  ctypedef struct gsl_function_fdf:
    double (* f) (double x, void * params) nogil
    double (* df) (double x, void * params) nogil
    void (* fdf) (double x, void * params, double * f, double * df) nogil
    void * params

  double GSL_FN_FDF_EVAL_F(gsl_function_fdf * FDF, double x) nogil 
  
  GSL_FN_FDF_EVAL_DF(gsl_function_fdf * FDF,double x) nogil 
  
  GSL_FN_FDF_EVAL_F_DF(gsl_function_fdf * FDF,double x, double y,double dy) nogil 

