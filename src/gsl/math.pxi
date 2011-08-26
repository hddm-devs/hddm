cdef extern from "math.h":
  double cos(double x) nogil
  double sin(double x) nogil
  double tan (double x) nogil
  double asin (double x) nogil
  double acos (double x) nogil
  double atan (double x) nogil
  double atan2 (double y, double x) nogil
  double exp (double x) nogil
  double exp2 (double x) nogil
  double exp10 (double x) nogil
  double log(double x) nogil
  double log10 (double x) nogil
  double log2 (double x) nogil
  double pow (double base, double power) nogil
  double sqrt(double x) nogil
  double sinh (double x) nogil
  double cosh (double x) nogil
  double tanh (double x) nogil
  double fabs(double x) nogil

