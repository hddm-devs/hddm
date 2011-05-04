cdef extern from "gsl_sf_elljac.h":

  int  gsl_sf_elljac_e(double u, double m, double * sn, double * cn, double * dn)

