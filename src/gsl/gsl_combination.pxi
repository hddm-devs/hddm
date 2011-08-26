cdef extern from "gsl/gsl_combination.h":

  ctypedef struct gsl_combination:
    size_t n
    size_t k
    size_t *data

  gsl_combination *  gsl_combination_alloc(size_t n, size_t k) nogil

  gsl_combination *  gsl_combination_calloc(size_t n, size_t k) nogil

  void  gsl_combination_init_first(gsl_combination * c) nogil

  void  gsl_combination_init_last(gsl_combination * c) nogil

  void  gsl_combination_free(gsl_combination * c) nogil

  int  gsl_combination_memcpy(gsl_combination * dest, gsl_combination * src) nogil

  int  gsl_combination_fread(FILE * stream, gsl_combination * c) nogil

  int  gsl_combination_fwrite(FILE * stream, gsl_combination * c) nogil

  int  gsl_combination_fscanf(FILE * stream, gsl_combination * c) nogil

  int  gsl_combination_fprintf(FILE * stream, gsl_combination * c, char *format) nogil

  size_t  gsl_combination_n(gsl_combination * c) nogil

  size_t  gsl_combination_k(gsl_combination * c) nogil

  size_t *  gsl_combination_data(gsl_combination * c) nogil

  size_t  gsl_combination_get(gsl_combination * c, size_t i) nogil

  int  gsl_combination_valid(gsl_combination * c) nogil

  int  gsl_combination_next(gsl_combination * c) nogil

  int  gsl_combination_prev(gsl_combination * c) nogil

