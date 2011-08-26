cdef extern from "gsl/gsl_block_double.h":

  ctypedef struct gsl_block:
    size_t size
    double * data
  
  gsl_block *  gsl_block_alloc(size_t n) nogil
  
  gsl_block *  gsl_block_calloc(size_t n) nogil
  
  void  gsl_block_free(gsl_block * b) nogil
  
  int  gsl_block_fread(FILE * stream, gsl_block * b) nogil

  int  gsl_block_fwrite(FILE * stream, gsl_block * b) nogil

  int  gsl_block_fscanf(FILE * stream, gsl_block * b) nogil
  
  int  gsl_block_fprintf(FILE * stream, gsl_block * b, char * format) nogil

  size_t gsl_block_size (gsl_block * b) nogil
  double * gsl_block_data (gsl_block * b) nogil

  
cdef extern from "gsl/gsl_block_complex_double.h":

  ctypedef struct gsl_block_complex:
    size_t size
    double * data
  
  gsl_block_complex *  gsl_block_complex_alloc(size_t n) nogil
  
  gsl_block_complex *  gsl_block_complex_calloc(size_t n) nogil
  
  void  gsl_block_complex_free(gsl_block_complex * b) nogil
  
  int  gsl_block_complex_fread(FILE * stream, gsl_block_complex * b) nogil

  int  gsl_block_complex_fwrite(FILE * stream, gsl_block_complex * b) nogil

  int  gsl_block_complex_fscanf(FILE * stream, gsl_block_complex * b) nogil
  
  int  gsl_block_complex_fprintf(FILE * stream, gsl_block_complex * b, char * format) nogil

  size_t gsl_block_complex_size (gsl_block_complex * b) nogil
  double * gsl_block_complex_data (gsl_block_complex * b) nogil
