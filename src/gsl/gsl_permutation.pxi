cdef extern from "gsl/gsl_permutation.h":

  ctypedef struct gsl_permutation:
    size_t size
    size_t *data

  # Allocation
  gsl_permutation *  gsl_permutation_alloc(size_t n) nogil

  gsl_permutation *  gsl_permutation_calloc(size_t n) nogil

  void  gsl_permutation_init(gsl_permutation * p) nogil

  void  gsl_permutation_free(gsl_permutation * p) nogil

  # Reading and writing permutations
  int  gsl_permutation_fread(FILE * stream, gsl_permutation * p) nogil

  int  gsl_permutation_fwrite(FILE * stream, gsl_permutation * p) nogil

  int  gsl_permutation_fscanf(FILE * stream, gsl_permutation * p) nogil

  int  gsl_permutation_fprintf(FILE * stream, gsl_permutation * p, char *format) nogil

  # Permutation properties
  size_t  gsl_permutation_size(gsl_permutation * p) nogil

  size_t *  gsl_permutation_data(gsl_permutation * p) nogil

  int  gsl_permutation_memcpy(gsl_permutation * dest, gsl_permutation * src) nogil

  size_t  gsl_permutation_get(gsl_permutation * p, size_t i) nogil

  int  gsl_permutation_swap(gsl_permutation * p, size_t i, size_t j) nogil

  int  gsl_permutation_valid(gsl_permutation * p) nogil

  # Permutation functions
  void  gsl_permutation_reverse(gsl_permutation * p) nogil

  int  gsl_permutation_inverse(gsl_permutation * inv, gsl_permutation * p) nogil

  int  gsl_permutation_next(gsl_permutation * p) nogil

  int  gsl_permutation_prev(gsl_permutation * p) nogil

  int  gsl_permutation_mul(gsl_permutation * p, gsl_permutation * pa, gsl_permutation * pb) nogil

  # Permutations in Cyclic Form
  int  gsl_permutation_linear_to_canonical(gsl_permutation * q, gsl_permutation * p) nogil

  int  gsl_permutation_canonical_to_linear(gsl_permutation * p, gsl_permutation * q) nogil

  size_t  gsl_permutation_inversions(gsl_permutation * p) nogil

  size_t  gsl_permutation_linear_cycles(gsl_permutation * p) nogil

  size_t  gsl_permutation_canonical_cycles(gsl_permutation * q) nogil



# Applying Permutations
cdef extern from "gsl/gsl_permute_double.h":
  int  gsl_permute(size_t * p, double * data, size_t stride, size_t n) nogil

  int  gsl_permute_inverse(size_t * p, double * data, size_t stride, size_t n) nogil


cdef extern from "gsl/gsl_permute_vector_double.h":
  int  gsl_permute_vector(gsl_permutation * p, gsl_vector * v) nogil

  int  gsl_permute_vector_inverse(gsl_permutation * p, gsl_vector * v) nogil

cdef extern from "gsl/gsl_permute_vector_complex_double.h":
  int gsl_permute_vector_complex ( gsl_permutation * p, gsl_vector_complex * v) nogil

  int gsl_permute_vector_complex_inverse ( gsl_permutation * p, gsl_vector_complex * v) nogil
  
