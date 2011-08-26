cdef extern from "gsl/gsl_vector.h":

  ctypedef struct gsl_vector:
    size_t size
    size_t stride
    double *data
    gsl_block *block
    int owner

  ctypedef struct gsl_vector_view:
    gsl_vector vector
    
  ctypedef struct gsl_vector_const_view:
    gsl_vector vector
   

  # Allocation
  gsl_vector *  gsl_vector_alloc(size_t n) nogil

  gsl_vector *  gsl_vector_calloc(size_t n) nogil

  gsl_vector_alloc_from_block(gsl_block * b, size_t offset,
                              size_t n, size_t stride) nogil

  gsl_vector *gsl_vector_alloc_from_vector(gsl_vector * v,
                         size_t offset, size_t n, size_t stride) nogil
  
  void  gsl_vector_free(gsl_vector * v) nogil

  # Views
  gsl_vector_view  gsl_vector_view_array(double *base, size_t n) nogil

  gsl_vector_view  gsl_vector_subvector(gsl_vector *v, size_t offset, size_t n) nogil

  gsl_vector_view  gsl_vector_view_array_with_stride(double * base, size_t stride, size_t n) nogil

  gsl_vector_const_view  gsl_vector_const_view_array(double *base, size_t n) nogil

  gsl_vector_const_view  gsl_vector_const_view_array_with_stride(double * base, size_t stride, size_t n) nogil

  gsl_vector_const_view  gsl_vector_const_subvector(gsl_vector * v, size_t offset, size_t n) nogil

  gsl_vector_view  gsl_vector_subvector_with_stride(gsl_vector *v, size_t offset, size_t stride, size_t n) nogil

  gsl_vector_const_view  gsl_vector_const_subvector_with_stride(gsl_vector * v, size_t offset, size_t stride, size_t n) nogil


  # Operations
  double  gsl_vector_get(gsl_vector * v, size_t i) nogil

  void  gsl_vector_set(gsl_vector * v, size_t i, double x) nogil

  double *  gsl_vector_ptr(gsl_vector * v, size_t i) nogil

  double *  gsl_vector_const_ptr(gsl_vector * v, size_t i) nogil

  void  gsl_vector_set_zero(gsl_vector * v) nogil

  void  gsl_vector_set_all(gsl_vector * v, double x) nogil

  int  gsl_vector_set_basis(gsl_vector * v, size_t i) nogil

  # Reading and writing vectors
  int  gsl_vector_fread(FILE * stream, gsl_vector * v) nogil

  int  gsl_vector_fwrite(FILE * stream, gsl_vector * v) nogil

  int  gsl_vector_fscanf(FILE * stream, gsl_vector * v) nogil

  int  gsl_vector_fprintf(FILE * stream, gsl_vector * v, char * format) nogil

  # Copying or exchanging elements
  int  gsl_vector_memcpy(gsl_vector * dest, gsl_vector * src) nogil

  int  gsl_vector_reverse(gsl_vector * v) nogil

  int  gsl_vector_swap(gsl_vector * v, gsl_vector * w) nogil

  int  gsl_vector_swap_elements(gsl_vector * v, size_t i, size_t j) nogil

  # Finding maximum and minimum elements of vectors

  double  gsl_vector_max(gsl_vector * v) nogil

  double  gsl_vector_min(gsl_vector * v) nogil

  void  gsl_vector_minmax(gsl_vector * v, double * min_out, double * max_out) nogil

  size_t  gsl_vector_max_index(gsl_vector * v) nogil

  size_t  gsl_vector_min_index(gsl_vector * v) nogil

  void  gsl_vector_minmax_index(gsl_vector * v, size_t * imin, size_t * imax) nogil

  # Vector operations
  int  gsl_vector_add(gsl_vector * a, gsl_vector * b) nogil

  int  gsl_vector_sub(gsl_vector * a, gsl_vector * b) nogil

  int  gsl_vector_mul(gsl_vector * a, gsl_vector * b) nogil

  int  gsl_vector_div(gsl_vector * a, gsl_vector * b) nogil

  int  gsl_vector_scale(gsl_vector * a, double x) nogil

  int  gsl_vector_add_constant(gsl_vector * a, double x) nogil

  int  gsl_vector_isnull(gsl_vector * v) nogil


