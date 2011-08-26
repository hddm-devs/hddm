cdef extern from "gsl/gsl_matrix_double.h":

  ctypedef struct gsl_matrix:
    size_t size1
    size_t size2
    size_t tda
    double * data
    gsl_block * block
    int owner
	      
  ctypedef struct gsl_matrix_view:
    gsl_matrix matrix

  ctypedef struct gsl_matrix_const_view

  # Allocation
  gsl_matrix *  gsl_matrix_alloc(size_t n1, size_t n2) nogil

  gsl_matrix *  gsl_matrix_calloc(size_t n1, size_t n2) nogil

  gsl_matrix *  gsl_matrix_alloc_from_block(gsl_block * b,
    size_t offset, size_t n1, size_t n2, size_t d2) nogil

  gsl_matrix * gsl_matrix_alloc_from_matrix (gsl_matrix * m,  size_t k1,  size_t k2,  size_t n1,  size_t n2) nogil
 
  gsl_vector * gsl_vector_alloc_row_from_matrix (gsl_matrix * m,  size_t i) nogil

  gsl_vector * gsl_vector_alloc_col_from_matrix (gsl_matrix * m,  size_t j) nogil
 
  void  gsl_matrix_free(gsl_matrix * m) nogil

  # Views
  gsl_matrix_view  gsl_matrix_submatrix(gsl_matrix * m, size_t k1, size_t k2, size_t n1, size_t n2) nogil

  gsl_vector_view  gsl_matrix_row(gsl_matrix * m, size_t i) nogil

  gsl_vector_view  gsl_matrix_column(gsl_matrix * m, size_t j) nogil

  gsl_vector_view  gsl_matrix_diagonal(gsl_matrix * m) nogil

  gsl_vector_view  gsl_matrix_subdiagonal(gsl_matrix * m, size_t k) nogil

  gsl_vector_view  gsl_matrix_superdiagonal(gsl_matrix * m, size_t k) nogil

  gsl_matrix_view  gsl_matrix_view_array(double * base, size_t n1, size_t n2) nogil

  gsl_matrix_view  gsl_matrix_view_array_with_tda(double * base, size_t n1, size_t n2, size_t tda) nogil

  gsl_matrix_view  gsl_matrix_view_vector(gsl_vector * v, size_t n1, size_t n2) nogil

  gsl_matrix_view  gsl_matrix_view_vector_with_tda(gsl_vector * v, size_t n1, size_t n2, size_t tda) nogil

  gsl_matrix_const_view  gsl_matrix_const_submatrix(gsl_matrix * m, size_t k1, size_t k2, size_t n1, size_t n2) nogil

  gsl_vector_const_view  gsl_matrix_const_row(gsl_matrix * m, size_t i) nogil

  gsl_vector_const_view  gsl_matrix_const_column(gsl_matrix * m, size_t j) nogil

  gsl_vector_const_view  gsl_matrix_const_diagonal(gsl_matrix * m) nogil

  gsl_vector_const_view  gsl_matrix_const_subdiagonal(gsl_matrix * m, size_t k) nogil

  gsl_vector_const_view  gsl_matrix_const_superdiagonal(gsl_matrix * m, size_t k) nogil
  
  gsl_matrix_const_view  gsl_matrix_const_view_array(double * base, size_t n1, size_t n2) nogil

  gsl_matrix_const_view  gsl_matrix_const_view_array_with_tda(double * base, size_t n1, size_t n2, size_t tda) nogil

  gsl_matrix_const_view  gsl_matrix_const_view_vector(gsl_vector * v, size_t n1, size_t n2) nogil

  gsl_matrix_const_view  gsl_matrix_const_view_vector_with_tda(gsl_vector * v, size_t n1, size_t n2, size_t tda) nogil

  
  # Operations
  double  gsl_matrix_get(gsl_matrix * m, size_t i, size_t j) nogil

  void  gsl_matrix_set(gsl_matrix * m, size_t i, size_t j, double x) nogil

  double *  gsl_matrix_ptr(gsl_matrix * m, size_t i, size_t j) nogil

  double *  gsl_matrix_const_ptr(gsl_matrix * m, size_t i, size_t j) nogil

  void  gsl_matrix_set_zero(gsl_matrix * m) nogil

  void  gsl_matrix_set_identity(gsl_matrix * m) nogil

  void  gsl_matrix_set_all(gsl_matrix * m, double x) nogil

  # Reading and writing matrices
  int  gsl_matrix_fread(FILE * stream, gsl_matrix * m) nogil

  int  gsl_matrix_fwrite(FILE * stream, gsl_matrix * m) nogil

  int  gsl_matrix_fscanf(FILE * stream, gsl_matrix * m) nogil

  int  gsl_matrix_fprintf(FILE * stream, gsl_matrix * m, char * format) nogil

  # Copying or exchanging elements
  int  gsl_matrix_memcpy(gsl_matrix * dest, gsl_matrix * src) nogil

  int  gsl_matrix_swap(gsl_matrix * m1, gsl_matrix * m2) nogil

  int  gsl_matrix_swap_rows(gsl_matrix * m, size_t i, size_t j) nogil

  int  gsl_matrix_swap_columns(gsl_matrix * m, size_t i, size_t j) nogil

  int  gsl_matrix_swap_rowcol(gsl_matrix * m, size_t i, size_t j) nogil

  int  gsl_matrix_transpose(gsl_matrix * m) nogil

  int  gsl_matrix_transpose_memcpy(gsl_matrix * dest, gsl_matrix * src) nogil

  # Finding maximum and minimum elements of matrices
  double  gsl_matrix_max(gsl_matrix * m) nogil

  double  gsl_matrix_min(gsl_matrix * m) nogil

  void  gsl_matrix_minmax(gsl_matrix * m, double * min_out, double * max_out) nogil

  void  gsl_matrix_max_index(gsl_matrix * m, size_t * imax, size_t * jmax) nogil

  void  gsl_matrix_min_index(gsl_matrix * m, size_t * imax, size_t * jmax) nogil

  void  gsl_matrix_minmax_index(gsl_matrix * m, size_t * imin, size_t * jmin, size_t * imax, size_t * jmax) nogil

  int  gsl_matrix_isnull(gsl_matrix * m) nogil

  # Matrix operations
  int  gsl_matrix_add(gsl_matrix * a, gsl_matrix * b) nogil

  int  gsl_matrix_sub(gsl_matrix * a, gsl_matrix * b) nogil

  int  gsl_matrix_mul_elements(gsl_matrix * a, gsl_matrix * b) nogil

  int  gsl_matrix_div_elements(gsl_matrix * a, gsl_matrix * b) nogil

  int  gsl_matrix_scale(gsl_matrix * a, double x) nogil

  int  gsl_matrix_add_constant(gsl_matrix * a, double x) nogil
  
  int gsl_matrix_add_diagonal (gsl_matrix * a,  double x) nogil
  
  # The functions below are obsolete
  int  gsl_matrix_get_row(gsl_vector * v, gsl_matrix * m, size_t i) nogil

  int  gsl_matrix_get_col(gsl_vector * v, gsl_matrix * m, size_t j) nogil

  int  gsl_matrix_set_row(gsl_matrix * m, size_t i, gsl_vector * v) nogil

  int  gsl_matrix_set_col(gsl_matrix * m, size_t j, gsl_vector * v) nogil

