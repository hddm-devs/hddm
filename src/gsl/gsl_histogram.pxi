cdef extern from "gsl/gsl_histogram.h":
  ctypedef struct gsl_histogram 
  ctypedef struct gsl_histogram_pdf 
  
  gsl_histogram * gsl_histogram_alloc (size_t n) nogil
  
  gsl_histogram * gsl_histogram_calloc (size_t n) nogil
  gsl_histogram * gsl_histogram_calloc_uniform ( size_t n,  double xmin,  double xmax) nogil
  void gsl_histogram_free (gsl_histogram * h) nogil
  int gsl_histogram_increment (gsl_histogram * h, double x) nogil
  int gsl_histogram_accumulate (gsl_histogram * h, double x, double weight) nogil
  int gsl_histogram_find ( gsl_histogram * h, 
                           double x, size_t * i) nogil
  
  double gsl_histogram_get ( gsl_histogram * h, size_t i) nogil
  int gsl_histogram_get_range ( gsl_histogram * h, size_t i, 
                               double * lower, double * upper) nogil
                                       
  double gsl_histogram_max ( gsl_histogram * h) nogil
  double gsl_histogram_min ( gsl_histogram * h) nogil
  size_t gsl_histogram_bins ( gsl_histogram * h) nogil
  
  void gsl_histogram_reset (gsl_histogram * h) nogil
  
  gsl_histogram * gsl_histogram_calloc_range(size_t n, double * range) nogil
  
  int gsl_histogram_set_ranges (gsl_histogram * h,  double range[], size_t size) nogil
  int gsl_histogram_set_ranges_uniform (gsl_histogram * h, double xmin, double xmax) nogil
  
  
  
  int gsl_histogram_memcpy(gsl_histogram * dest,  gsl_histogram * source) nogil
  
  gsl_histogram * gsl_histogram_clone( gsl_histogram * source) nogil
  
  double gsl_histogram_max_val ( gsl_histogram * h) nogil
  
  size_t gsl_histogram_max_bin ( gsl_histogram * h) nogil
  
  double gsl_histogram_min_val ( gsl_histogram * h) nogil
  
  size_t gsl_histogram_min_bin ( gsl_histogram * h) nogil
  
  int gsl_histogram_equal_bins_p( gsl_histogram *h1,  gsl_histogram *h2) nogil
  
  int gsl_histogram_add(gsl_histogram *h1,  gsl_histogram *h2) nogil
  
  int gsl_histogram_sub(gsl_histogram *h1,  gsl_histogram *h2) nogil
  
  int gsl_histogram_mul(gsl_histogram *h1,  gsl_histogram *h2) nogil
   
  int gsl_histogram_div(gsl_histogram *h1,  gsl_histogram *h2) nogil
  
  int gsl_histogram_scale(gsl_histogram *h, double scale) nogil
  
  int gsl_histogram_shift (gsl_histogram * h, double shift) nogil
  
  
  double gsl_histogram_sigma ( gsl_histogram * h) nogil
  
  double gsl_histogram_mean ( gsl_histogram * h) nogil
  
  double gsl_histogram_sum ( gsl_histogram * h) nogil
  
  int gsl_histogram_fwrite (FILE * stream,  gsl_histogram * h) nogil 
  int gsl_histogram_fread (FILE * stream, gsl_histogram * h) nogil
  int gsl_histogram_fprintf (FILE * stream,  gsl_histogram * h, 
                              char * range_format,  char * bin_format) nogil
  int gsl_histogram_fscanf (FILE * stream, gsl_histogram * h) nogil
  
  gsl_histogram_pdf * gsl_histogram_pdf_alloc ( size_t n) nogil
  int gsl_histogram_pdf_init (gsl_histogram_pdf * p,  gsl_histogram * h) nogil
  void gsl_histogram_pdf_free (gsl_histogram_pdf * p) nogil
  double gsl_histogram_pdf_sample ( gsl_histogram_pdf * p, double r) nogil
  

cdef extern from "gsl/gsl_histogram2d.h":
  ctypedef struct gsl_histogram2d:
    size_t nx, ny
    double * xrange
    double * yrange
    double * bin
  
  ctypedef struct gsl_histogram2d_pdf 
  
  gsl_histogram2d * gsl_histogram2d_alloc ( size_t nx,  size_t ny) nogil
  gsl_histogram2d * gsl_histogram2d_calloc ( size_t nx,  size_t ny) nogil
  gsl_histogram2d * gsl_histogram2d_calloc_uniform ( size_t nx,  size_t ny,
                                                double xmin,  double xmax,
                                                double ymin,  double ymax) nogil
  
  void gsl_histogram2d_free (gsl_histogram2d * h) nogil
  
  int gsl_histogram2d_increment (gsl_histogram2d * h, double x, double y) nogil
  int gsl_histogram2d_accumulate (gsl_histogram2d * h, 
                                  double x, double y, double weight) nogil
  int gsl_histogram2d_find ( gsl_histogram2d * h, 
                             double x,  double y, size_t * i, size_t * j) nogil
  
  double gsl_histogram2d_get ( gsl_histogram2d * h,  size_t i,  size_t j) nogil
  int gsl_histogram2d_get_xrange ( gsl_histogram2d * h,  size_t i,
                                  double * xlower, double * xupper) nogil
  int gsl_histogram2d_get_yrange ( gsl_histogram2d * h,  size_t j,
                                  double * ylower, double * yupper) nogil
  
                                       
  double gsl_histogram2d_xmax ( gsl_histogram2d * h) nogil
  double gsl_histogram2d_xmin ( gsl_histogram2d * h) nogil
  size_t gsl_histogram2d_nx ( gsl_histogram2d * h) nogil
  
  double gsl_histogram2d_ymax ( gsl_histogram2d * h) nogil
  double gsl_histogram2d_ymin ( gsl_histogram2d * h) nogil
  size_t gsl_histogram2d_ny ( gsl_histogram2d * h) nogil
  
  void gsl_histogram2d_reset (gsl_histogram2d * h) nogil
  
  gsl_histogram2d * gsl_histogram2d_calloc_range(size_t nx, size_t ny, double *xrange, double *yrange) nogil
  
  int gsl_histogram2d_set_ranges_uniform (gsl_histogram2d * h, 
                                      double xmin, double xmax,
                                      double ymin, double ymax) nogil
  
  int gsl_histogram2d_set_ranges (gsl_histogram2d * h, 
                               double xrange[], size_t xsize,
                               double yrange[], size_t ysize) nogil
  
  int gsl_histogram2d_memcpy(gsl_histogram2d *dest,  gsl_histogram2d *source) nogil
  
  gsl_histogram2d * gsl_histogram2d_clone( gsl_histogram2d * source) nogil
  
  double gsl_histogram2d_max_val( gsl_histogram2d *h) nogil
  
  void gsl_histogram2d_max_bin ( gsl_histogram2d *h, size_t *i, size_t *j) nogil
  
  double gsl_histogram2d_min_val( gsl_histogram2d *h) nogil
  
  void gsl_histogram2d_min_bin ( gsl_histogram2d *h, size_t *i, size_t *j) nogil
  
  double gsl_histogram2d_xmean ( gsl_histogram2d * h) nogil
  
  double gsl_histogram2d_ymean ( gsl_histogram2d * h) nogil
  
  double gsl_histogram2d_xsigma ( gsl_histogram2d * h) nogil
  
  double gsl_histogram2d_ysigma ( gsl_histogram2d * h) nogil
  
  double gsl_histogram2d_cov ( gsl_histogram2d * h) nogil
  
  double gsl_histogram2d_sum ( gsl_histogram2d *h) nogil
  
  int gsl_histogram2d_equal_bins_p( gsl_histogram2d *h1,  gsl_histogram2d *h2) nogil 
  
  int gsl_histogram2d_add(gsl_histogram2d *h1,  gsl_histogram2d *h2) nogil
  
  int gsl_histogram2d_sub(gsl_histogram2d *h1,  gsl_histogram2d *h2) nogil
  
  int gsl_histogram2d_mul(gsl_histogram2d *h1,  gsl_histogram2d *h2) nogil
  
  int gsl_histogram2d_div(gsl_histogram2d *h1,  gsl_histogram2d *h2) nogil
  
  int gsl_histogram2d_scale(gsl_histogram2d *h, double scale) nogil
  
  int gsl_histogram2d_shift(gsl_histogram2d *h, double shift) nogil
  
  int gsl_histogram2d_fwrite (FILE * stream,  gsl_histogram2d * h) nogil 
  int gsl_histogram2d_fread (FILE * stream, gsl_histogram2d * h) nogil
  int gsl_histogram2d_fprintf (FILE * stream,  gsl_histogram2d * h,  char * range_format,  char * bin_format) nogil
  int gsl_histogram2d_fscanf (FILE * stream, gsl_histogram2d * h) nogil
  
  gsl_histogram2d_pdf * gsl_histogram2d_pdf_alloc ( size_t nx,  size_t ny) nogil
  int gsl_histogram2d_pdf_init (gsl_histogram2d_pdf * p,  gsl_histogram2d * h) nogil
  void gsl_histogram2d_pdf_free (gsl_histogram2d_pdf * p) nogil
  int gsl_histogram2d_pdf_sample ( gsl_histogram2d_pdf * p, double r1, double r2, double * x, double * y) nogil
  
