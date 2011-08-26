cdef extern from "stdio.h":
  ctypedef struct FILE
  FILE *fopen(char *path, char *mode) nogil
  int fclose(FILE *strea) nogil
  cdef FILE *stdout
  int scanf(char *format, ...) nogil

