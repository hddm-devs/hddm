/* xmalloc.c - memory allocation with error checking
 *
 * Copyright (C) 1998, 2003  Jochen Voss
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fast-dm.h"


static void
fatal (const char *message)
{
  fprintf (stderr, "error: %s, aborting\n", message);
  abort ();
}

void *
xmalloc (size_t size)
/* Like 'malloc', but check for shortage of memory.  For positive
 * sizes 'xmalloc' never returns 'NULL'.  */
{
  void *ptr;

  if (size == 0)  return NULL;
  ptr = malloc (size);
  if (ptr == NULL)  fatal ("memory exhausted");
  return  ptr;
}

void *
xrealloc (void *ptr, size_t newsize)
/* Like 'realloc', but check for shortage of memory.  For positive
 * sizes 'xrealloc' never returns 'NULL'.  */
{
  if (newsize == 0) {
    if (ptr)  free (ptr);
    return NULL;
  }
  ptr = ptr ? realloc (ptr, newsize) : malloc(newsize);
  if (ptr == NULL)  fatal ("memory exhausted");
  return  ptr;
}

void
xfree (void *ptr)
/* Like 'free', but ignores NULL pointers.  */
{
  if (ptr)  free (ptr);
}

char *
xstrdup (const char *s)
{
  char *ptr;

  ptr = strdup (s);
  if (ptr == NULL)  fatal ("memory exhausted");
  return  ptr;
}

char *
xstrndup (const char *s, size_t n)
{
  char *ptr;

  ptr = xmalloc(n+1);
  memcpy(ptr, s, n);
  ptr[n] = '\0';
  return  ptr;
}
