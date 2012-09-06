/* win32dir.h - emulate opendir/readdir/closedir on MS Windows systems
 *
 * This is a modified version of the file build/win32/dirent/dirent.h
 * as distributed by glib-2.12.4.  The original file comes with the
 * following notice:
 *
 *   DIRENT.H (formerly DIRLIB.H) This file has no copyright assigned
 *   and is placed in the Public Domain.  This file is a part of the
 *   mingw-runtime package.  No warranty is given; refer to the file
 *   DISCLAIMER within the package.
 *
 * All changed from the original file are copyright (C) 2006 Jochen
 * Voss and can be redistributed under the terms of the GNU General
 * Public License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 */

#ifndef FILE_WIN32DIR_H_SEEN
#define FILE_WIN32DIR_H_SEEN

#include <stdio.h>
#include <io.h>

struct dirent
{
	long		d_ino;		/* Always zero. */
	unsigned short	d_reclen;	/* Always zero. */
	unsigned short	d_namlen;	/* Length of name in d_name. */
	char		d_name[FILENAME_MAX]; /* File name. */
};

/*
 * This is an internal data structure. Good programmers will not use it
 * except as an argument to one of the functions below.
 * dd_stat field is now int (was short in older versions).
 */
typedef struct
{
	/* disk transfer area for this dir */
	struct _finddata_t	dd_dta;

	/* dirent struct to return from dir (NOTE: this makes this thread
	 * safe as long as only one thread uses a particular DIR struct at
	 * a time) */
	struct dirent		dd_dir;

	/* _findnext handle */
	long			dd_handle;

	/*
         * Status of search:
	 *   0 = not started yet (next entry to read is first entry)
	 *  -1 = off the end
	 *   positive = 0 based index of next entry
	 */
	int			dd_stat;

	/* given path for dir with search pattern (struct is extended) */
	char			dd_name[1];
} DIR;

DIR* __cdecl opendir (const char*);
struct dirent* __cdecl readdir (DIR*);
int __cdecl closedir (DIR*);


/* wide char versions */

struct _wdirent
{
	long		d_ino;		/* Always zero. */
	unsigned short	d_reclen;	/* Always zero. */
	unsigned short	d_namlen;	/* Length of name in d_name. */
	wchar_t		d_name[FILENAME_MAX]; /* File name. */
};

/*
 * This is an internal data structure. Good programmers will not use it
 * except as an argument to one of the functions below.
 */
typedef struct
{
	/* disk transfer area for this dir */
	struct _wfinddata_t	dd_dta;

	/* dirent struct to return from dir (NOTE: this makes this thread
	 * safe as long as only one thread uses a particular DIR struct at
	 * a time) */
	struct _wdirent		dd_dir;

	/* _findnext handle */
	long			dd_handle;

	/*
         * Status of search:
	 *   0 = not started yet (next entry to read is first entry)
	 *  -1 = off the end
	 *   positive = 0 based index of next entry
	 */
	int			dd_stat;

	/* given path for dir with search pattern (struct is extended) */
	wchar_t			dd_name[1];
} _WDIR;



_WDIR* __cdecl _wopendir (const wchar_t*);
struct _wdirent*  __cdecl _wreaddir (_WDIR*);
int __cdecl _wclosedir (_WDIR*);

#endif /* FILE_WIN32DIR_H_SEEN */
