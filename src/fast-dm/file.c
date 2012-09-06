/* file.c - handle control and data files
 *
 * Copyright (C) 2006  Jochen Voss, Andreas Voss.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <errno.h>
#ifndef _WIN32
#include <sys/types.h>
#include <dirent.h>
#define PATH_SEP '/'
#else
#include "win32dir.h"
#define PATH_SEP '\\'
#endif

#include "fast-dm.h"


/**********************************************************************
 * handle file names
 */

struct array *
file_names_find (const char *pattern)
{
	const char *ptr, *star;
	char *path, *pre, *post;
	size_t  pre_len, post_len;
	struct array *keys;

	/* find the path for 'pattern' */
	ptr = strrchr(pattern, PATH_SEP);
	if (ptr) {
		path = xstrndup(pattern, ptr-pattern);
		pattern = ptr+1;
	} else {
		path = xstrdup(".");
	}

	/* find the globbing character "*" */
	star = NULL;
	for (ptr=pattern; *ptr; ++ptr) {
		if (*ptr != '*')  continue;
		if (star) {
			fprintf(stderr, "error: invalid pattern \"%s\"\n",
				pattern);
			exit(0);
		}
		star = ptr;
	}
	if (star) {
		pre = xstrndup(pattern, star-pattern);
		post = xstrndup(star+1, ptr-star-1);
	} else {
		pre = xstrdup(pattern);
		post = xstrdup("");
	}
	pre_len = strlen(pre);
	post_len = strlen(post);

	keys = new_array();
	if (star) {
		DIR *dp;
		struct dirent *ep;

		dp = opendir(path);
		if (!dp) {
			perror("opendir failed");
			exit(1);
		}
		while ((ep = readdir(dp))) {
			const char *fname = ep->d_name;
			size_t fname_len = strlen(fname);
			char *key;

			if (strncmp(fname,pre,pre_len) != 0)
				continue;
			if (strcmp(fname+fname_len-post_len,post) != 0)
				continue;
			key = xstrndup(fname+pre_len,
				       fname_len-pre_len-post_len);
			array_append(keys, key);
			xfree(key);
		}
		closedir (dp);
	} else {
		array_append(keys, "");
	}
	array_sort(keys);

	xfree(post);
	xfree(pre);
	xfree(path);

	return  keys;
}

char *
file_names_replace_star(const char *template, const char *key)
{
	const char *ptr;
	size_t  template_len, pre_len;
	char *res;

	ptr = strchr(template, '*');
	if (! ptr)  return  xstrdup(template);

	template_len = strlen(template);
	pre_len = ptr-template;

	res = xnew(char, template_len-1+strlen(key)+1);
	res[0] = '\0';
	strncat(res, template, pre_len);
	strcat(res, key);
	strcat(res, ptr+1);
	return  res;
}

/**********************************************************************
 * handle file contents
 */

struct file {
	FILE *fd;
	char *fname;
	int line_no;

	int line_used, line_alloc;
	char *line;

	int tokens_used, tokens_alloc;
	const char **tokens;
};

struct file *
new_file (const char *fname)
{
	struct file *file;

	file = xnew (struct file, 1);
	file->fd = fopen (fname, "r");
	if (!file->fd) {
		if (errno == ENOENT) {
			xfree (file);
			return NULL;
		} else {
			fprintf (stderr, "%s: open failed (%s)\n",
				 fname, strerror (errno));
			exit (1);
		}
	}

	file->fname = xstrdup (fname);
	file->line_no = 0;

	file->line_used = 0;
	file->line_alloc = 80;
	file->line = xnew (char, file->line_alloc);

	file->tokens_used = 0;
	file->tokens_alloc = 80;
	file->tokens = xnew (const char *, file->tokens_alloc);

	return file;
}

void
delete_file (struct file *file)
{
	fclose (file->fd);
	xfree (file->tokens);
	xfree (file->line);
	xfree (file->fname);
	xfree (file);
}

const char *
file_name (const struct file *file)
{
	return  file->fname;
}

static void
file_store_char (struct file *file, char c)
{
	if (file->line_used >= file->line_alloc) {
		file->line_alloc += 80;
		file->line = xrenew (char, file->line, file->line_alloc);
	}
	file->line[file->line_used++] = c;
}

static void
file_start_token (struct file *file)
{
	file_store_char (file, '\0');
	if (file->tokens_used >= file->tokens_alloc) {
		file->tokens_alloc += 80;
		file->tokens = xrenew (const char *, file->tokens,
				       file->tokens_alloc);
	}
	file->tokens[file->tokens_used++] = file->line + file->line_used;
}

void
file_error (struct file *file, const char *format, ...)
/* Abort the program with an error message. */
{
	va_list  ap;

	va_start (ap, format);
	fprintf (stderr, "%s:%d: ", file->fname, file->line_no);
	vfprintf (stderr, format, ap);
	fprintf (stderr, ", aborting\n");
	va_end (ap);
	exit (1);
}

void
file_message (struct file *file, const char *format, ...)
/* print an error message */
{
	va_list  ap;

	va_start (ap, format);
	fprintf (stderr, "%s:%d: ", file->fname, file->line_no);
	vfprintf (stderr, format, ap);
	fprintf (stderr, "\n");
	va_end (ap);
}

int
file_read (struct file *file, const char *const**w_ptr, int *n_ptr)
/* Read a new input line and convert it into tokens.
 *
 * All lines must be terminated by an end of line.  The function
 * returns 0 on end of file.  Otherwise 1 is returned and
 * '*w_ptr' contains '*n_ptr' tokens of a complete, non-empty input
 * line.
 */
{
	enum { s_start, s_token, s_space, s_string, s_comment } state;
	int c, done;

	file->line_no += 1;
	file->line_used = 0;
	file->tokens_used = 0;
	state = s_start;
	done = 0;
	while (!done) {
		c = fgetc (file->fd);
		if (c == EOF && ferror(file->fd) != 0) {
			fprintf (stderr, "%s:%d: read failed (%s)\n",
				 file->fname, file->line_no,
				 strerror (errno));
			exit (1);
		}
		switch (state) {
		case s_start:
			if (c == EOF) {
				done = 1;
			}
			else if (c == '\n') {
				file->line_no += 1;
			}
			else if (isspace (c)) {
				/* do nothing */
			}
			else if (c == '#') {
				state = s_comment;
			}
			else if (c == '"') {
				file_start_token (file);
				state = s_string;
			}
			else if (isgraph (c)) {
				file_start_token (file);
				file_store_char (file, c);
				state = s_token;
			}
			else {
				fprintf (stderr,
					 "%s:%d: unexpected character '%c', "
					 "aborting\n",
					 file->fname, file->line_no, c);
				exit (1);
			}
			break;
		case s_space:
			if (c == EOF) {
				fprintf (stderr,
					 "%s:%d: warning: "
					 "unexpected end of file\n",
					 file->fname, file->line_no);
				done = 1;
			}
			else if (c == '\n' || c == '\r') {
				done = 1;
			}
			else if (isspace (c)) {
				/* do nothing */
			}
			else if (c == '#') {
				state = s_comment;
			}
			else if (c == '"') {
				file_start_token (file);
				state = s_string;
			}
			else if (isgraph (c)) {
				file_start_token (file);
				file_store_char (file, c);
				state = s_token;
			}
			else {
				fprintf (stderr,
					 "%s:%d: unexpected character '%c', "
					 "aborting\n",
					 file->fname, file->line_no, c);
				exit (1);
			}
			break;
		case s_token:
			if (c == EOF) {
				fprintf (stderr,
					 "%s:%d: warning: "
					 "unexpected end of file\n",
					 file->fname, file->line_no);
				done = 1;
			}
			else if (c == '\n' || c == '\r') {
				done = 1;
			}
			else if (isspace (c)) {
				state = s_space;
			}
			else if (c == '#') {
				state = s_comment;
			}
			else if (c == '"') {
				state = s_string;
			}
			else if (isgraph (c)) {
				file_store_char (file, c);
			}
			else {
				fprintf (stderr,
					 "%s:%d: unexpected character '%c', "
					 "aborting\n",
					 file->fname, file->line_no, c);
				exit (1);
			}
			break;
		case s_string:
			if (c == EOF || c == '\n' || c == '\r') {
				file_error (file, "unterminated string");
			}
			else if (c == '"') {
				state = s_token;
			}
			else {
				file_store_char (file, c);
			}
			break;
		case s_comment:
			if (c == EOF) {
				fprintf (stderr,
					 "%s:%d: warning: "
					 "unexpected end of file\n",
					 file->fname, file->line_no);
				done = 1;
			}
			else if (c == '\n' || c == '\r') {
				if (file->tokens_used > 0) {
					done = 1;
				}
				else {
					file->line_no += 1;
					state = s_start;
				}
			}
			else {
				/* do nothing */
			}
			break;
		}
	}
	file_store_char (file, '\0');

	*w_ptr = file->tokens;
	*n_ptr = file->tokens_used;
	return (c != EOF);
}

/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
