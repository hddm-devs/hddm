/* container.c - container data types for storing strings
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

#include <string.h>
#include <stdlib.h>

#include "fast-dm.h"

/**********************************************************************
 * sets of strings (use only for small sets)
 */

struct set *
new_set (void)
{
	struct set *set;
	set = xnew(struct set, 1);
	set->used = 0;
	set->alloc = 8;
	set->item = xnew(char *, set->alloc);
	return set;
}

void
delete_set (struct set *set)
{
	int i;

	for (i = 0; i < set->used; ++i) {
		xfree(set->item[i]);
	}
	xfree(set->item);
	xfree(set);
}

int
set_item (struct set *set, const char *item, int add)
/* Test whether 'set' contains 'item' and conditionally add 'item'.
 *
 * This function tests whether a string equal to 'item' is already
 * contained in 'set'.  If this is the case, the index of the
 * corresponding entry in set is returned.  Otherwise it 'add' is
 * true, a copy of 'item' is added to 'set' and the index of the newly
 * added element is returned.  Otherwise -1 is returned.
 */
{
	int i;

	for (i = 0; i < set->used; ++i) {
		if (strcmp (set->item[i], item) == 0)
			return i;
	}
	if (! add)  return -1;
	if (set->used >= set->alloc) {
		set->alloc += 8;
		set->item = xrenew (char *, set->item, set->alloc);
	}
	set->item[set->used] = xstrdup (item);
	return  set->used++;
}

/**********************************************************************
 * dictionaries of strings (use only for small dictionaries)
 */

struct dict {
	int  used, alloc;
	char **keys;
	char **values;
};

struct dict *
new_dict (void)
{
	struct dict *dict;
	dict = xnew(struct dict, 1);
	dict->used = 0;
	dict->alloc = 8;
	dict->keys = xnew(char *, dict->alloc);
	dict->values = xnew(char *, dict->alloc);
	return dict;
}

void
dict_clear (struct dict *dict)
{
	int i;

	for (i = 0; i < dict->used; ++i) {
		xfree(dict->keys[i]);
		xfree(dict->values[i]);
	}
	dict->used = 0;
}

void
delete_dict (struct dict *dict)
{
	dict_clear (dict);
	xfree(dict->keys);
	xfree(dict->values);
	xfree(dict);
}

void
dict_add (struct dict *dict, const char *key, const char *value)
{
	int i;

	for (i = 0; i < dict->used; ++i) {
		if (strcmp (dict->keys[i], key) == 0) {
			xfree(dict->values[i]);
			dict->values[i] = xstrdup(value);
			return;
		}
	}
	if (dict->used >= dict->alloc) {
		dict->alloc += 8;
		dict->keys = xrenew (char *, dict->keys, dict->alloc);
		dict->values = xrenew (char *, dict->values, dict->alloc);
	}
	dict->keys[dict->used] = xstrdup (key);
	dict->values[dict->used] = xstrdup (value);
	dict->used++;
}

const char *
dict_lookup (const struct dict *dict, const char *key)
{
	int i;

	for (i = 0; i < dict->used; ++i) {
		if (strcmp (dict->keys[i], key) == 0) {
			return  dict->values[i];
		}
	}
	return  NULL;
}

/**********************************************************************
 * arrays of strings
 */

struct array *
new_array (void)
{
	struct array *array;
	array = xnew(struct array, 1);
	array->used = 0;
	array->alloc = 16;
	array->entry = xnew(char *, array->alloc);
	return array;
}

void
delete_array (struct array *array)
{
	int  i;
	for (i=0; i<array->used; ++i)
		xfree(array->entry[i]);
	xfree(array->entry);
	xfree(array);
}

void
array_clear (struct array *array)
{
	int  i;
	for (i=0; i<array->used; ++i)
		xfree(array->entry[i]);
	array->used = 0;
}

int
array_find (struct array *array, const char *str)
{
	int i;

	for (i = 0; i < array->used; ++i) {
		if (strcmp (array->entry[i], str) == 0)
			return i;
	}
	return  -1;
}


void
array_append (struct array *array, const char *str)
{
	if (array->used >= array->alloc) {
		array->alloc += 16;
		array->entry = xrenew(char *, array->entry, array->alloc);
	}
	array->entry[array->used] = xstrdup(str);
	array->used += 1;
}

static int
compare_strings (const void *a, const void *b)
{
	const char *aa = *(const char **)a;
	const char *bb = *(const char **)b;
	return  strcmp(aa, bb);
}

void
array_sort (struct array *array)
{
	qsort(array->entry, array->used, sizeof(const char *),
	      compare_strings);
}


/*
 * Local Variables:
 * c-file-style: "linux"
 * End:
 */
