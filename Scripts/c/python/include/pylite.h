#ifndef PYLITE_H
#define PYLITE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A small, standalone teaching implementation of CPython-like containers.
 * Containers own references to their elements. Functions returning an object
 * pointer return a borrowed reference unless the name ends in _new/_copy or is
 * a conversion constructor.
 */

typedef int64_t PyLiteHash;

typedef enum {
    PYLITE_NONE,
    PYLITE_INT,
    PYLITE_STRING,
    PYLITE_PAIR,
    PYLITE_COLLIDER
} PyLiteType;

typedef struct PyLiteObject PyLiteObject;
typedef struct PyLiteList PyLiteList;
typedef struct PyLiteSet PyLiteSet;
typedef struct PyLiteDict PyLiteDict;

typedef struct {
    uint64_t lookups;
    uint64_t probes;
    uint64_t comparisons;
    uint64_t resizes;
    uint64_t moved_slots;
} PyLiteStats;

typedef int (*PyLiteVisitFn)(PyLiteObject *value, void *context);
typedef int (*PyLiteDictVisitFn)(PyLiteObject *key, PyLiteObject *value,
                                 void *context);

/* Object layer --------------------------------------------------------- */

PyLiteObject *pylite_none(void);                 /* immortal, borrowed */
PyLiteObject *pylite_int_new(int64_t value);     /* new reference */
PyLiteObject *pylite_string_new(const char *s);  /* new reference */
PyLiteObject *pylite_pair_new(PyLiteObject *first, PyLiteObject *second);
/* A test object whose distinct values all hash to 42: Python can create the
 * same worst case with a class whose __hash__ always returns 42. */
PyLiteObject *pylite_collider_new(int64_t identity);

void pylite_incref(PyLiteObject *object);
void pylite_decref(PyLiteObject *object);
PyLiteType pylite_type(const PyLiteObject *object);
int64_t pylite_int_value(const PyLiteObject *object);
const char *pylite_string_value(const PyLiteObject *object);
PyLiteObject *pylite_pair_first(const PyLiteObject *object);   /* borrowed */
PyLiteObject *pylite_pair_second(const PyLiteObject *object);  /* borrowed */
PyLiteHash pylite_hash(const PyLiteObject *object);
bool pylite_equal(const PyLiteObject *left, const PyLiteObject *right);
int pylite_compare(const PyLiteObject *left, const PyLiteObject *right,
                   int *result);
void pylite_object_print(const PyLiteObject *object, FILE *stream);

/* Error and tracing helpers ------------------------------------------- */

const char *pylite_last_error(void);
void pylite_clear_error(void);
void pylite_trace_set(bool enabled);
bool pylite_trace_enabled(void);

/* List ---------------------------------------------------------------- */

PyLiteList *pylite_list_new(void);
PyLiteList *pylite_list_copy(const PyLiteList *list);
void pylite_list_free(PyLiteList *list);
size_t pylite_list_len(const PyLiteList *list);
size_t pylite_list_capacity(const PyLiteList *list);
PyLiteObject *pylite_list_get(const PyLiteList *list, ptrdiff_t index);
int pylite_list_set(PyLiteList *list, ptrdiff_t index, PyLiteObject *value);
int pylite_list_append(PyLiteList *list, PyLiteObject *value);
int pylite_list_extend(PyLiteList *list, const PyLiteList *other);
int pylite_list_insert(PyLiteList *list, ptrdiff_t index, PyLiteObject *value);
PyLiteObject *pylite_list_pop(PyLiteList *list, ptrdiff_t index); /* new ref */
int pylite_list_remove(PyLiteList *list, const PyLiteObject *value);
ptrdiff_t pylite_list_index(PyLiteList *list, const PyLiteObject *value,
                            ptrdiff_t start, ptrdiff_t stop);
size_t pylite_list_count(PyLiteList *list, const PyLiteObject *value);
bool pylite_list_contains(PyLiteList *list, const PyLiteObject *value);
void pylite_list_reverse(PyLiteList *list);
/* Stable O(n log n) educational sort. See README for the CPython difference. */
int pylite_list_sort(PyLiteList *list, bool reverse);
void pylite_list_clear(PyLiteList *list);
int pylite_list_visit(const PyLiteList *list, PyLiteVisitFn visit,
                      void *context);
const PyLiteStats *pylite_list_stats(const PyLiteList *list);
void pylite_list_reset_stats(PyLiteList *list);
void pylite_list_print(const PyLiteList *list, FILE *stream);

/* Set ----------------------------------------------------------------- */

PyLiteSet *pylite_set_new(void);
PyLiteSet *pylite_set_copy(const PyLiteSet *set);
void pylite_set_free(PyLiteSet *set);
size_t pylite_set_len(const PyLiteSet *set);
int pylite_set_add(PyLiteSet *set, PyLiteObject *key); /* 1 new, 0 existed */
bool pylite_set_contains(PyLiteSet *set, const PyLiteObject *key);
int pylite_set_discard(PyLiteSet *set, const PyLiteObject *key);
int pylite_set_remove(PyLiteSet *set, const PyLiteObject *key);
PyLiteObject *pylite_set_pop(PyLiteSet *set); /* new reference */
void pylite_set_clear(PyLiteSet *set);
int pylite_set_update(PyLiteSet *set, const PyLiteSet *other);
PyLiteSet *pylite_set_union(const PyLiteSet *left, const PyLiteSet *right);
PyLiteSet *pylite_set_intersection(PyLiteSet *left, PyLiteSet *right);
PyLiteSet *pylite_set_difference(PyLiteSet *left, PyLiteSet *right);
PyLiteSet *pylite_set_symmetric_difference(PyLiteSet *left,
                                            PyLiteSet *right);
int pylite_set_intersection_update(PyLiteSet *set, PyLiteSet *other);
int pylite_set_difference_update(PyLiteSet *set, PyLiteSet *other);
int pylite_set_symmetric_difference_update(PyLiteSet *set, PyLiteSet *other);
bool pylite_set_is_subset(PyLiteSet *left, PyLiteSet *right);
bool pylite_set_is_superset(PyLiteSet *left, PyLiteSet *right);
bool pylite_set_is_disjoint(PyLiteSet *left, PyLiteSet *right);
int pylite_set_visit(const PyLiteSet *set, PyLiteVisitFn visit, void *context);
const PyLiteStats *pylite_set_stats(const PyLiteSet *set);
void pylite_set_reset_stats(PyLiteSet *set);
void pylite_set_print(const PyLiteSet *set, FILE *stream);

/* Dict ---------------------------------------------------------------- */

PyLiteDict *pylite_dict_new(void);
PyLiteDict *pylite_dict_copy(const PyLiteDict *dict);
void pylite_dict_free(PyLiteDict *dict);
size_t pylite_dict_len(const PyLiteDict *dict);
int pylite_dict_set(PyLiteDict *dict, PyLiteObject *key, PyLiteObject *value);
PyLiteObject *pylite_dict_get(PyLiteDict *dict, const PyLiteObject *key);
PyLiteObject *pylite_dict_get_default(PyLiteDict *dict,
                                      const PyLiteObject *key,
                                      PyLiteObject *default_value);
bool pylite_dict_contains(PyLiteDict *dict, const PyLiteObject *key);
int pylite_dict_del(PyLiteDict *dict, const PyLiteObject *key);
PyLiteObject *pylite_dict_setdefault(PyLiteDict *dict, PyLiteObject *key,
                                     PyLiteObject *default_value);
PyLiteObject *pylite_dict_pop(PyLiteDict *dict, const PyLiteObject *key,
                              PyLiteObject *default_value); /* new ref */
PyLiteObject *pylite_dict_popitem(PyLiteDict *dict); /* new pair ref */
void pylite_dict_clear(PyLiteDict *dict);
int pylite_dict_update(PyLiteDict *dict, const PyLiteDict *other);
int pylite_dict_visit(const PyLiteDict *dict, PyLiteDictVisitFn visit,
                      void *context);
const PyLiteStats *pylite_dict_stats(const PyLiteDict *dict);
void pylite_dict_reset_stats(PyLiteDict *dict);
void pylite_dict_print(const PyLiteDict *dict, FILE *stream);

/* Python-style conversions ------------------------------------------- */

PyLiteList *pylite_list_from_set(const PyLiteSet *set);          /* list(set) */
PyLiteList *pylite_list_from_dict(const PyLiteDict *dict);       /* list(dict) */
PyLiteList *pylite_list_from_dict_items(const PyLiteDict *dict); /* list(d.items()) */
PyLiteList *pylite_list_from_dict_values(const PyLiteDict *dict);/* list(d.values()) */
PyLiteSet *pylite_set_from_list(const PyLiteList *list);         /* set(list) */
PyLiteSet *pylite_set_from_dict(const PyLiteDict *dict);         /* set(dict) */
PyLiteDict *pylite_dict_from_pairs(const PyLiteList *pairs);     /* dict(pairs) */
PyLiteDict *pylite_dict_fromkeys_list(const PyLiteList *keys,
                                      PyLiteObject *default_value);
PyLiteDict *pylite_dict_fromkeys_set(const PyLiteSet *keys,
                                     PyLiteObject *default_value);

#ifdef __cplusplus
}
#endif

#endif
