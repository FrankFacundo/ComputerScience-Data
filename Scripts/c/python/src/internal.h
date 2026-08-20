#ifndef PYLITE_INTERNAL_H
#define PYLITE_INTERNAL_H

#include "pylite.h"

#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define PYLITE_SET_MINSIZE 8u
#define PYLITE_DICT_MINSIZE 8u
#define PYLITE_PERTURB_SHIFT 5u
#define PYLITE_LINEAR_PROBES 9u
#define PYLITE_DICT_EMPTY ((ptrdiff_t)-1)
#define PYLITE_DICT_DUMMY ((ptrdiff_t)-2)
#define PYLITE_HASH_ERROR ((PyLiteHash)-1)

typedef struct {
    PyLiteObject *first;
    PyLiteObject *second;
} PyLitePair;

struct PyLiteObject {
    size_t refcount;
    PyLiteType type;
    bool hash_cached;
    PyLiteHash cached_hash;
    union {
        int64_t integer;
        struct {
            size_t length;
            char *bytes;
        } string;
        PyLitePair pair;
        int64_t collider_identity;
    } as;
};

struct PyLiteList {
    size_t size;
    size_t allocated;
    PyLiteObject **items;
    PyLiteStats stats;
};

typedef struct {
    PyLiteObject *key;
    PyLiteHash hash;
} PyLiteSetEntry;

struct PyLiteSet {
    size_t fill;
    size_t used;
    size_t mask;
    PyLiteSetEntry *table;
    size_t finger;
    PyLiteSetEntry smalltable[PYLITE_SET_MINSIZE];
    PyLiteStats stats;
};

typedef struct {
    PyLiteHash hash;
    PyLiteObject *key;
    PyLiteObject *value;
} PyLiteDictEntry;

struct PyLiteDict {
    size_t used;
    size_t table_size;
    size_t usable;
    size_t nentries;
    unsigned index_width;
    void *indices;
    PyLiteDictEntry *entries;
    PyLiteStats stats;
};

extern PyLiteObject pylite_dummy_object;

void pylite_error_set(const char *format, ...);
void pylite_trace(const char *format, ...);
bool pylite_size_multiply(size_t left, size_t right, size_t *result);

/* Intentionally visible in debug symbols: these are useful breakpoints. */
int pylite_list_resize(PyLiteList *list, size_t new_size);
PyLiteSetEntry *pylite_set_lookup(PyLiteSet *set, const PyLiteObject *key,
                                  PyLiteHash hash);
int pylite_set_table_resize(PyLiteSet *set, size_t minimum_used);
ptrdiff_t pylite_dict_lookup(PyLiteDict *dict, const PyLiteObject *key,
                             PyLiteHash hash, size_t *hash_position);
int pylite_dict_resize(PyLiteDict *dict, size_t minimum_used);

#endif
