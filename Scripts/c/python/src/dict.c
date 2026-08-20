#include "internal.h"

#include <stdint.h>

static size_t
usable_fraction(size_t table_size)
{
    return (table_size << 1) / 3;
}

static unsigned
index_width_for(size_t table_size)
{
    if (table_size <= 128) {
        return 1;
    }
    if (table_size <= 32768) {
        return 2;
    }
#if SIZE_MAX > UINT32_MAX
    if (table_size > UINT32_MAX) {
        return 8;
    }
#endif
    return 4;
}

static ptrdiff_t
dict_index_get(const PyLiteDict *dict, size_t position)
{
    switch (dict->index_width) {
        case 1:
            return ((const int8_t *)dict->indices)[position];
        case 2:
            return ((const int16_t *)dict->indices)[position];
        case 4:
            return ((const int32_t *)dict->indices)[position];
        default:
            return (ptrdiff_t)((const int64_t *)dict->indices)[position];
    }
}

static void
dict_index_set(PyLiteDict *dict, size_t position, ptrdiff_t value)
{
    switch (dict->index_width) {
        case 1:
            ((int8_t *)dict->indices)[position] = (int8_t)value;
            break;
        case 2:
            ((int16_t *)dict->indices)[position] = (int16_t)value;
            break;
        case 4:
            ((int32_t *)dict->indices)[position] = (int32_t)value;
            break;
        default:
            ((int64_t *)dict->indices)[position] = (int64_t)value;
            break;
    }
}

static int
allocate_tables(size_t table_size, unsigned *width_out, void **indices_out,
                PyLiteDictEntry **entries_out)
{
    unsigned width = index_width_for(table_size);
    size_t index_bytes;
    size_t entry_bytes;
    if (!pylite_size_multiply(table_size, width, &index_bytes) ||
        !pylite_size_multiply(usable_fraction(table_size),
                              sizeof(PyLiteDictEntry), &entry_bytes)) {
        return -1;
    }
    void *indices = malloc(index_bytes);
    PyLiteDictEntry *entries = calloc(1, entry_bytes);
    if (indices == NULL || entries == NULL) {
        free(indices);
        free(entries);
        pylite_error_set("out of memory allocating dict table");
        return -1;
    }
    memset(indices, 0xff, index_bytes); /* two's-complement -1 / EMPTY */
    *width_out = width;
    *indices_out = indices;
    *entries_out = entries;
    return 0;
}

PyLiteDict *
pylite_dict_new(void)
{
    PyLiteDict *dict = calloc(1, sizeof(*dict));
    if (dict == NULL) {
        pylite_error_set("out of memory allocating dict");
        return NULL;
    }
    if (allocate_tables(PYLITE_DICT_MINSIZE, &dict->index_width,
                        &dict->indices, &dict->entries) < 0) {
        free(dict);
        return NULL;
    }
    dict->table_size = PYLITE_DICT_MINSIZE;
    dict->usable = usable_fraction(PYLITE_DICT_MINSIZE);
    return dict;
}

static size_t
find_empty_slot(PyLiteDict *dict, PyLiteHash hash)
{
    size_t mask = dict->table_size - 1;
    size_t perturb = (size_t)hash;
    size_t index = (size_t)hash & mask;
    while (dict_index_get(dict, index) >= 0) {
        perturb >>= PYLITE_PERTURB_SHIFT;
        index = (index * 5 + perturb + 1) & mask;
    }
    return index;
}

/*
 * Compact dict lookup. indices[] is the sparse hash table and entries[] is a
 * dense insertion-order array. The two-probe inner loop mirrors CPython
 * 3.14.7's manual loop unrolling.
 */
ptrdiff_t
pylite_dict_lookup(PyLiteDict *dict, const PyLiteObject *key,
                   PyLiteHash hash, size_t *hash_position)
{
    if (dict == NULL || key == NULL) {
        pylite_error_set("invalid dict lookup");
        return PYLITE_DICT_EMPTY;
    }
    size_t mask = dict->table_size - 1;
    size_t perturb = (size_t)hash;
    size_t index = (size_t)hash & mask;
    dict->stats.lookups++;

    for (;;) {
        for (unsigned unrolled = 0; unrolled < 2; unrolled++) {
            ptrdiff_t entry_index = dict_index_get(dict, index);
            dict->stats.probes++;
            pylite_trace("dict_lookup hash=%lld slot=%zu entry=%td",
                         (long long)hash, index, entry_index);
            if (entry_index >= 0) {
                PyLiteDictEntry *entry = &dict->entries[entry_index];
                if (entry->hash == hash) {
                    dict->stats.comparisons++;
                    if (entry->key == key || pylite_equal(entry->key, key)) {
                        if (hash_position != NULL) {
                            *hash_position = index;
                        }
                        return entry_index;
                    }
                }
            }
            else if (entry_index == PYLITE_DICT_EMPTY) {
                if (hash_position != NULL) {
                    *hash_position = index;
                }
                return PYLITE_DICT_EMPTY;
            }
            perturb >>= PYLITE_PERTURB_SHIFT;
            index = (index * 5 + perturb + 1) & mask;
        }
    }
}

static void
build_indices(PyLiteDict *dict)
{
    size_t mask = dict->table_size - 1;
    for (size_t entry_index = 0; entry_index < dict->nentries; entry_index++) {
        PyLiteHash hash = dict->entries[entry_index].hash;
        size_t perturb = (size_t)hash;
        size_t position = (size_t)hash & mask;
        while (dict_index_get(dict, position) != PYLITE_DICT_EMPTY) {
            perturb >>= PYLITE_PERTURB_SHIFT;
            position = (position * 5 + perturb + 1) & mask;
        }
        dict_index_set(dict, position, (ptrdiff_t)entry_index);
    }
}

int
pylite_dict_resize(PyLiteDict *dict, size_t minimum_size)
{
    if (dict == NULL) {
        pylite_error_set("dict is NULL");
        return -1;
    }
    size_t new_size = PYLITE_DICT_MINSIZE;
    while (new_size < minimum_size) {
        if (new_size > SIZE_MAX / 2) {
            pylite_error_set("dict is too large");
            return -1;
        }
        new_size <<= 1;
    }
    while (usable_fraction(new_size) < dict->used) {
        if (new_size > SIZE_MAX / 2) {
            pylite_error_set("dict is too large");
            return -1;
        }
        new_size <<= 1;
    }

    unsigned new_width;
    void *new_indices;
    PyLiteDictEntry *new_entries;
    if (allocate_tables(new_size, &new_width, &new_indices, &new_entries) < 0) {
        return -1;
    }

    size_t output = 0;
    for (size_t i = 0; i < dict->nentries; i++) {
        if (dict->entries[i].value != NULL) {
            new_entries[output++] = dict->entries[i];
            dict->stats.moved_slots++;
        }
    }
    size_t old_size = dict->table_size;
    free(dict->indices);
    free(dict->entries);
    dict->indices = new_indices;
    dict->entries = new_entries;
    dict->index_width = new_width;
    dict->table_size = new_size;
    dict->nentries = output;
    dict->usable = usable_fraction(new_size) - output;
    build_indices(dict);
    dict->stats.resizes++;
    pylite_trace("dict_resize slots=%zu->%zu used=%zu index_width=%u",
                 old_size, new_size, dict->used, new_width);
    return 0;
}

int
pylite_dict_set(PyLiteDict *dict, PyLiteObject *key, PyLiteObject *value)
{
    if (dict == NULL || key == NULL || value == NULL) {
        pylite_error_set("invalid dict assignment");
        return -1;
    }
    PyLiteHash hash = pylite_hash(key);
    if (hash == PYLITE_HASH_ERROR) {
        return -1;
    }
    ptrdiff_t found = pylite_dict_lookup(dict, key, hash, NULL);
    if (found >= 0) {
        PyLiteObject *old_value = dict->entries[found].value;
        pylite_incref(value);
        dict->entries[found].value = value;
        pylite_decref(old_value);
        return 0;
    }

    if (dict->usable == 0) {
        if (dict->used > SIZE_MAX / 3 ||
            pylite_dict_resize(dict, dict->used * 3) < 0) {
            return -1;
        }
    }
    size_t hash_position = find_empty_slot(dict, hash);
    size_t entry_index = dict->nentries;
    dict_index_set(dict, hash_position, (ptrdiff_t)entry_index);
    pylite_incref(key);
    pylite_incref(value);
    dict->entries[entry_index].hash = hash;
    dict->entries[entry_index].key = key;
    dict->entries[entry_index].value = value;
    dict->nentries++;
    dict->usable--;
    dict->used++;
    return 0;
}

PyLiteObject *
pylite_dict_get(PyLiteDict *dict, const PyLiteObject *key)
{
    if (dict == NULL || key == NULL) {
        pylite_error_set("invalid dict get");
        return NULL;
    }
    PyLiteHash hash = pylite_hash(key);
    if (hash == PYLITE_HASH_ERROR) {
        return NULL;
    }
    ptrdiff_t found = pylite_dict_lookup(dict, key, hash, NULL);
    return found >= 0 ? dict->entries[found].value : NULL;
}

PyLiteObject *
pylite_dict_get_default(PyLiteDict *dict, const PyLiteObject *key,
                        PyLiteObject *default_value)
{
    PyLiteObject *value = pylite_dict_get(dict, key);
    return value != NULL ? value : default_value;
}

bool
pylite_dict_contains(PyLiteDict *dict, const PyLiteObject *key)
{
    if (dict == NULL || key == NULL) {
        return false;
    }
    PyLiteHash hash = pylite_hash(key);
    return hash != PYLITE_HASH_ERROR &&
        pylite_dict_lookup(dict, key, hash, NULL) >= 0;
}

static PyLiteObject *
delete_at(PyLiteDict *dict, size_t entry_index, size_t hash_position)
{
    PyLiteDictEntry *entry = &dict->entries[entry_index];
    PyLiteObject *key = entry->key;
    PyLiteObject *value = entry->value;
    dict_index_set(dict, hash_position, PYLITE_DICT_DUMMY);
    entry->hash = 0;
    entry->key = NULL;
    entry->value = NULL;
    dict->used--;
    pylite_decref(key);
    return value; /* transfer dict's reference */
}

int
pylite_dict_del(PyLiteDict *dict, const PyLiteObject *key)
{
    if (dict == NULL || key == NULL) {
        pylite_error_set("invalid dict deletion");
        return -1;
    }
    PyLiteHash hash = pylite_hash(key);
    size_t hash_position;
    ptrdiff_t found = hash == PYLITE_HASH_ERROR ? PYLITE_DICT_EMPTY :
        pylite_dict_lookup(dict, key, hash, &hash_position);
    if (found < 0) {
        pylite_error_set("dict key not found");
        return -1;
    }
    PyLiteObject *value = delete_at(dict, (size_t)found, hash_position);
    pylite_decref(value);
    return 0;
}

PyLiteObject *
pylite_dict_setdefault(PyLiteDict *dict, PyLiteObject *key,
                       PyLiteObject *default_value)
{
    if (dict == NULL || key == NULL || default_value == NULL) {
        pylite_error_set("invalid dict setdefault");
        return NULL;
    }
    PyLiteObject *value = pylite_dict_get(dict, key);
    if (value != NULL) {
        return value;
    }
    if (pylite_dict_set(dict, key, default_value) < 0) {
        return NULL;
    }
    return default_value;
}

PyLiteObject *
pylite_dict_pop(PyLiteDict *dict, const PyLiteObject *key,
                 PyLiteObject *default_value)
{
    if (dict == NULL || key == NULL) {
        pylite_error_set("invalid dict pop");
        return NULL;
    }
    PyLiteHash hash = pylite_hash(key);
    size_t hash_position;
    ptrdiff_t found = hash == PYLITE_HASH_ERROR ? PYLITE_DICT_EMPTY :
        pylite_dict_lookup(dict, key, hash, &hash_position);
    if (found < 0) {
        if (default_value != NULL) {
            pylite_incref(default_value);
            return default_value;
        }
        pylite_error_set("dict key not found");
        return NULL;
    }
    return delete_at(dict, (size_t)found, hash_position);
}

PyLiteObject *
pylite_dict_popitem(PyLiteDict *dict)
{
    if (dict == NULL || dict->used == 0) {
        pylite_error_set("popitem(): dictionary is empty");
        return NULL;
    }
    size_t entry_index = dict->nentries;
    do {
        entry_index--;
    } while (dict->entries[entry_index].value == NULL);

    PyLiteDictEntry *entry = &dict->entries[entry_index];
    size_t hash_position;
    ptrdiff_t found = pylite_dict_lookup(dict, entry->key, entry->hash,
                                          &hash_position);
    if (found < 0) {
        pylite_error_set("internal dict inconsistency");
        return NULL;
    }
    PyLiteObject *pair = pylite_pair_new(entry->key, entry->value);
    if (pair == NULL) {
        return NULL;
    }
    PyLiteObject *value = delete_at(dict, entry_index, hash_position);
    pylite_decref(value);
    /* Like CPython, discard trailing deleted entries but do not restore usable:
     * a DUMMY still exists in the sparse indices table. */
    dict->nentries = entry_index;
    return pair;
}

void
pylite_dict_clear(PyLiteDict *dict)
{
    if (dict == NULL) {
        return;
    }
    unsigned new_width = 0;
    void *new_indices = NULL;
    PyLiteDictEntry *new_entries = NULL;
    int allocated_small_table = allocate_tables(
        PYLITE_DICT_MINSIZE, &new_width, &new_indices, &new_entries);

    for (size_t i = 0; i < dict->nentries; i++) {
        if (dict->entries[i].value != NULL) {
            pylite_decref(dict->entries[i].key);
            pylite_decref(dict->entries[i].value);
        }
    }
    dict->used = 0;
    dict->nentries = 0;
    if (allocated_small_table == 0) {
        free(dict->indices);
        free(dict->entries);
        dict->indices = new_indices;
        dict->entries = new_entries;
        dict->index_width = new_width;
        dict->table_size = PYLITE_DICT_MINSIZE;
    }
    else {
        /* Clear cannot leave an unusable object if shrinking allocation fails. */
        memset(dict->indices, 0xff, dict->table_size * dict->index_width);
        memset(dict->entries, 0,
               usable_fraction(dict->table_size) * sizeof(*dict->entries));
        pylite_clear_error();
    }
    dict->usable = usable_fraction(dict->table_size);
}

void
pylite_dict_free(PyLiteDict *dict)
{
    if (dict == NULL) {
        return;
    }
    for (size_t i = 0; i < dict->nentries; i++) {
        if (dict->entries[i].value != NULL) {
            pylite_decref(dict->entries[i].key);
            pylite_decref(dict->entries[i].value);
        }
    }
    free(dict->indices);
    free(dict->entries);
    free(dict);
}

size_t
pylite_dict_len(const PyLiteDict *dict)
{
    return dict == NULL ? 0 : dict->used;
}

int
pylite_dict_update(PyLiteDict *dict, const PyLiteDict *other)
{
    if (dict == NULL || other == NULL) {
        pylite_error_set("invalid dict update");
        return -1;
    }
    if (dict == other) {
        return 0;
    }
    if (other->used > dict->usable) {
        size_t expected = dict->used + other->used;
        if (expected < dict->used || expected > (SIZE_MAX - 1) / 3) {
            pylite_error_set("dict is too large");
            return -1;
        }
        size_t minimum_table_size = (expected * 3 + 1) / 2;
        if (pylite_dict_resize(dict, minimum_table_size) < 0) {
            return -1;
        }
    }
    for (size_t i = 0; i < other->nentries; i++) {
        const PyLiteDictEntry *entry = &other->entries[i];
        if (entry->value != NULL &&
            pylite_dict_set(dict, entry->key, entry->value) < 0) {
            return -1;
        }
    }
    return 0;
}

PyLiteDict *
pylite_dict_copy(const PyLiteDict *dict)
{
    if (dict == NULL) {
        pylite_error_set("dict is NULL");
        return NULL;
    }
    PyLiteDict *copy = pylite_dict_new();
    if (copy == NULL || pylite_dict_update(copy, dict) < 0) {
        pylite_dict_free(copy);
        return NULL;
    }
    return copy;
}

int
pylite_dict_visit(const PyLiteDict *dict, PyLiteDictVisitFn visit,
                  void *context)
{
    if (dict == NULL || visit == NULL) {
        pylite_error_set("invalid dict visitor");
        return -1;
    }
    for (size_t i = 0; i < dict->nentries; i++) {
        if (dict->entries[i].value != NULL) {
            int result = visit(dict->entries[i].key, dict->entries[i].value,
                               context);
            if (result != 0) {
                return result;
            }
        }
    }
    return 0;
}

const PyLiteStats *
pylite_dict_stats(const PyLiteDict *dict)
{
    return dict == NULL ? NULL : &dict->stats;
}

void
pylite_dict_reset_stats(PyLiteDict *dict)
{
    if (dict != NULL) {
        memset(&dict->stats, 0, sizeof(dict->stats));
    }
}

void
pylite_dict_print(const PyLiteDict *dict, FILE *stream)
{
    if (stream == NULL) {
        stream = stdout;
    }
    if (dict == NULL) {
        fputs("<null dict>", stream);
        return;
    }
    fputc('{', stream);
    bool first = true;
    for (size_t i = 0; i < dict->nentries; i++) {
        if (dict->entries[i].value != NULL) {
            if (!first) {
                fputs(", ", stream);
            }
            pylite_object_print(dict->entries[i].key, stream);
            fputs(": ", stream);
            pylite_object_print(dict->entries[i].value, stream);
            first = false;
        }
    }
    fputc('}', stream);
}
