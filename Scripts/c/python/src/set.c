#include "internal.h"

static PyLiteObject *
dummy_key(void)
{
    return &pylite_dummy_object;
}

static bool
entry_is_empty(const PyLiteSetEntry *entry)
{
    return entry->key == NULL && entry->hash == 0;
}

static bool
entry_is_active(const PyLiteSetEntry *entry)
{
    return entry->key != NULL && entry->key != dummy_key();
}

PyLiteSet *
pylite_set_new(void)
{
    PyLiteSet *set = calloc(1, sizeof(*set));
    if (set == NULL) {
        pylite_error_set("out of memory allocating set");
        return NULL;
    }
    set->mask = PYLITE_SET_MINSIZE - 1;
    set->table = set->smalltable;
    return set;
}

/*
 * CPython 3.14 set lookup: up to 9 additional adjacent probes improve cache
 * locality, then the same 5*j+1+perturb recurrence used by dictionaries.
 */
PyLiteSetEntry *
pylite_set_lookup(PyLiteSet *set, const PyLiteObject *key, PyLiteHash hash)
{
    if (set == NULL || key == NULL) {
        pylite_error_set("invalid set lookup");
        return NULL;
    }
    size_t mask = set->mask;
    size_t perturb = (size_t)hash;
    size_t index = (size_t)hash & mask;
    set->stats.lookups++;

    for (;;) {
        PyLiteSetEntry *entry = &set->table[index];
        size_t extra = index + PYLITE_LINEAR_PROBES <= mask
            ? PYLITE_LINEAR_PROBES : 0;
        for (size_t probe = 0; probe <= extra; probe++, entry++) {
            set->stats.probes++;
            pylite_trace("set_lookup hash=%lld slot=%zu state=%s",
                         (long long)hash, (size_t)(entry - set->table),
                         entry_is_empty(entry) ? "empty" :
                         entry_is_active(entry) ? "active" : "dummy");
            if (entry_is_empty(entry)) {
                return entry;
            }
            if (entry->hash == hash && entry_is_active(entry)) {
                set->stats.comparisons++;
                if (entry->key == key || pylite_equal(entry->key, key)) {
                    return entry;
                }
            }
        }
        perturb >>= PYLITE_PERTURB_SHIFT;
        index = (index * 5 + 1 + perturb) & mask;
    }
}

static void
set_insert_clean(PyLiteSetEntry *table, size_t mask, PyLiteObject *key,
                 PyLiteHash hash)
{
    size_t perturb = (size_t)hash;
    size_t index = (size_t)hash & mask;
    for (;;) {
        PyLiteSetEntry *entry = &table[index];
        if (entry->key == NULL) {
            entry->key = key;
            entry->hash = hash;
            return;
        }
        if (index + PYLITE_LINEAR_PROBES <= mask) {
            for (size_t j = 0; j < PYLITE_LINEAR_PROBES; j++) {
                entry++;
                if (entry->key == NULL) {
                    entry->key = key;
                    entry->hash = hash;
                    return;
                }
            }
        }
        perturb >>= PYLITE_PERTURB_SHIFT;
        index = (index * 5 + 1 + perturb) & mask;
    }
}

int
pylite_set_table_resize(PyLiteSet *set, size_t minimum_used)
{
    if (set == NULL) {
        pylite_error_set("set is NULL");
        return -1;
    }
    size_t new_size = PYLITE_SET_MINSIZE;
    while (new_size <= minimum_used) {
        if (new_size > SIZE_MAX / 2) {
            pylite_error_set("set is too large");
            return -1;
        }
        new_size <<= 1;
    }

    PyLiteSetEntry *old_table = set->table;
    size_t old_mask = set->mask;
    bool old_was_allocated = old_table != set->smalltable;
    PyLiteSetEntry *allocated_table_to_free =
        old_was_allocated ? old_table : NULL;
    PyLiteSetEntry small_copy[PYLITE_SET_MINSIZE];
    PyLiteSetEntry *new_table;

    if (new_size == PYLITE_SET_MINSIZE) {
        new_table = set->smalltable;
        if (old_table == new_table) {
            if (set->fill == set->used) {
                return 0;
            }
            memcpy(small_copy, old_table, sizeof(small_copy));
            old_table = small_copy;
        }
    }
    else {
        size_t bytes;
        if (!pylite_size_multiply(new_size, sizeof(*new_table), &bytes)) {
            return -1;
        }
        new_table = malloc(bytes);
        if (new_table == NULL) {
            pylite_error_set("out of memory resizing set to %zu slots", new_size);
            return -1;
        }
    }

    memset(new_table, 0, new_size * sizeof(*new_table));
    set->table = new_table;
    set->mask = new_size - 1;
    set->fill = set->used;
    for (size_t i = 0; i <= old_mask; i++) {
        if (entry_is_active(&old_table[i])) {
            set_insert_clean(new_table, set->mask, old_table[i].key,
                             old_table[i].hash);
            set->stats.moved_slots++;
        }
    }
    if (old_was_allocated) {
        free(allocated_table_to_free);
    }
    set->stats.resizes++;
    pylite_trace("set_resize slots=%zu->%zu used=%zu",
                 old_mask + 1, new_size, set->used);
    return 0;
}

int
pylite_set_add(PyLiteSet *set, PyLiteObject *key)
{
    if (set == NULL || key == NULL) {
        pylite_error_set("invalid set add");
        return -1;
    }
    PyLiteHash hash = pylite_hash(key);
    if (hash == PYLITE_HASH_ERROR) {
        return -1;
    }
    size_t mask = set->mask;
    size_t perturb = (size_t)hash;
    size_t index = (size_t)hash & mask;
    PyLiteSetEntry *free_slot = NULL;
    set->stats.lookups++;

    for (;;) {
        PyLiteSetEntry *entry = &set->table[index];
        size_t extra = index + PYLITE_LINEAR_PROBES <= mask
            ? PYLITE_LINEAR_PROBES : 0;
        for (size_t probe = 0; probe <= extra; probe++, entry++) {
            set->stats.probes++;
            pylite_trace("set_add hash=%lld slot=%zu state=%s",
                         (long long)hash, (size_t)(entry - set->table),
                         entry_is_empty(entry) ? "empty" :
                         entry_is_active(entry) ? "active" : "dummy");
            if (entry_is_empty(entry)) {
                PyLiteSetEntry *destination = free_slot != NULL ? free_slot : entry;
                pylite_incref(key);
                destination->key = key;
                destination->hash = hash;
                set->used++;
                if (free_slot == NULL) {
                    set->fill++;
                    if (set->fill * 5 >= set->mask * 3) {
                        size_t target = set->used > 50000
                            ? set->used * 2 : set->used * 4;
                        if (pylite_set_table_resize(set, target) < 0) {
                            return -1;
                        }
                    }
                }
                return 1;
            }
            if (entry->key == dummy_key()) {
                /* CPython remembers the latest dummy encountered. */
                free_slot = entry;
            }
            else if (entry->hash == hash) {
                set->stats.comparisons++;
                if (entry->key == key || pylite_equal(entry->key, key)) {
                    return 0;
                }
            }
        }
        perturb >>= PYLITE_PERTURB_SHIFT;
        index = (index * 5 + 1 + perturb) & mask;
    }
}

bool
pylite_set_contains(PyLiteSet *set, const PyLiteObject *key)
{
    if (set == NULL || key == NULL) {
        return false;
    }
    PyLiteHash hash = pylite_hash(key);
    if (hash == PYLITE_HASH_ERROR) {
        return false;
    }
    PyLiteSetEntry *entry = pylite_set_lookup(set, key, hash);
    return entry != NULL && entry_is_active(entry);
}

int
pylite_set_discard(PyLiteSet *set, const PyLiteObject *key)
{
    if (set == NULL || key == NULL) {
        pylite_error_set("invalid set discard");
        return -1;
    }
    PyLiteHash hash = pylite_hash(key);
    if (hash == PYLITE_HASH_ERROR) {
        return -1;
    }
    PyLiteSetEntry *entry = pylite_set_lookup(set, key, hash);
    if (entry == NULL || !entry_is_active(entry)) {
        return 0;
    }
    PyLiteObject *old_key = entry->key;
    entry->key = dummy_key();
    entry->hash = -1;
    set->used--;
    pylite_decref(old_key);
    return 1;
}

int
pylite_set_remove(PyLiteSet *set, const PyLiteObject *key)
{
    int result = pylite_set_discard(set, key);
    if (result == 0) {
        pylite_error_set("set element not found");
        return -1;
    }
    return result < 0 ? -1 : 0;
}

PyLiteObject *
pylite_set_pop(PyLiteSet *set)
{
    if (set == NULL || set->used == 0) {
        pylite_error_set("pop from an empty set");
        return NULL;
    }
    size_t position = set->finger & set->mask;
    while (!entry_is_active(&set->table[position])) {
        position = (position + 1) & set->mask;
    }
    PyLiteObject *key = set->table[position].key; /* transfer set reference */
    set->table[position].key = dummy_key();
    set->table[position].hash = -1;
    set->used--;
    set->finger = position + 1;
    return key;
}

void
pylite_set_clear(PyLiteSet *set)
{
    if (set == NULL) {
        return;
    }
    PyLiteSetEntry *old_table = set->table;
    size_t old_mask = set->mask;
    bool old_was_allocated = old_table != set->smalltable;

    if (old_table == set->smalltable) {
        PyLiteSetEntry copy[PYLITE_SET_MINSIZE];
        memcpy(copy, old_table, sizeof(copy));
        memset(set->smalltable, 0, sizeof(set->smalltable));
        set->table = set->smalltable;
        set->mask = PYLITE_SET_MINSIZE - 1;
        set->fill = 0;
        set->used = 0;
        set->finger = 0;
        for (size_t i = 0; i <= old_mask; i++) {
            if (entry_is_active(&copy[i])) {
                pylite_decref(copy[i].key);
            }
        }
        return;
    }

    memset(set->smalltable, 0, sizeof(set->smalltable));
    set->table = set->smalltable;
    set->mask = PYLITE_SET_MINSIZE - 1;
    set->fill = 0;
    set->used = 0;
    set->finger = 0;
    for (size_t i = 0; i <= old_mask; i++) {
        if (entry_is_active(&old_table[i])) {
            pylite_decref(old_table[i].key);
        }
    }
    if (old_was_allocated) {
        free(old_table);
    }
}

void
pylite_set_free(PyLiteSet *set)
{
    if (set != NULL) {
        pylite_set_clear(set);
        free(set);
    }
}

size_t
pylite_set_len(const PyLiteSet *set)
{
    return set == NULL ? 0 : set->used;
}

int
pylite_set_update(PyLiteSet *set, const PyLiteSet *other)
{
    if (set == NULL || other == NULL) {
        pylite_error_set("invalid set update");
        return -1;
    }
    if (set == other) {
        return 0;
    }
    /* CPython's set_merge_lock_held does one large resize up front. */
    if ((set->fill + other->used) * 5 >= set->mask * 3) {
        if (pylite_set_table_resize(
                set, (set->used + other->used) * 2) < 0) {
            return -1;
        }
    }
    if (set->fill == 0 && set->mask == other->mask &&
        other->fill == other->used) {
        for (size_t i = 0; i <= other->mask; i++) {
            if (entry_is_active(&other->table[i])) {
                pylite_incref(other->table[i].key);
                set->table[i] = other->table[i];
                set->stats.moved_slots++;
            }
        }
        set->fill = other->fill;
        set->used = other->used;
        return 0;
    }
    if (set->fill == 0) {
        set->fill = other->used;
        set->used = other->used;
        for (size_t i = 0; i <= other->mask; i++) {
            if (entry_is_active(&other->table[i])) {
                pylite_incref(other->table[i].key);
                set_insert_clean(set->table, set->mask, other->table[i].key,
                                 other->table[i].hash);
                set->stats.moved_slots++;
            }
        }
        return 0;
    }
    for (size_t i = 0; i <= other->mask; i++) {
        if (entry_is_active(&other->table[i]) &&
            pylite_set_add(set, other->table[i].key) < 0) {
            return -1;
        }
    }
    return 0;
}

PyLiteSet *
pylite_set_copy(const PyLiteSet *set)
{
    if (set == NULL) {
        pylite_error_set("set is NULL");
        return NULL;
    }
    PyLiteSet *copy = pylite_set_new();
    if (copy == NULL || pylite_set_update(copy, set) < 0) {
        pylite_set_free(copy);
        return NULL;
    }
    return copy;
}

PyLiteSet *
pylite_set_union(const PyLiteSet *left, const PyLiteSet *right)
{
    if (left == NULL || right == NULL) {
        pylite_error_set("invalid set union");
        return NULL;
    }
    PyLiteSet *result = pylite_set_copy(left);
    if (result == NULL || pylite_set_update(result, right) < 0) {
        pylite_set_free(result);
        return NULL;
    }
    return result;
}

PyLiteSet *
pylite_set_intersection(PyLiteSet *left, PyLiteSet *right)
{
    if (left == NULL || right == NULL) {
        pylite_error_set("invalid set intersection");
        return NULL;
    }
    if (left->used > right->used) {
        PyLiteSet *temporary = left;
        left = right;
        right = temporary;
    }
    PyLiteSet *result = pylite_set_new();
    if (result == NULL) {
        return NULL;
    }
    for (size_t i = 0; i <= left->mask; i++) {
        PyLiteSetEntry *entry = &left->table[i];
        if (entry_is_active(entry) && pylite_set_contains(right, entry->key) &&
            pylite_set_add(result, entry->key) < 0) {
            pylite_set_free(result);
            return NULL;
        }
    }
    return result;
}

PyLiteSet *
pylite_set_difference(PyLiteSet *left, PyLiteSet *right)
{
    if (left == NULL || right == NULL) {
        pylite_error_set("invalid set difference");
        return NULL;
    }
    PyLiteSet *result = pylite_set_new();
    if (result == NULL) {
        return NULL;
    }
    for (size_t i = 0; i <= left->mask; i++) {
        PyLiteSetEntry *entry = &left->table[i];
        if (entry_is_active(entry) && !pylite_set_contains(right, entry->key) &&
            pylite_set_add(result, entry->key) < 0) {
            pylite_set_free(result);
            return NULL;
        }
    }
    return result;
}

PyLiteSet *
pylite_set_symmetric_difference(PyLiteSet *left, PyLiteSet *right)
{
    PyLiteSet *result = pylite_set_difference(left, right);
    if (result == NULL) {
        return NULL;
    }
    for (size_t i = 0; i <= right->mask; i++) {
        PyLiteSetEntry *entry = &right->table[i];
        if (entry_is_active(entry) && !pylite_set_contains(left, entry->key) &&
            pylite_set_add(result, entry->key) < 0) {
            pylite_set_free(result);
            return NULL;
        }
    }
    return result;
}

int
pylite_set_intersection_update(PyLiteSet *set, PyLiteSet *other)
{
    if (set == NULL || other == NULL) {
        pylite_error_set("invalid set intersection_update");
        return -1;
    }
    if (set == other) {
        return 0;
    }
    for (size_t i = 0; i <= set->mask; i++) {
        PyLiteSetEntry *entry = &set->table[i];
        if (entry_is_active(entry) &&
            !pylite_set_contains(other, entry->key)) {
            PyLiteObject *old_key = entry->key;
            entry->key = dummy_key();
            entry->hash = -1;
            set->used--;
            pylite_decref(old_key);
        }
    }
    return 0;
}

int
pylite_set_difference_update(PyLiteSet *set, PyLiteSet *other)
{
    if (set == NULL || other == NULL) {
        pylite_error_set("invalid set difference_update");
        return -1;
    }
    if (set == other) {
        pylite_set_clear(set);
        return 0;
    }
    for (size_t i = 0; i <= set->mask; i++) {
        PyLiteSetEntry *entry = &set->table[i];
        if (entry_is_active(entry) &&
            pylite_set_contains(other, entry->key)) {
            PyLiteObject *old_key = entry->key;
            entry->key = dummy_key();
            entry->hash = -1;
            set->used--;
            pylite_decref(old_key);
        }
    }
    return 0;
}

int
pylite_set_symmetric_difference_update(PyLiteSet *set, PyLiteSet *other)
{
    if (set == NULL || other == NULL) {
        pylite_error_set("invalid set symmetric_difference_update");
        return -1;
    }
    if (set == other) {
        pylite_set_clear(set);
        return 0;
    }
    for (size_t i = 0; i <= other->mask; i++) {
        PyLiteSetEntry *entry = &other->table[i];
        if (!entry_is_active(entry)) {
            continue;
        }
        int discarded = pylite_set_discard(set, entry->key);
        if (discarded < 0 ||
            (discarded == 0 && pylite_set_add(set, entry->key) < 0)) {
            return -1;
        }
    }
    return 0;
}

bool
pylite_set_is_subset(PyLiteSet *left, PyLiteSet *right)
{
    if (left == NULL || right == NULL || left->used > right->used) {
        return false;
    }
    for (size_t i = 0; i <= left->mask; i++) {
        if (entry_is_active(&left->table[i]) &&
            !pylite_set_contains(right, left->table[i].key)) {
            return false;
        }
    }
    return true;
}

bool
pylite_set_is_superset(PyLiteSet *left, PyLiteSet *right)
{
    return pylite_set_is_subset(right, left);
}

bool
pylite_set_is_disjoint(PyLiteSet *left, PyLiteSet *right)
{
    if (left == NULL || right == NULL) {
        return false;
    }
    if (left->used > right->used) {
        PyLiteSet *temporary = left;
        left = right;
        right = temporary;
    }
    for (size_t i = 0; i <= left->mask; i++) {
        if (entry_is_active(&left->table[i]) &&
            pylite_set_contains(right, left->table[i].key)) {
            return false;
        }
    }
    return true;
}

int
pylite_set_visit(const PyLiteSet *set, PyLiteVisitFn visit, void *context)
{
    if (set == NULL || visit == NULL) {
        pylite_error_set("invalid set visitor");
        return -1;
    }
    for (size_t i = 0; i <= set->mask; i++) {
        if (entry_is_active(&set->table[i])) {
            int result = visit(set->table[i].key, context);
            if (result != 0) {
                return result;
            }
        }
    }
    return 0;
}

const PyLiteStats *
pylite_set_stats(const PyLiteSet *set)
{
    return set == NULL ? NULL : &set->stats;
}

void
pylite_set_reset_stats(PyLiteSet *set)
{
    if (set != NULL) {
        memset(&set->stats, 0, sizeof(set->stats));
    }
}

void
pylite_set_print(const PyLiteSet *set, FILE *stream)
{
    if (stream == NULL) {
        stream = stdout;
    }
    if (set == NULL) {
        fputs("<null set>", stream);
        return;
    }
    if (set->used == 0) {
        fputs("set()", stream);
        return;
    }
    fputc('{', stream);
    bool first = true;
    for (size_t i = 0; i <= set->mask; i++) {
        if (entry_is_active(&set->table[i])) {
            if (!first) {
                fputs(", ", stream);
            }
            pylite_object_print(set->table[i].key, stream);
            first = false;
        }
    }
    fputc('}', stream);
}
