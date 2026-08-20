#include "internal.h"

#include <assert.h>
#include <stdint.h>

static bool
normalize_index(const PyLiteList *list, ptrdiff_t index, size_t *normalized)
{
    if (list == NULL) {
        pylite_error_set("list is NULL");
        return false;
    }
    ptrdiff_t size = (ptrdiff_t)list->size;
    if (index < 0) {
        index += size;
    }
    if (index < 0 || index >= size) {
        pylite_error_set("list index out of range");
        return false;
    }
    *normalized = (size_t)index;
    return true;
}

PyLiteList *
pylite_list_new(void)
{
    PyLiteList *list = calloc(1, sizeof(*list));
    if (list == NULL) {
        pylite_error_set("out of memory allocating list");
    }
    return list;
}

/*
 * Adapted from CPython 3.14.7 list_resize:
 *   new_allocated = (newsize + newsize/8 + 6) rounded down to a multiple of 4
 * and do not shrink while at least half the allocation remains in use.
 */
int
pylite_list_resize(PyLiteList *list, size_t new_size)
{
    if (list == NULL) {
        pylite_error_set("list is NULL");
        return -1;
    }
    size_t allocated = list->allocated;
    if (allocated >= new_size && new_size >= (allocated >> 1)) {
        list->size = new_size;
        return 0;
    }
    if (new_size > SIZE_MAX - (new_size >> 3) - 6) {
        pylite_error_set("list is too large");
        return -1;
    }

    size_t new_allocated = (new_size + (new_size >> 3) + 6) & ~(size_t)3;
    if (new_size > list->size &&
        new_size - list->size > new_allocated - new_size) {
        new_allocated = (new_size + 3) & ~(size_t)3;
    }
    if (new_size == 0) {
        new_allocated = 0;
    }

    size_t bytes;
    if (!pylite_size_multiply(new_allocated, sizeof(*list->items), &bytes)) {
        return -1;
    }
    pylite_trace("list_resize size=%zu->%zu capacity=%zu->%zu",
                 list->size, new_size, allocated, new_allocated);
    PyLiteObject **new_items = NULL;
    if (new_allocated == 0) {
        free(list->items);
    }
    else {
        new_items = realloc(list->items, bytes);
        if (new_items == NULL) {
            pylite_error_set("out of memory resizing list to %zu slots",
                             new_allocated);
            return -1;
        }
    }
    list->items = new_items;
    list->allocated = new_allocated;
    list->size = new_size;
    list->stats.resizes++;
    list->stats.moved_slots += allocated < new_allocated ? allocated : new_allocated;
    return 0;
}

void
pylite_list_clear(PyLiteList *list)
{
    if (list == NULL) {
        return;
    }
    /* Clear the visible list first, mirroring CPython's re-entrancy-safe order. */
    PyLiteObject **old_items = list->items;
    size_t old_size = list->size;
    list->items = NULL;
    list->size = 0;
    list->allocated = 0;
    for (size_t i = old_size; i > 0; i--) {
        pylite_decref(old_items[i - 1]);
    }
    free(old_items);
}

void
pylite_list_free(PyLiteList *list)
{
    if (list != NULL) {
        pylite_list_clear(list);
        free(list);
    }
}

size_t
pylite_list_len(const PyLiteList *list)
{
    return list == NULL ? 0 : list->size;
}

size_t
pylite_list_capacity(const PyLiteList *list)
{
    return list == NULL ? 0 : list->allocated;
}

PyLiteObject *
pylite_list_get(const PyLiteList *list, ptrdiff_t index)
{
    size_t normalized;
    return normalize_index(list, index, &normalized) ? list->items[normalized] : NULL;
}

int
pylite_list_set(PyLiteList *list, ptrdiff_t index, PyLiteObject *value)
{
    size_t normalized;
    if (value == NULL || !normalize_index(list, index, &normalized)) {
        if (value == NULL) {
            pylite_error_set("list value cannot be NULL");
        }
        return -1;
    }
    pylite_incref(value);
    PyLiteObject *old_value = list->items[normalized];
    list->items[normalized] = value;
    pylite_decref(old_value);
    return 0;
}

int
pylite_list_append(PyLiteList *list, PyLiteObject *value)
{
    if (list == NULL || value == NULL || list->size == SIZE_MAX) {
        pylite_error_set("invalid list append");
        return -1;
    }
    size_t old_size = list->size;
    if (pylite_list_resize(list, old_size + 1) < 0) {
        return -1;
    }
    assert(list->items != NULL);
    pylite_incref(value);
    list->items[old_size] = value;
    return 0;
}

int
pylite_list_extend(PyLiteList *list, const PyLiteList *other)
{
    if (list == NULL || other == NULL) {
        pylite_error_set("cannot extend a NULL list");
        return -1;
    }
    size_t old_size = list->size;
    size_t add_size = other->size;
    if (add_size > SIZE_MAX - old_size) {
        pylite_error_set("list is too large");
        return -1;
    }
    if (pylite_list_resize(list, old_size + add_size) < 0) {
        return -1;
    }
    assert(add_size == 0 || list->items != NULL);
    /* add_size was captured before resizing, so self-extension is safe. */
    for (size_t i = 0; i < add_size; i++) {
        PyLiteObject *value = other->items[i];
        pylite_incref(value);
        list->items[old_size + i] = value;
    }
    return 0;
}

int
pylite_list_insert(PyLiteList *list, ptrdiff_t index, PyLiteObject *value)
{
    if (list == NULL || value == NULL || list->size == SIZE_MAX) {
        pylite_error_set("invalid list insert");
        return -1;
    }
    ptrdiff_t size = (ptrdiff_t)list->size;
    if (index < 0) {
        index += size;
        if (index < 0) {
            index = 0;
        }
    }
    if (index > size) {
        index = size;
    }
    size_t position = (size_t)index;
    size_t old_size = list->size;
    if (pylite_list_resize(list, old_size + 1) < 0) {
        return -1;
    }
    assert(list->items != NULL);
    if (old_size > position) {
        memmove(&list->items[position + 1], &list->items[position],
                (old_size - position) * sizeof(*list->items));
    }
    list->stats.moved_slots += old_size - position;
    pylite_incref(value);
    list->items[position] = value;
    return 0;
}

PyLiteObject *
pylite_list_pop(PyLiteList *list, ptrdiff_t index)
{
    size_t position;
    if (!normalize_index(list, index, &position)) {
        return NULL;
    }
    PyLiteObject *result = list->items[position]; /* transfer list's reference */
    size_t old_size = list->size;
    memmove(&list->items[position], &list->items[position + 1],
            (old_size - position - 1) * sizeof(*list->items));
    list->stats.moved_slots += old_size - position - 1;
    list->items[old_size - 1] = NULL;
    if (pylite_list_resize(list, old_size - 1) < 0) {
        /* A failed shrink must not make a successful pop unusable. */
        list->size = old_size - 1;
    }
    return result;
}

static void
normalize_slice(const PyLiteList *list, ptrdiff_t *start, ptrdiff_t *stop)
{
    ptrdiff_t size = (ptrdiff_t)list->size;
    if (*start < 0) {
        *start += size;
        if (*start < 0) {
            *start = 0;
        }
    }
    else if (*start > size) {
        *start = size;
    }
    if (*stop < 0) {
        *stop += size;
        if (*stop < 0) {
            *stop = 0;
        }
    }
    else if (*stop > size) {
        *stop = size;
    }
}

ptrdiff_t
pylite_list_index(PyLiteList *list, const PyLiteObject *value,
                  ptrdiff_t start, ptrdiff_t stop)
{
    if (list == NULL || value == NULL) {
        pylite_error_set("invalid list index search");
        return -1;
    }
    normalize_slice(list, &start, &stop);
    list->stats.lookups++;
    for (ptrdiff_t i = start; i < stop; i++) {
        list->stats.probes++;
        list->stats.comparisons++;
        if (pylite_equal(list->items[i], value)) {
            return i;
        }
    }
    pylite_error_set("value is not in list");
    return -1;
}

size_t
pylite_list_count(PyLiteList *list, const PyLiteObject *value)
{
    if (list == NULL || value == NULL) {
        return 0;
    }
    size_t count = 0;
    list->stats.lookups++;
    for (size_t i = 0; i < list->size; i++) {
        list->stats.probes++;
        list->stats.comparisons++;
        count += pylite_equal(list->items[i], value);
    }
    return count;
}

bool
pylite_list_contains(PyLiteList *list, const PyLiteObject *value)
{
    if (list == NULL || value == NULL) {
        return false;
    }
    list->stats.lookups++;
    for (size_t i = 0; i < list->size; i++) {
        list->stats.probes++;
        list->stats.comparisons++;
        if (pylite_equal(list->items[i], value)) {
            return true;
        }
    }
    return false;
}

int
pylite_list_remove(PyLiteList *list, const PyLiteObject *value)
{
    if (list == NULL || value == NULL) {
        pylite_error_set("invalid list remove");
        return -1;
    }
    list->stats.lookups++;
    for (size_t i = 0; i < list->size; i++) {
        list->stats.probes++;
        list->stats.comparisons++;
        if (pylite_equal(list->items[i], value)) {
            PyLiteObject *removed = pylite_list_pop(list, (ptrdiff_t)i);
            pylite_decref(removed);
            return 0;
        }
    }
    pylite_error_set("value is not in list");
    return -1;
}

void
pylite_list_reverse(PyLiteList *list)
{
    if (list == NULL) {
        return;
    }
    for (size_t left = 0, right = list->size; left < right && left < --right;
         left++) {
        PyLiteObject *temporary = list->items[left];
        list->items[left] = list->items[right];
        list->items[right] = temporary;
    }
}

static int
merge_ranges(PyLiteList *list, PyLiteObject **temporary, size_t begin,
             size_t middle, size_t end, bool reverse)
{
    size_t left = begin;
    size_t right = middle;
    size_t output = begin;
    while (left < middle && right < end) {
        int comparison;
        list->stats.comparisons++;
        if (pylite_compare(list->items[left], list->items[right], &comparison) < 0) {
            return -1;
        }
        bool take_left = reverse ? comparison >= 0 : comparison <= 0;
        temporary[output++] = take_left ? list->items[left++] : list->items[right++];
    }
    while (left < middle) {
        temporary[output++] = list->items[left++];
    }
    while (right < end) {
        temporary[output++] = list->items[right++];
    }
    memcpy(&list->items[begin], &temporary[begin],
           (end - begin) * sizeof(*temporary));
    list->stats.moved_slots += end - begin;
    return 0;
}

int
pylite_list_sort(PyLiteList *list, bool reverse)
{
    if (list == NULL) {
        pylite_error_set("list is NULL");
        return -1;
    }
    if (list->size < 2) {
        return 0;
    }
    size_t bytes;
    if (!pylite_size_multiply(list->size, sizeof(*list->items), &bytes)) {
        return -1;
    }
    PyLiteObject **temporary = malloc(bytes);
    if (temporary == NULL) {
        pylite_error_set("out of memory sorting list");
        return -1;
    }
    for (size_t width = 1; width < list->size;) {
        for (size_t begin = 0; begin < list->size; begin += width * 2) {
            size_t middle = begin + width < list->size ? begin + width : list->size;
            size_t end = middle + width < list->size ? middle + width : list->size;
            if (merge_ranges(list, temporary, begin, middle, end, reverse) < 0) {
                free(temporary);
                return -1;
            }
        }
        if (width > list->size / 2) {
            break;
        }
        width *= 2;
    }
    free(temporary);
    return 0;
}

PyLiteList *
pylite_list_copy(const PyLiteList *list)
{
    if (list == NULL) {
        pylite_error_set("list is NULL");
        return NULL;
    }
    PyLiteList *copy = pylite_list_new();
    if (copy == NULL || pylite_list_extend(copy, list) < 0) {
        pylite_list_free(copy);
        return NULL;
    }
    return copy;
}

int
pylite_list_visit(const PyLiteList *list, PyLiteVisitFn visit, void *context)
{
    if (list == NULL || visit == NULL) {
        pylite_error_set("invalid list visitor");
        return -1;
    }
    for (size_t i = 0; i < list->size; i++) {
        int result = visit(list->items[i], context);
        if (result != 0) {
            return result;
        }
    }
    return 0;
}

const PyLiteStats *
pylite_list_stats(const PyLiteList *list)
{
    return list == NULL ? NULL : &list->stats;
}

void
pylite_list_reset_stats(PyLiteList *list)
{
    if (list != NULL) {
        memset(&list->stats, 0, sizeof(list->stats));
    }
}

void
pylite_list_print(const PyLiteList *list, FILE *stream)
{
    if (stream == NULL) {
        stream = stdout;
    }
    if (list == NULL) {
        fputs("<null list>", stream);
        return;
    }
    fputc('[', stream);
    for (size_t i = 0; i < list->size; i++) {
        if (i != 0) {
            fputs(", ", stream);
        }
        pylite_object_print(list->items[i], stream);
    }
    fputc(']', stream);
}
