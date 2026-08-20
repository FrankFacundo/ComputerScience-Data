#include "internal.h"

typedef struct {
    PyLiteList *list;
    size_t next;
    bool preallocated;
    int failed;
} ListBuildContext;

typedef struct {
    PyLiteSet *set;
    int failed;
} SetBuildContext;

typedef struct {
    PyLiteDict *dict;
    PyLiteObject *default_value;
    int failed;
} DictBuildContext;

static int
append_value(PyLiteObject *value, void *opaque)
{
    ListBuildContext *context = opaque;
    if (context->preallocated) {
        if (context->next >= context->list->size) {
            context->failed = 1;
            pylite_error_set("container changed during list conversion");
            return -1;
        }
        pylite_incref(value);
        context->list->items[context->next++] = value;
        return 0;
    }
    if (pylite_list_append(context->list, value) < 0) {
        context->failed = 1;
        return -1;
    }
    return 0;
}

static PyLiteList *
new_preallocated_list(size_t size)
{
    PyLiteList *list = pylite_list_new();
    if (list == NULL) {
        return NULL;
    }
    if (size != 0 && pylite_list_resize(list, size) < 0) {
        pylite_list_free(list);
        return NULL;
    }
    if (size != 0) {
        memset(list->items, 0, size * sizeof(*list->items));
    }
    return list;
}

static int
reserve_dict(PyLiteDict *dict, size_t expected_entries)
{
    if (expected_entries <= dict->usable) {
        return 0;
    }
    if (expected_entries > (SIZE_MAX - 1) / 3) {
        pylite_error_set("dict is too large");
        return -1;
    }
    return pylite_dict_resize(dict, (expected_entries * 3 + 1) / 2);
}

static int
append_key(PyLiteObject *key, PyLiteObject *value, void *opaque)
{
    (void)value;
    return append_value(key, opaque);
}

static int
append_dict_value(PyLiteObject *key, PyLiteObject *value, void *opaque)
{
    (void)key;
    return append_value(value, opaque);
}

static int
append_item(PyLiteObject *key, PyLiteObject *value, void *opaque)
{
    ListBuildContext *context = opaque;
    PyLiteObject *pair = pylite_pair_new(key, value);
    if (pair == NULL) {
        context->failed = 1;
        return -1;
    }
    int result = append_value(pair, context);
    pylite_decref(pair);
    if (result < 0) {
        context->failed = 1;
    }
    return result;
}

PyLiteList *
pylite_list_from_set(const PyLiteSet *set)
{
    if (set == NULL) {
        pylite_error_set("cannot convert NULL set to list");
        return NULL;
    }
    PyLiteList *list = new_preallocated_list(set->used);
    if (list == NULL) {
        return NULL;
    }
    ListBuildContext context = {
        .list = list, .next = 0, .preallocated = true, .failed = 0
    };
    if (pylite_set_visit(set, append_value, &context) < 0 || context.failed) {
        pylite_list_free(list);
        return NULL;
    }
    return list;
}

PyLiteList *
pylite_list_from_dict(const PyLiteDict *dict)
{
    if (dict == NULL) {
        pylite_error_set("cannot convert NULL dict to list");
        return NULL;
    }
    PyLiteList *list = new_preallocated_list(dict->used);
    if (list == NULL) {
        return NULL;
    }
    ListBuildContext context = {
        .list = list, .next = 0, .preallocated = true, .failed = 0
    };
    if (pylite_dict_visit(dict, append_key, &context) < 0 || context.failed) {
        pylite_list_free(list);
        return NULL;
    }
    return list;
}

PyLiteList *
pylite_list_from_dict_items(const PyLiteDict *dict)
{
    if (dict == NULL) {
        pylite_error_set("cannot convert NULL dict items to list");
        return NULL;
    }
    PyLiteList *list = new_preallocated_list(dict->used);
    if (list == NULL) {
        return NULL;
    }
    ListBuildContext context = {
        .list = list, .next = 0, .preallocated = true, .failed = 0
    };
    if (pylite_dict_visit(dict, append_item, &context) < 0 || context.failed) {
        pylite_list_free(list);
        return NULL;
    }
    return list;
}

PyLiteList *
pylite_list_from_dict_values(const PyLiteDict *dict)
{
    if (dict == NULL) {
        pylite_error_set("cannot convert NULL dict values to list");
        return NULL;
    }
    PyLiteList *list = new_preallocated_list(dict->used);
    if (list == NULL) {
        return NULL;
    }
    ListBuildContext context = {
        .list = list, .next = 0, .preallocated = true, .failed = 0
    };
    if (pylite_dict_visit(dict, append_dict_value, &context) < 0 ||
        context.failed) {
        pylite_list_free(list);
        return NULL;
    }
    return list;
}

static int
add_value(PyLiteObject *value, void *opaque)
{
    SetBuildContext *context = opaque;
    if (pylite_set_add(context->set, value) < 0) {
        context->failed = 1;
        return -1;
    }
    return 0;
}

static int
add_key(PyLiteObject *key, PyLiteObject *value, void *opaque)
{
    (void)value;
    return add_value(key, opaque);
}

PyLiteSet *
pylite_set_from_list(const PyLiteList *list)
{
    if (list == NULL) {
        pylite_error_set("cannot convert NULL list to set");
        return NULL;
    }
    PyLiteSet *set = pylite_set_new();
    if (set == NULL) {
        return NULL;
    }
    SetBuildContext context = {.set = set, .failed = 0};
    if (pylite_list_visit(list, add_value, &context) < 0 || context.failed) {
        pylite_set_free(set);
        return NULL;
    }
    return set;
}

PyLiteSet *
pylite_set_from_dict(const PyLiteDict *dict)
{
    if (dict == NULL) {
        pylite_error_set("cannot convert NULL dict to set");
        return NULL;
    }
    PyLiteSet *set = pylite_set_new();
    if (set == NULL) {
        return NULL;
    }
    if (dict->used > SIZE_MAX / 2 ||
        (dict->used != 0 &&
         pylite_set_table_resize(set, dict->used * 2) < 0)) {
        pylite_set_free(set);
        return NULL;
    }
    SetBuildContext context = {.set = set, .failed = 0};
    if (pylite_dict_visit(dict, add_key, &context) < 0 || context.failed) {
        pylite_set_free(set);
        return NULL;
    }
    return set;
}

PyLiteDict *
pylite_dict_from_pairs(const PyLiteList *pairs)
{
    if (pairs == NULL) {
        pylite_error_set("cannot convert NULL pairs list to dict");
        return NULL;
    }
    PyLiteDict *dict = pylite_dict_new();
    if (dict == NULL) {
        return NULL;
    }
    if (reserve_dict(dict, pairs->size) < 0) {
        pylite_dict_free(dict);
        return NULL;
    }
    for (size_t i = 0; i < pairs->size; i++) {
        PyLiteObject *pair = pairs->items[i];
        if (pylite_type(pair) != PYLITE_PAIR) {
            pylite_error_set("dict conversion element %zu is not a pair", i);
            pylite_dict_free(dict);
            return NULL;
        }
        if (pylite_dict_set(dict, pylite_pair_first(pair),
                            pylite_pair_second(pair)) < 0) {
            pylite_dict_free(dict);
            return NULL;
        }
    }
    return dict;
}

static int
set_default(PyLiteObject *key, void *opaque)
{
    DictBuildContext *context = opaque;
    if (pylite_dict_set(context->dict, key, context->default_value) < 0) {
        context->failed = 1;
        return -1;
    }
    return 0;
}

PyLiteDict *
pylite_dict_fromkeys_list(const PyLiteList *keys,
                          PyLiteObject *default_value)
{
    if (keys == NULL) {
        pylite_error_set("cannot convert NULL list to dict");
        return NULL;
    }
    PyLiteDict *dict = pylite_dict_new();
    if (dict == NULL) {
        return NULL;
    }
    if (reserve_dict(dict, keys->size) < 0) {
        pylite_dict_free(dict);
        return NULL;
    }
    DictBuildContext context = {
        .dict = dict,
        .default_value = default_value != NULL ? default_value : pylite_none(),
        .failed = 0
    };
    if (pylite_list_visit(keys, set_default, &context) < 0 || context.failed) {
        pylite_dict_free(dict);
        return NULL;
    }
    return dict;
}

PyLiteDict *
pylite_dict_fromkeys_set(const PyLiteSet *keys,
                         PyLiteObject *default_value)
{
    if (keys == NULL) {
        pylite_error_set("cannot convert NULL set to dict");
        return NULL;
    }
    PyLiteDict *dict = pylite_dict_new();
    if (dict == NULL) {
        return NULL;
    }
    if (reserve_dict(dict, keys->used) < 0) {
        pylite_dict_free(dict);
        return NULL;
    }
    DictBuildContext context = {
        .dict = dict,
        .default_value = default_value != NULL ? default_value : pylite_none(),
        .failed = 0
    };
    if (pylite_set_visit(keys, set_default, &context) < 0 || context.failed) {
        pylite_dict_free(dict);
        return NULL;
    }
    return dict;
}
