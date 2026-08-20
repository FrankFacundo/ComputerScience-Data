#include "pylite.h"

#include <stdio.h>
#include <string.h>

static int
append_int(PyLiteList *list, int64_t value)
{
    PyLiteObject *object = pylite_int_new(value);
    if (object == NULL) {
        return -1;
    }
    int result = pylite_list_append(list, object);
    pylite_decref(object);
    return result;
}

static int
show_list_and_conversions(void)
{
    PyLiteList *list = pylite_list_new();
    if (list == NULL) {
        return -1;
    }
    int64_t values[] = {3, 1, 3, 2, 5, 2};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
        if (append_int(list, values[i]) < 0) {
            pylite_list_free(list);
            return -1;
        }
    }
    printf("list                       = ");
    pylite_list_print(list, stdout);
    printf("  (size=%zu, capacity=%zu)\n",
           pylite_list_len(list), pylite_list_capacity(list));

    PyLiteSet *set = pylite_set_from_list(list);
    PyLiteDict *dict = pylite_dict_fromkeys_list(list, NULL);
    PyLiteList *set_as_list = set == NULL ? NULL : pylite_list_from_set(set);
    if (set == NULL || dict == NULL || set_as_list == NULL) {
        pylite_list_free(set_as_list);
        pylite_dict_free(dict);
        pylite_set_free(set);
        pylite_list_free(list);
        return -1;
    }
    printf("set(list)                  = ");
    pylite_set_print(set, stdout);
    putchar('\n');
    printf("dict.fromkeys(list)        = ");
    pylite_dict_print(dict, stdout);
    putchar('\n');
    printf("list(set(list))            = ");
    pylite_list_print(set_as_list, stdout);
    putchar('\n');

    PyLiteObject *needle = pylite_int_new(5);
    if (needle == NULL) {
        return -1;
    }
    pylite_list_reset_stats(list);
    pylite_set_reset_stats(set);
    pylite_dict_reset_stats(dict);
    (void)pylite_list_contains(list, needle);
    (void)pylite_set_contains(set, needle);
    (void)pylite_dict_contains(dict, needle);
    printf("\nprobes to find 5: list=%llu set=%llu dict=%llu\n",
           (unsigned long long)pylite_list_stats(list)->probes,
           (unsigned long long)pylite_set_stats(set)->probes,
           (unsigned long long)pylite_dict_stats(dict)->probes);
    pylite_decref(needle);

    pylite_list_free(set_as_list);
    pylite_dict_free(dict);
    pylite_set_free(set);
    pylite_list_free(list);
    return 0;
}

static int
show_collision_case(void)
{
    PyLiteSet *set = pylite_set_new();
    PyLiteDict *dict = pylite_dict_new();
    if (set == NULL || dict == NULL) {
        pylite_set_free(set);
        pylite_dict_free(dict);
        return -1;
    }
    for (int64_t i = 0; i < 12; i++) {
        PyLiteObject *key = pylite_collider_new(i);
        if (key == NULL || pylite_set_add(set, key) < 0 ||
            pylite_dict_set(dict, key, key) < 0) {
            pylite_decref(key);
            pylite_set_free(set);
            pylite_dict_free(dict);
            return -1;
        }
        pylite_decref(key);
    }
    PyLiteObject *missing = pylite_collider_new(999);
    if (missing == NULL) {
        return -1;
    }
    pylite_set_reset_stats(set);
    pylite_dict_reset_stats(dict);
    (void)pylite_set_contains(set, missing);
    (void)pylite_dict_contains(dict, missing);
    printf("same-hash miss (12 keys):  set comparisons=%llu, dict comparisons=%llu\n",
           (unsigned long long)pylite_set_stats(set)->comparisons,
           (unsigned long long)pylite_dict_stats(dict)->comparisons);
    pylite_decref(missing);
    pylite_set_free(set);
    pylite_dict_free(dict);
    return 0;
}

int
main(int argc, char **argv)
{
    bool trace = argc > 1 && strcmp(argv[1], "--trace") == 0;
    pylite_trace_set(trace);
    if (show_list_and_conversions() < 0 || show_collision_case() < 0) {
        fprintf(stderr, "error: %s\n", pylite_last_error());
        return 1;
    }
    return 0;
}
