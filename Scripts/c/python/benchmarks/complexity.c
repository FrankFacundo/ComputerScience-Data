#include "pylite.h"

#include <stdio.h>
#include <time.h>

static uint64_t
nanoseconds(void)
{
    struct timespec value;
    (void)clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
        (uint64_t)value.tv_nsec;
}

static int
populate(size_t size, PyLiteList **list_out, PyLiteSet **set_out,
         PyLiteDict **dict_out)
{
    PyLiteList *list = pylite_list_new();
    PyLiteSet *set = pylite_set_new();
    PyLiteDict *dict = pylite_dict_new();
    if (list == NULL || set == NULL || dict == NULL) {
        return -1;
    }
    for (size_t i = 0; i < size; i++) {
        PyLiteObject *value = pylite_int_new((int64_t)i);
        if (value == NULL || pylite_list_append(list, value) < 0 ||
            pylite_set_add(set, value) < 0 ||
            pylite_dict_set(dict, value, value) < 0) {
            pylite_decref(value);
            return -1;
        }
        pylite_decref(value);
    }
    *list_out = list;
    *set_out = set;
    *dict_out = dict;
    return 0;
}

static double
time_list_misses(PyLiteList *list, PyLiteObject **missing, size_t sample_count,
                 size_t repetitions)
{
    uint64_t start = nanoseconds();
    for (size_t i = 0; i < repetitions; i++) {
        (void)pylite_list_contains(list, missing[i % sample_count]);
    }
    return (double)(nanoseconds() - start) / (double)repetitions;
}

static double
time_set_misses(PyLiteSet *set, PyLiteObject **missing, size_t sample_count,
                size_t repetitions)
{
    uint64_t start = nanoseconds();
    for (size_t i = 0; i < repetitions; i++) {
        (void)pylite_set_contains(set, missing[i % sample_count]);
    }
    return (double)(nanoseconds() - start) / (double)repetitions;
}

static double
time_dict_misses(PyLiteDict *dict, PyLiteObject **missing, size_t sample_count,
                 size_t repetitions)
{
    uint64_t start = nanoseconds();
    for (size_t i = 0; i < repetitions; i++) {
        (void)pylite_dict_contains(dict, missing[i % sample_count]);
    }
    return (double)(nanoseconds() - start) / (double)repetitions;
}

static uint64_t
mix64(uint64_t value)
{
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static int
average_case_row(size_t size)
{
    PyLiteList *list = NULL;
    PyLiteSet *set = NULL;
    PyLiteDict *dict = NULL;
    if (populate(size, &list, &set, &dict) < 0) {
        return -1;
    }
    enum { SAMPLE_COUNT = 128 };
    PyLiteObject *missing[SAMPLE_COUNT];
    for (size_t i = 0; i < SAMPLE_COUNT; i++) {
        int64_t candidate = (int64_t)(mix64((uint64_t)size + i) & INT64_MAX);
        if ((uint64_t)candidate < size) {
            candidate += (int64_t)size + 1;
        }
        missing[i] = pylite_int_new(candidate);
        if (missing[i] == NULL) {
            return -1;
        }
    }
    pylite_list_reset_stats(list);
    pylite_set_reset_stats(set);
    pylite_dict_reset_stats(dict);

    size_t repetitions = size < 1000 ? 10000 : (size < 10000 ? 2000 : 200);
    double list_ns = time_list_misses(
        list, missing, SAMPLE_COUNT, repetitions);
    double set_ns = time_set_misses(
        set, missing, SAMPLE_COUNT, repetitions);
    double dict_ns = time_dict_misses(
        dict, missing, SAMPLE_COUNT, repetitions);
    const PyLiteStats *list_stats = pylite_list_stats(list);
    const PyLiteStats *set_stats = pylite_set_stats(set);
    const PyLiteStats *dict_stats = pylite_dict_stats(dict);
    printf("%8zu %12.2f %12.2f %12.2f %12.1f %12.1f %12.1f\n",
           size,
           (double)list_stats->probes / (double)list_stats->lookups,
           (double)set_stats->probes / (double)set_stats->lookups,
           (double)dict_stats->probes / (double)dict_stats->lookups,
           list_ns, set_ns, dict_ns);
    for (size_t i = 0; i < SAMPLE_COUNT; i++) {
        pylite_decref(missing[i]);
    }
    pylite_list_free(list);
    pylite_set_free(set);
    pylite_dict_free(dict);
    return 0;
}

static int
collision_row(size_t size)
{
    PyLiteSet *set = pylite_set_new();
    PyLiteDict *dict = pylite_dict_new();
    if (set == NULL || dict == NULL) {
        return -1;
    }
    for (size_t i = 0; i < size; i++) {
        PyLiteObject *key = pylite_collider_new((int64_t)i);
        if (key == NULL || pylite_set_add(set, key) < 0 ||
            pylite_dict_set(dict, key, key) < 0) {
            pylite_decref(key);
            return -1;
        }
        pylite_decref(key);
    }
    PyLiteObject *missing = pylite_collider_new(-1);
    if (missing == NULL) {
        return -1;
    }
    pylite_set_reset_stats(set);
    pylite_dict_reset_stats(dict);
    (void)pylite_set_contains(set, missing);
    (void)pylite_dict_contains(dict, missing);
    printf("%8zu %18llu %18llu\n", size,
           (unsigned long long)pylite_set_stats(set)->comparisons,
           (unsigned long long)pylite_dict_stats(dict)->comparisons);
    pylite_decref(missing);
    pylite_set_free(set);
    pylite_dict_free(dict);
    return 0;
}

int
main(void)
{
    const size_t average_sizes[] = {100, 1000, 10000, 100000};
    puts("Average-case missing lookup (release build; lower is better)");
    puts("       n  list probes   set probes  dict probes list ns/op    set ns/op   dict ns/op");
    for (size_t i = 0; i < sizeof(average_sizes) / sizeof(average_sizes[0]); i++) {
        if (average_case_row(average_sizes[i]) < 0) {
            fprintf(stderr, "benchmark failed: %s\n", pylite_last_error());
            return 1;
        }
    }

    const size_t collision_sizes[] = {16, 64, 256, 1024};
    puts("\nAdversarial missing lookup: every distinct key has hash 42");
    puts("       n    set comparisons   dict comparisons");
    for (size_t i = 0; i < sizeof(collision_sizes) / sizeof(collision_sizes[0]); i++) {
        if (collision_row(collision_sizes[i]) < 0) {
            fprintf(stderr, "benchmark failed: %s\n", pylite_last_error());
            return 1;
        }
    }
    return 0;
}
