#include "pylite.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static PyLiteObject *
integer(int64_t value)
{
    PyLiteObject *object = pylite_int_new(value);
    assert(object != NULL);
    return object;
}

static void
append_integer(PyLiteList *list, int64_t value)
{
    PyLiteObject *object = integer(value);
    assert(pylite_list_append(list, object) == 0);
    pylite_decref(object);
}

static void
add_integer(PyLiteSet *set, int64_t value)
{
    PyLiteObject *object = integer(value);
    assert(pylite_set_add(set, object) >= 0);
    pylite_decref(object);
}

static void
set_integer(PyLiteDict *dict, int64_t key_value, int64_t value_value)
{
    PyLiteObject *key = integer(key_value);
    PyLiteObject *value = integer(value_value);
    assert(pylite_dict_set(dict, key, value) == 0);
    pylite_decref(key);
    pylite_decref(value);
}

static bool
has_integer_in_set(PyLiteSet *set, int64_t value)
{
    PyLiteObject *key = integer(value);
    bool result = pylite_set_contains(set, key);
    pylite_decref(key);
    return result;
}

static PyLiteObject *
get_integer_key(PyLiteDict *dict, int64_t value)
{
    PyLiteObject *key = integer(value);
    PyLiteObject *result = pylite_dict_get(dict, key);
    pylite_decref(key);
    return result;
}

static void
test_objects(void)
{
    PyLiteObject *one = integer(1);
    PyLiteObject *other_one = integer(1);
    PyLiteObject *minus_one = integer(-1);
    PyLiteObject *hello = pylite_string_new("hello");
    PyLiteObject *other_hello = pylite_string_new("hello");
    assert(hello != NULL && other_hello != NULL);
    assert(pylite_equal(one, other_one));
    assert(pylite_hash(one) == 1);
    assert(pylite_hash(minus_one) == -2);
    assert(pylite_equal(hello, other_hello));
    assert(pylite_hash(hello) == pylite_hash(other_hello));

    PyLiteObject *pair = pylite_pair_new(one, hello);
    PyLiteObject *other_pair = pylite_pair_new(other_one, other_hello);
    assert(pair != NULL && other_pair != NULL);
    assert(pylite_equal(pair, other_pair));
    assert(pylite_hash(pair) == pylite_hash(other_pair));

    pylite_decref(pair);
    pylite_decref(other_pair);
    pylite_decref(one);
    pylite_decref(other_one);
    pylite_decref(minus_one);
    pylite_decref(hello);
    pylite_decref(other_hello);
}

static void
test_list(void)
{
    PyLiteList *list = pylite_list_new();
    assert(list != NULL);
    for (int64_t i = 0; i < 20; i++) {
        append_integer(list, i);
    }
    assert(pylite_list_len(list) == 20);
    assert(pylite_list_capacity(list) == 24);
    assert(pylite_int_value(pylite_list_get(list, -1)) == 19);
    assert(pylite_list_get(list, 20) == NULL);

    PyLiteObject *hundred = integer(100);
    assert(pylite_list_insert(list, -100, hundred) == 0);
    pylite_decref(hundred);
    assert(pylite_int_value(pylite_list_get(list, 0)) == 100);

    PyLiteObject *popped = pylite_list_pop(list, -1);
    assert(popped != NULL && pylite_int_value(popped) == 19);
    pylite_decref(popped);

    PyLiteObject *five = integer(5);
    assert(pylite_list_contains(list, five));
    assert(pylite_list_index(list, five, 0, (ptrdiff_t)pylite_list_len(list)) == 6);
    assert(pylite_list_count(list, five) == 1);
    assert(pylite_list_remove(list, five) == 0);
    assert(!pylite_list_contains(list, five));
    pylite_decref(five);

    PyLiteList *copy = pylite_list_copy(list);
    assert(copy != NULL && pylite_list_len(copy) == pylite_list_len(list));
    assert(pylite_list_extend(copy, copy) == 0);
    assert(pylite_list_len(copy) == 2 * pylite_list_len(list));

    pylite_list_reverse(list);
    assert(pylite_list_sort(list, false) == 0);
    for (size_t i = 1; i < pylite_list_len(list); i++) {
        assert(pylite_int_value(pylite_list_get(list, (ptrdiff_t)i - 1)) <=
               pylite_int_value(pylite_list_get(list, (ptrdiff_t)i)));
    }
    pylite_list_free(copy);
    pylite_list_free(list);
}

static void
test_set(void)
{
    PyLiteSet *left = pylite_set_new();
    PyLiteSet *right = pylite_set_new();
    assert(left != NULL && right != NULL);
    for (int64_t i = 0; i < 100; i++) {
        add_integer(left, i);
    }
    for (int64_t i = 50; i < 150; i++) {
        add_integer(right, i);
    }
    assert(pylite_set_len(left) == 100);
    assert(has_integer_in_set(left, 42));
    assert(!has_integer_in_set(left, 142));

    PyLiteObject *forty_two = integer(42);
    assert(pylite_set_add(left, forty_two) == 0);
    assert(pylite_set_discard(left, forty_two) == 1);
    assert(pylite_set_discard(left, forty_two) == 0);
    pylite_decref(forty_two);

    PyLiteSet *intersection = pylite_set_intersection(left, right);
    PyLiteSet *difference = pylite_set_difference(left, right);
    PyLiteSet *united = pylite_set_union(left, right);
    PyLiteSet *symmetric = pylite_set_symmetric_difference(left, right);
    PyLiteSet *intersection_in_place = pylite_set_copy(left);
    PyLiteSet *difference_in_place = pylite_set_copy(left);
    PyLiteSet *symmetric_in_place = pylite_set_copy(left);
    assert(intersection != NULL && pylite_set_len(intersection) == 50);
    assert(difference != NULL && pylite_set_len(difference) == 49);
    assert(united != NULL && pylite_set_len(united) == 149);
    assert(symmetric != NULL && pylite_set_len(symmetric) == 99);
    assert(intersection_in_place != NULL &&
           pylite_set_intersection_update(intersection_in_place, right) == 0 &&
           pylite_set_len(intersection_in_place) == 50);
    assert(difference_in_place != NULL &&
           pylite_set_difference_update(difference_in_place, right) == 0 &&
           pylite_set_len(difference_in_place) == 49);
    assert(symmetric_in_place != NULL &&
           pylite_set_symmetric_difference_update(
               symmetric_in_place, right) == 0 &&
           pylite_set_len(symmetric_in_place) == 99);
    assert(pylite_set_is_subset(intersection, united));
    assert(pylite_set_is_superset(united, intersection));
    assert(pylite_set_is_disjoint(intersection, difference));

    size_t previous = pylite_set_len(left);
    PyLiteObject *popped = pylite_set_pop(left);
    assert(popped != NULL && pylite_set_len(left) == previous - 1);
    pylite_decref(popped);

    pylite_set_free(intersection);
    pylite_set_free(difference);
    pylite_set_free(united);
    pylite_set_free(symmetric);
    pylite_set_free(intersection_in_place);
    pylite_set_free(difference_in_place);
    pylite_set_free(symmetric_in_place);
    pylite_set_free(left);
    pylite_set_free(right);
}

static void
test_dict(void)
{
    PyLiteDict *dict = pylite_dict_new();
    assert(dict != NULL);
    for (int64_t i = 0; i < 100; i++) {
        set_integer(dict, i, i * 10);
    }
    assert(pylite_dict_len(dict) == 100);
    assert(pylite_int_value(get_integer_key(dict, 37)) == 370);
    assert(get_integer_key(dict, 1000) == NULL);

    set_integer(dict, 37, 999);
    assert(pylite_int_value(get_integer_key(dict, 37)) == 999);
    assert(pylite_dict_len(dict) == 100);

    PyLiteObject *key = integer(37);
    assert(pylite_dict_del(dict, key) == 0);
    assert(!pylite_dict_contains(dict, key));
    PyLiteObject *fallback = integer(-99);
    PyLiteObject *default_result = pylite_dict_get_default(dict, key, fallback);
    assert(default_result == fallback);
    assert(pylite_dict_setdefault(dict, key, fallback) == fallback);
    assert(pylite_dict_contains(dict, key));
    pylite_decref(fallback);
    pylite_decref(key);

    PyLiteObject *last_pair = pylite_dict_popitem(dict);
    assert(last_pair != NULL && pylite_type(last_pair) == PYLITE_PAIR);
    assert(pylite_int_value(pylite_pair_first(last_pair)) == 37);
    pylite_decref(last_pair);

    PyLiteObject *zero = integer(0);
    PyLiteObject *popped = pylite_dict_pop(dict, zero, NULL);
    assert(popped != NULL && pylite_int_value(popped) == 0);
    pylite_decref(popped);
    pylite_decref(zero);

    PyLiteDict *copy = pylite_dict_copy(dict);
    assert(copy != NULL && pylite_dict_len(copy) == pylite_dict_len(dict));
    pylite_dict_clear(copy);
    assert(pylite_dict_len(copy) == 0);
    pylite_dict_free(copy);
    pylite_dict_free(dict);
}

static void
test_conversions(void)
{
    PyLiteList *list = pylite_list_new();
    assert(list != NULL);
    append_integer(list, 3);
    append_integer(list, 1);
    append_integer(list, 3);
    append_integer(list, 2);

    PyLiteSet *set = pylite_set_from_list(list);
    assert(set != NULL && pylite_set_len(set) == 3);
    PyLiteList *roundtrip = pylite_list_from_set(set);
    assert(roundtrip != NULL && pylite_list_len(roundtrip) == 3);

    PyLiteDict *from_list = pylite_dict_fromkeys_list(list, NULL);
    PyLiteDict *from_set = pylite_dict_fromkeys_set(set, NULL);
    assert(from_list != NULL && pylite_dict_len(from_list) == 3);
    assert(from_set != NULL && pylite_dict_len(from_set) == 3);
    PyLiteList *keys = pylite_list_from_dict(from_list);
    PyLiteSet *key_set = pylite_set_from_dict(from_list);
    assert(keys != NULL && pylite_list_len(keys) == 3);
    assert(key_set != NULL && pylite_set_len(key_set) == 3);

    PyLiteObject *a = pylite_string_new("a");
    PyLiteObject *b = pylite_string_new("b");
    PyLiteObject *one = integer(1);
    PyLiteObject *two = integer(2);
    PyLiteObject *pair_a = pylite_pair_new(a, one);
    PyLiteObject *pair_b = pylite_pair_new(b, two);
    PyLiteList *pairs = pylite_list_new();
    assert(a && b && pair_a && pair_b && pairs);
    assert(pylite_list_append(pairs, pair_a) == 0);
    assert(pylite_list_append(pairs, pair_b) == 0);
    PyLiteDict *from_pairs = pylite_dict_from_pairs(pairs);
    assert(from_pairs != NULL && pylite_dict_len(from_pairs) == 2);
    PyLiteList *items = pylite_list_from_dict_items(from_pairs);
    PyLiteList *values = pylite_list_from_dict_values(from_pairs);
    assert(items != NULL && pylite_list_len(items) == 2);
    assert(values != NULL && pylite_list_len(values) == 2);

    pylite_decref(a);
    pylite_decref(b);
    pylite_decref(one);
    pylite_decref(two);
    pylite_decref(pair_a);
    pylite_decref(pair_b);
    pylite_list_free(values);
    pylite_list_free(items);
    pylite_dict_free(from_pairs);
    pylite_list_free(pairs);
    pylite_set_free(key_set);
    pylite_list_free(keys);
    pylite_dict_free(from_set);
    pylite_dict_free(from_list);
    pylite_list_free(roundtrip);
    pylite_set_free(set);
    pylite_list_free(list);
}

static void
test_collision_worst_case(void)
{
    PyLiteSet *set = pylite_set_new();
    PyLiteDict *dict = pylite_dict_new();
    assert(set != NULL && dict != NULL);
    for (int64_t i = 0; i < 128; i++) {
        PyLiteObject *key = pylite_collider_new(i);
        PyLiteObject *value = integer(i);
        assert(key != NULL);
        assert(pylite_set_add(set, key) >= 0);
        assert(pylite_dict_set(dict, key, value) == 0);
        pylite_decref(key);
        pylite_decref(value);
    }
    PyLiteObject *missing = pylite_collider_new(999);
    assert(missing != NULL);
    pylite_set_reset_stats(set);
    pylite_dict_reset_stats(dict);
    assert(!pylite_set_contains(set, missing));
    assert(!pylite_dict_contains(dict, missing));
    assert(pylite_set_stats(set)->comparisons >= 128);
    assert(pylite_dict_stats(dict)->comparisons >= 128);
    pylite_decref(missing);
    pylite_set_free(set);
    pylite_dict_free(dict);
}

static uint32_t
next_random(uint32_t *state)
{
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static void
test_randomized_set_and_dict(void)
{
    enum { KEY_COUNT = 257, OPERATIONS = 50000 };
    bool set_present[KEY_COUNT] = {false};
    bool dict_present[KEY_COUNT] = {false};
    int64_t dict_values[KEY_COUNT] = {0};
    size_t expected_set_size = 0;
    size_t expected_dict_size = 0;
    uint32_t random_state = UINT32_C(0x12345678);
    PyLiteSet *set = pylite_set_new();
    PyLiteDict *dict = pylite_dict_new();
    assert(set != NULL && dict != NULL);

    for (size_t operation = 0; operation < OPERATIONS; operation++) {
        uint32_t random_value = next_random(&random_state);
        size_t key_index = random_value % KEY_COUNT;
        unsigned action = (random_value >> 16) % 6;
        PyLiteObject *key = integer((int64_t)key_index);

        if (action == 0) {
            int result = pylite_set_add(set, key);
            assert(result == (set_present[key_index] ? 0 : 1));
            if (!set_present[key_index]) {
                set_present[key_index] = true;
                expected_set_size++;
            }
        }
        else if (action == 1) {
            int result = pylite_set_discard(set, key);
            assert(result == (set_present[key_index] ? 1 : 0));
            if (set_present[key_index]) {
                set_present[key_index] = false;
                expected_set_size--;
            }
        }
        else if (action == 2) {
            assert(pylite_set_contains(set, key) == set_present[key_index]);
        }
        else if (action == 3) {
            int64_t model_value = (int64_t)operation;
            PyLiteObject *value = integer(model_value);
            assert(pylite_dict_set(dict, key, value) == 0);
            pylite_decref(value);
            if (!dict_present[key_index]) {
                dict_present[key_index] = true;
                expected_dict_size++;
            }
            dict_values[key_index] = model_value;
        }
        else if (action == 4) {
            int result = pylite_dict_del(dict, key);
            assert(result == (dict_present[key_index] ? 0 : -1));
            pylite_clear_error();
            if (dict_present[key_index]) {
                dict_present[key_index] = false;
                expected_dict_size--;
            }
        }
        else {
            PyLiteObject *value = pylite_dict_get(dict, key);
            assert((value != NULL) == dict_present[key_index]);
            if (value != NULL) {
                assert(pylite_int_value(value) == dict_values[key_index]);
            }
        }
        pylite_decref(key);
        assert(pylite_set_len(set) == expected_set_size);
        assert(pylite_dict_len(dict) == expected_dict_size);
    }

    for (size_t i = 0; i < KEY_COUNT; i++) {
        PyLiteObject *key = integer((int64_t)i);
        assert(pylite_set_contains(set, key) == set_present[i]);
        PyLiteObject *value = pylite_dict_get(dict, key);
        assert((value != NULL) == dict_present[i]);
        if (value != NULL) {
            assert(pylite_int_value(value) == dict_values[i]);
        }
        pylite_decref(key);
    }
    pylite_set_free(set);
    pylite_dict_free(dict);
}

int
main(void)
{
    test_objects();
    test_list();
    test_set();
    test_dict();
    test_conversions();
    test_collision_worst_case();
    test_randomized_set_and_dict();
    puts("all tests passed");
    return 0;
}
