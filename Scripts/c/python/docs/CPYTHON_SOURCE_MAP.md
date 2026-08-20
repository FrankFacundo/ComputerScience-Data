# CPython 3.14.7 source map

This file maps the standalone functions to the exact CPython 3.14.7 source
used to design them. Run make upstream to put the upstream files under
build/upstream/Python-3.14.7 for side-by-side reading.

The upstream line numbers below refer to the v3.14.7 release tarball.

## List

Upstream file:
https://github.com/python/cpython/blob/v3.14.7/Objects/listobject.c

| Python behavior | CPython function / line | PyLite function |
|---|---|---|
| storage growth/shrink | list_resize, 108 | pylite_list_resize |
| insert | list_insert_impl, 1128 and ins1 | pylite_list_insert |
| clear | py_list_clear_impl, 1145 | pylite_list_clear |
| copy | list_copy_impl, 1160 | pylite_list_copy |
| append | list_append_impl, 1177 | pylite_list_append |
| extend | list_extend_impl, 1468 | pylite_list_extend |
| pop | list_pop_impl, 1531 | pylite_list_pop |
| sort | list_sort_impl, 2900 | pylite_list_sort is intentionally simpler |
| reverse | list_reverse_impl, 3190 | pylite_list_reverse |
| index | list_index_impl, 3283 | pylite_list_index |
| count | list_count_impl, 3324 | pylite_list_count |
| remove | list_remove_impl, 3362 | pylite_list_remove |
| membership | list_contains, 650 | pylite_list_contains |

For the object layout, compare
https://github.com/python/cpython/blob/v3.14.7/Include/cpython/listobject.h
with PyLiteList in src/internal.h.

## Set

Upstream file:
https://github.com/python/cpython/blob/v3.14.7/Objects/setobject.c

| Python behavior | CPython function / line | PyLite function |
|---|---|---|
| lookup/probing | set_lookkey, 79 | pylite_set_lookup |
| add core | set_add_entry_takeref, 135 | pylite_set_add |
| clean reinsert | set_insert_clean, 265 | set_insert_clean |
| resize/rebuild | set_table_resize, 301 | pylite_set_table_resize |
| discard core | set_discard_entry, 394 | pylite_set_discard |
| pop | set_pop_impl, 720 | pylite_set_pop |
| update | set_update_impl, 1120 | pylite_set_update |
| copy | set_copy_impl, 1331 | pylite_set_copy |
| clear | set_clear_impl, 1368 | pylite_set_clear |
| union | set_union_impl, 1384 | pylite_set_union |
| intersection | set_intersection, 1446 | pylite_set_intersection |
| intersection update | set_intersection_update, 1566 | pylite_set_intersection_update |
| isdisjoint | set_isdisjoint_impl, 1648 | pylite_set_is_disjoint |
| difference update | set_difference_update_impl, 1777 | pylite_set_difference_update |
| symmetric difference update | set_symmetric_difference_update_impl, 2026 | pylite_set_symmetric_difference_update |
| symmetric difference | set_symmetric_difference_impl, 2073 | pylite_set_symmetric_difference |
| subset/superset | set_issubset_impl, 2129 / set_issuperset_impl, 2174 | pylite_set_is_subset / pylite_set_is_superset |
| public add/remove/discard | set_add_impl, 2261 / set_remove_impl, 2369 / set_discard_impl, 2409 | corresponding PyLite functions |

For the fill, used, mask, finger, entry, and smalltable layout, compare
https://github.com/python/cpython/blob/v3.14.7/Include/cpython/setobject.h
with PyLiteSet in src/internal.h.

## Dict

Upstream file:
https://github.com/python/cpython/blob/v3.14.7/Objects/dictobject.c

| Python behavior | CPython function / line | PyLite function |
|---|---|---|
| lookup/probing | do_lookup, 1002 | pylite_dict_lookup |
| find insert position | find_empty_slot, 1766 | find_empty_slot |
| insert/replace | insertdict, 1891 | pylite_dict_set |
| rebuild indices | build_indices_generic, 2007 | build_indices |
| resize/compact | dictresize, 2065 | pylite_dict_resize |
| delete | delitem_common, 2791 | delete_at |
| fromkeys | dict_fromkeys_impl, 3727 | pylite_dict_fromkeys_list / set |
| copy | dict_copy_impl, 4119 | pylite_dict_copy |
| get | dict_get_impl, 4387 | pylite_dict_get_default |
| setdefault | dict_setdefault_impl, 4542 | pylite_dict_setdefault |
| clear | dict_clear_impl, 4559 | pylite_dict_clear |
| pop | dict_pop_impl, 4580 | pylite_dict_pop |
| popitem | dict_popitem_impl, 4597 | pylite_dict_popitem |
| keys/items/values | dict_keys_impl, 6566 / dict_items_impl, 6678 / dict_values_impl, 6768 | visitors and list conversion helpers |

The full CPython dict has combined, split, general-key, and Unicode-key
variants. PyLite implements the generic combined form because that is the
form that most clearly exposes sparse indices, dense ordered entries,
probing, deletion holes, and compaction.

Compare CPython's internal keys layout at
https://github.com/python/cpython/blob/v3.14.7/Include/internal/pycore_dict.h
with PyLiteDict and PyLiteDictEntry in src/internal.h.

## Hashing

- CPython byte/string hashing:
  https://github.com/python/cpython/blob/v3.14.7/Python/pyhash.c
- CPython pair/tuple hash mixing:
  https://github.com/python/cpython/blob/v3.14.7/Objects/tupleobject.c
- PyLite implementation: src/object.c

PyLite deliberately fixes the SipHash key for repeatable probe traces.
CPython initializes a randomized secret for each process unless its hash seed
is configured.
