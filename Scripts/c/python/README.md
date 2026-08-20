# PyLite: CPython 3.14.7 containers in standalone C

PyLite is a debugger-friendly implementation of Python-like list, set, and
dict objects. It follows the important storage, growth, deletion, ordering,
and collision-resolution algorithms in the CPython 3.14.7 source.

It is designed to answer questions such as:

- Why is list membership linear?
- Why are set and dict lookups constant time on average but linear in the
  worst case?
- What does amortized constant time mean for list.append?
- Why do deleted hash-table entries become dummy markers?
- How does a dict remember insertion order without using a linked list?
- What work is performed by list, set, and dict conversions?

Start with docs/IMPLEMENTATION_GUIDE.md while stepping through the code.

## Build and run

From this directory:

    make
    ./build/release/playground
    make test
    make check
    make bench

make check runs the optimized test binary and undefined-behavior checks. On
non-macOS platforms it also enables AddressSanitizer; the Apple Clang runtime
available on this machine cannot reliably initialize AddressSanitizer.

To download and SHA-256-verify the exact upstream source beside the build:

    make upstream

The release build uses -O3. The benchmark reports both elapsed time and
internal probe counts. Probe counts are usually more useful for learning
complexity because they are deterministic and less sensitive to the machine.

To print every set/dict probe and every resize:

    make trace

## Debug one C line at a time

The debug build uses -O0, -g3, -fno-inline, and -fno-omit-frame-pointer:

    make debug
    lldb ./build/debug/playground

Useful LLDB commands:

    (lldb) breakpoint set --name pylite_list_resize
    (lldb) breakpoint set --name pylite_set_lookup
    (lldb) breakpoint set --name pylite_set_table_resize
    (lldb) breakpoint set --name pylite_dict_lookup
    (lldb) breakpoint set --name pylite_dict_resize
    (lldb) run --trace
    (lldb) next
    (lldb) step
    (lldb) frame variable
    (lldb) continue

Or load all useful breakpoints at once:

    lldb -s debug/lldb_commands.txt ./build/debug/playground

VS Code launch and build configurations are included under .vscode. They use
the CodeLLDB debugger type. Open this repository, select “PyLite: playground
(LLDB),” and press F5.

## The important answer about lookup complexity

| Container and operation | Average | Worst case | Why |
|---|---:|---:|---|
| value in list | O(n) | O(n) | Compare elements from left to right |
| value in set | O(1) | O(n) | Hash directs the search; collisions can form a long probe sequence |
| key in dict / dict[key] | O(1) | O(n) | Same open-addressing principle; values live in ordered entries |
| list append | O(1) amortized | O(n) | Most appends use spare capacity; a resize copies pointers |
| list insert/delete at i | O(n-i) | O(n) | Shift the suffix of the pointer array |
| set add/delete | O(1) | O(n) | A resize or adversarial collision chain can scan/reinsert many slots |
| dict set/delete | O(1) | O(n) | Same qualification; resize also compacts deleted entries |
| list/set/dict copy | O(n) | O(n) | Every live reference is visited |

There are two important qualifications:

1. Hashing a new string costs O(length of string). Its cached hash makes later
   container lookups independent of string length, apart from equality checks
   after collisions.
2. “O(1)” is an average-case statement based on a good hash distribution and
   a controlled load factor. It is not a promise that every lookup takes one
   probe.

The Collider object in the benchmark models a legal Python class whose
__hash__ always returns 42. It makes distinct keys collide deliberately, so
the linear worst case is visible:

    make bench

## What is faithful to CPython 3.14.7

List:

- contiguous array of owned object pointers;
- separate logical size and allocated capacity;
- the exact 3.14.7 overallocation expression and half-full shrink rule;
- pointer shifting for insert, remove, and non-tail pop;
- reference ownership and negative index normalization.

Set:

- fill, used, mask, table, finger, and an inline eight-slot small table;
- empty, active, and dummy states;
- cached hashes in entries;
- up to nine additional adjacent probes, followed by perturb probing;
- the 5*j + 1 + perturb recurrence and PERTURB_SHIFT of 5;
- the 3/5 fill trigger and CPython's used*4 / used*2 resize targets;
- finger-based pop and dummy purging during rebuild.

Dict:

- a compact, insertion-ordered combined table;
- a sparse indices array pointing into a dense entries array;
- signed 1-, 2-, 4-, or 8-byte indices as the table grows;
- EMPTY and DUMMY sparse-index states;
- a maximum usable fraction of two thirds;
- the same perturb recurrence and a two-probe unrolled lookup loop;
- growth based on used*3;
- deletion holes, compaction on resize, and LIFO popitem behavior.

Objects:

- owned references with increment/decrement operations;
- cached hashes;
- int, string, pair, None, and deliberate-collision keys;
- SipHash-1-3 for strings and CPython's 64-bit tuple-style pair mixing.

## Deliberate boundaries

This is a standalone teaching runtime, not a fork of the complete Python
interpreter. The core container algorithms are real; the following
interpreter-specific features are intentionally out of scope:

- arbitrary Python objects and user-defined equality callbacks;
- cyclic garbage collection and weak references;
- GIL-disabled atomic reads, per-object locks, and QSBR memory reclamation;
- dict split tables used by object instance attributes;
- dict watchers, key versions, and Unicode-only specialized tables;
- CPython's full Powersort/Timsort implementation.

PyLite list.sort is a stable merge sort with the same O(n log n) worst-case
complexity, but it is explicitly not a copy of CPython's adaptive list sort.
String hashing uses the same SipHash family with a fixed key so debugger traces
are reproducible; CPython randomizes its secret at process startup.

Therefore, use the release benchmark to study algorithmic behavior and this
implementation's C speed. Do not present its nanosecond values as CPython
interpreter benchmark results. For exact production behavior, step through a
debug build of CPython itself.

## Conversions

| PyLite function | Python meaning | Result |
|---|---|---|
| pylite_set_from_list | set(a_list) | Deduplicated elements |
| pylite_list_from_set | list(a_set) | Elements in table iteration order |
| pylite_set_from_dict | set(a_dict) | Dict keys |
| pylite_list_from_dict | list(a_dict) | Keys in insertion order |
| pylite_list_from_dict_items | list(a_dict.items()) | Pair objects in insertion order |
| pylite_list_from_dict_values | list(a_dict.values()) | Values in insertion order |
| pylite_dict_from_pairs | dict(pair_list) | Pair first/second become key/value |
| pylite_dict_fromkeys_list | dict.fromkeys(a_list, value) | Unique list items become keys |
| pylite_dict_fromkeys_set | dict.fromkeys(a_set, value) | Set items become keys |

A direct dict(a_set) only works in Python when every set element is itself a
two-element iterable. The fromkeys conversion is therefore the explicit,
general set-to-dict operation.

## Source layout

- include/pylite.h: public API and reference-ownership contract
- src/object.c: values, reference counts, equality, and hashing
- src/list.c: list methods and resize policy
- src/set.c: set table, probing, dummy entries, and set algebra
- src/dict.c: compact ordered dict, adaptive indices, and dict methods
- src/convert.c: all list/set/dict conversions
- examples/playground.c: small traceable scenario
- benchmarks/complexity.c: average and adversarial lookup experiment
- tests/test_objects.c: behavior, conversions, collisions, and ownership tests
- docs/CPYTHON_SOURCE_MAP.md: local-to-upstream method/function index

## Upstream source used

The implementation was checked against the exact v3.14.7 tag:

- https://github.com/python/cpython/blob/v3.14.7/Objects/listobject.c
- https://github.com/python/cpython/blob/v3.14.7/Objects/setobject.c
- https://github.com/python/cpython/blob/v3.14.7/Objects/dictobject.c
- https://github.com/python/cpython/blob/v3.14.7/Objects/tupleobject.c
- https://github.com/python/cpython/blob/v3.14.7/Python/pyhash.c
- https://github.com/python/cpython/blob/v3.14.7/Include/cpython/listobject.h
- https://github.com/python/cpython/blob/v3.14.7/Include/cpython/setobject.h
- https://github.com/python/cpython/blob/v3.14.7/Include/internal/pycore_dict.h

Python 3.14.7 was released on August 5, 2026. CPython is distributed under the
Python Software Foundation License; this project is an original standalone
adaptation for study and does not vendor CPython source files.
