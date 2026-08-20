# Implementation guide

This guide is a reading order for the code. Run the debug playground with
--trace and stop at the named functions.

## 1. Object references and hashes

Read src/object.c first.

Every collection stores PyLiteObject pointers, just as CPython collections
store PyObject pointers. Insertion increments the reference count. Removal
either decrements it or transfers the owned reference to the caller for pop
operations.

Hash-table entries cache a key's hash. This avoids recomputing a string hash
at every probe. Equality is only tested when cached hashes match.

The Collider type is intentionally artificial but models normal Python code:

    class Collider:
        def __init__(self, identity):
            self.identity = identity
        def __hash__(self):
            return 42
        def __eq__(self, other):
            return self.identity == other.identity

## 2. List

Read src/list.c in this order:

1. pylite_list_resize
2. pylite_list_append
3. pylite_list_insert
4. pylite_list_pop
5. pylite_list_contains and pylite_list_index

The list structure is:

    size       number of visible elements
    allocated  number of pointer slots
    items      contiguous PyLiteObject pointer array

Resize computes:

    allocated = (new_size + new_size / 8 + 6) rounded down to multiple of 4

This gives the characteristic capacity sequence 0, 4, 8, 16, 24, 32, 40,
52, 64, 76, and so on. A long append sequence reallocates only occasionally,
so the total number of moved pointers across n appends is O(n). That is why
one append is O(1) amortized even though a resizing append is O(n).

Method mechanics:

| Method | Implementation |
|---|---|
| get/set | Normalize negative index, then directly access items[index] |
| append | Resize to size+1 if needed and store one owned pointer |
| extend | Resize once, increment each incoming reference, copy pointers |
| insert | Resize and memmove the suffix one position right |
| pop | Transfer one reference and memmove the suffix left |
| remove | Linear equality search followed by pop at the found index |
| index/contains/count | Linear equality scan |
| reverse | Swap pointers from the two ends |
| copy | Allocate a list and extend from the source |
| clear | Detach storage, then decrement references in reverse order |
| sort | Stable bottom-up merge sort in PyLite; see the limitation in README |

## 3. Set

Read src/set.c in this order:

1. pylite_set_lookup
2. pylite_set_add
3. pylite_set_table_resize
4. pylite_set_discard
5. pylite_set_pop

Each slot has a key pointer and cached hash. Slots have three states:

| State | key | hash |
|---|---|---:|
| Empty, never used | NULL | 0 |
| Active | object pointer | cached object hash |
| Dummy, deleted | dummy sentinel | -1 |

An unsuccessful search must stop only at EMPTY. Stopping at DUMMY would be a
bug: a colliding key may have been placed later in the probe sequence before
the deletion occurred.

For hash h and mask table_size-1:

    i = h & mask
    inspect i and up to nine following slots when they fit
    perturb >>= 5
    i = (i * 5 + 1 + perturb) & mask
    repeat until EMPTY or equal key

Adjacent probes improve cache locality. The perturb recurrence eventually
uses high bits of the hash instead of relying only on its low bits.

fill counts ACTIVE+DUMMY; used counts only ACTIVE. Resize is triggered by fill
because dummy accumulation also makes searches slower. Rebuilding reinserts
only active entries and therefore purges all dummies.

Set algebra deliberately iterates the smaller operand for intersection and
disjoint tests. That gives intersection average complexity O(min(n, m)) when
membership tests remain average O(1).

## 4. Dict

Read src/dict.c in this order:

1. dict_index_get and dict_index_set
2. pylite_dict_lookup
3. pylite_dict_set
4. delete_at
5. pylite_dict_resize
6. pylite_dict_popitem

The compact combined layout has two arrays:

    sparse indices: [EMPTY, 2, EMPTY, DUMMY, 0, ...]
                              |                |
                              v                v
    dense entries:  [(key0, value0), (key1, value1), (key2, value2), ...]

The dense array is appended in insertion order. Iteration walks that array,
which is why normal dict iteration is ordered. The sparse array contains
signed entry numbers rather than keys. Its element width grows from 1 to 2,
4, or 8 bytes only when required.

Lookup follows the same perturb recurrence as CPython:

    i = hash & mask
    perturb >>= 5
    i = (i * 5 + perturb + 1) & mask

The code handles two probes per loop iteration because CPython manually
unrolls this hot loop.

Deletion changes the sparse index to DUMMY and clears the dense entry. A new
key is appended to the dense entries array, even when its sparse probe reuses
a DUMMY. Thus deleting and reinserting a key moves it to the end of iteration
order.

At most two thirds of sparse slots are usable. When no dense entry space
remains, growth chooses the smallest power-of-two table at least used*3.
Resize also compacts the dense entries, removing deletion holes.

Method mechanics:

| Method | Implementation |
|---|---|
| get/contains | Hash, probe sparse indices, check dense entry |
| set existing | Replace only the value; insertion position is unchanged |
| set new | Find a reusable sparse slot and append a dense entry |
| del | Mark sparse position DUMMY and clear the dense entry |
| setdefault | Lookup, then insert default only on a miss |
| pop | Lookup and transfer the value's owned reference |
| popitem | Scan backward to the last live dense entry; LIFO result |
| update/copy | Iterate source entries in order and call set |
| clear | Decrement all live key/value references and reset the table |

## 5. Conversions

Read src/convert.c. Every conversion is a linear visit plus the destination
operation:

- list to set: n average-constant set insertions, O(n) average;
- set to list: visit every table slot, O(capacity). This is normally O(n), but
  capacity can remain large after many deletions;
- dict to list/set: visit the dense entries, O(nentries). Deletion holes mean
  nentries can exceed the current len(dict) until a resize compacts them;
- pairs list to dict: n average-constant dict assignments, O(n) average;
- fromkeys: one dict assignment per input item, O(n) average.

Duplicate removal does not change the construction complexity. Adversarial
same-hash keys can make hash-based conversions O(n squared).

## 6. Instrumentation

Each container records:

| Counter | Meaning |
|---|---|
| lookups | Top-level search/add operations |
| probes | Slots or list elements inspected |
| comparisons | Equality/order calls |
| resizes | Backing table/array replacements |
| moved_slots | Pointers or entries moved by shifting/rebuild/sort |

Call the reset_stats function immediately before the operation being studied.
The benchmark demonstrates this pattern.

## Suggested debugger experiments

1. Stop in pylite_list_resize and record the capacity after each append.
2. Stop in pylite_set_lookup and compare a normal integer with Collider keys.
3. Delete a set element and inspect key/hash in the former slot.
4. Stop in pylite_dict_set and inspect indices and entries separately.
5. Delete and reinsert a dict key, then observe its iteration position.
6. Run the collision benchmark and compare comparisons with n.
