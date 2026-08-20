#include "internal.h"

#include <inttypes.h>

#define ROTATE_LEFT(value, bits) \
    (((value) << (bits)) | ((value) >> (64u - (bits))))

PyLiteObject pylite_dummy_object = {
    .refcount = SIZE_MAX,
    .type = PYLITE_NONE,
    .hash_cached = true,
    .cached_hash = -1
};

static PyLiteObject none_object = {
    .refcount = SIZE_MAX,
    .type = PYLITE_NONE,
    .hash_cached = true,
    .cached_hash = 0x632be59bd9b4e019LL
};

static PyLiteObject *
object_allocate(PyLiteType type)
{
    PyLiteObject *object = calloc(1, sizeof(*object));
    if (object == NULL) {
        pylite_error_set("out of memory allocating object");
        return NULL;
    }
    object->refcount = 1;
    object->type = type;
    return object;
}

PyLiteObject *
pylite_none(void)
{
    return &none_object;
}

PyLiteObject *
pylite_int_new(int64_t value)
{
    PyLiteObject *object = object_allocate(PYLITE_INT);
    if (object != NULL) {
        object->as.integer = value;
    }
    return object;
}

PyLiteObject *
pylite_string_new(const char *s)
{
    if (s == NULL) {
        pylite_error_set("cannot construct a string from NULL");
        return NULL;
    }
    PyLiteObject *object = object_allocate(PYLITE_STRING);
    if (object == NULL) {
        return NULL;
    }
    object->as.string.length = strlen(s);
    object->as.string.bytes = malloc(object->as.string.length + 1);
    if (object->as.string.bytes == NULL) {
        free(object);
        pylite_error_set("out of memory copying string");
        return NULL;
    }
    memcpy(object->as.string.bytes, s, object->as.string.length + 1);
    return object;
}

PyLiteObject *
pylite_pair_new(PyLiteObject *first, PyLiteObject *second)
{
    if (first == NULL || second == NULL) {
        pylite_error_set("pair members cannot be NULL");
        return NULL;
    }
    PyLiteObject *object = object_allocate(PYLITE_PAIR);
    if (object == NULL) {
        return NULL;
    }
    pylite_incref(first);
    pylite_incref(second);
    object->as.pair.first = first;
    object->as.pair.second = second;
    return object;
}

PyLiteObject *
pylite_collider_new(int64_t identity)
{
    PyLiteObject *object = object_allocate(PYLITE_COLLIDER);
    if (object != NULL) {
        object->as.collider_identity = identity;
        object->hash_cached = true;
        object->cached_hash = 42;
    }
    return object;
}

void
pylite_incref(PyLiteObject *object)
{
    if (object != NULL && object->refcount != SIZE_MAX) {
        object->refcount++;
    }
}

void
pylite_decref(PyLiteObject *object)
{
    if (object == NULL || object->refcount == SIZE_MAX) {
        return;
    }
    if (--object->refcount != 0) {
        return;
    }
    if (object->type == PYLITE_STRING) {
        free(object->as.string.bytes);
    }
    else if (object->type == PYLITE_PAIR) {
        pylite_decref(object->as.pair.first);
        pylite_decref(object->as.pair.second);
    }
    free(object);
}

PyLiteType
pylite_type(const PyLiteObject *object)
{
    return object == NULL ? PYLITE_NONE : object->type;
}

int64_t
pylite_int_value(const PyLiteObject *object)
{
    return object != NULL && object->type == PYLITE_INT ? object->as.integer : 0;
}

const char *
pylite_string_value(const PyLiteObject *object)
{
    return object != NULL && object->type == PYLITE_STRING
        ? object->as.string.bytes : NULL;
}

PyLiteObject *
pylite_pair_first(const PyLiteObject *object)
{
    return object != NULL && object->type == PYLITE_PAIR
        ? object->as.pair.first : NULL;
}

PyLiteObject *
pylite_pair_second(const PyLiteObject *object)
{
    return object != NULL && object->type == PYLITE_PAIR
        ? object->as.pair.second : NULL;
}

/* SipHash-1-3, CPython 3.14's default byte-hash family. A fixed teaching
 * secret makes traces reproducible; CPython randomizes this per process. */
static uint64_t
siphash13(const unsigned char *data, size_t length)
{
    const uint64_t k0 = UINT64_C(0x0706050403020100);
    const uint64_t k1 = UINT64_C(0x0f0e0d0c0b0a0908);
    uint64_t v0 = UINT64_C(0x736f6d6570736575) ^ k0;
    uint64_t v1 = UINT64_C(0x646f72616e646f6d) ^ k1;
    uint64_t v2 = UINT64_C(0x6c7967656e657261) ^ k0;
    uint64_t v3 = UINT64_C(0x7465646279746573) ^ k1;
    const unsigned char *end = data + (length & ~(size_t)7);

#define SIPROUND() do { \
    v0 += v1; v1 = ROTATE_LEFT(v1, 13); v1 ^= v0; v0 = ROTATE_LEFT(v0, 32); \
    v2 += v3; v3 = ROTATE_LEFT(v3, 16); v3 ^= v2; \
    v0 += v3; v3 = ROTATE_LEFT(v3, 21); v3 ^= v0; \
    v2 += v1; v1 = ROTATE_LEFT(v1, 17); v1 ^= v2; v2 = ROTATE_LEFT(v2, 32); \
} while (0)

    for (; data != end; data += 8) {
        uint64_t lane = 0;
        for (unsigned i = 0; i < 8; i++) {
            lane |= (uint64_t)data[i] << (8u * i);
        }
        v3 ^= lane;
        SIPROUND();
        v0 ^= lane;
    }

    uint64_t tail = (uint64_t)length << 56;
    for (size_t i = 0; i < (length & 7u); i++) {
        tail |= (uint64_t)end[i] << (8u * i);
    }
    v3 ^= tail;
    SIPROUND();
    v0 ^= tail;
    v2 ^= 0xff;
    SIPROUND();
    SIPROUND();
    SIPROUND();
#undef SIPROUND
    return v0 ^ v1 ^ v2 ^ v3;
}

static PyLiteHash
pair_hash(const PyLiteObject *object)
{
    /* Same xxHash-inspired tuple mixing constants used by 64-bit CPython. */
    uint64_t accumulator = UINT64_C(2870177450012600261);
    const PyLiteObject *members[2] = {
        object->as.pair.first, object->as.pair.second
    };
    for (size_t i = 0; i < 2; i++) {
        uint64_t lane = (uint64_t)pylite_hash(members[i]);
        accumulator += lane * UINT64_C(14029467366897019727);
        accumulator = ROTATE_LEFT(accumulator, 31);
        accumulator *= UINT64_C(11400714785074694791);
    }
    accumulator += 2u ^ (UINT64_C(2870177450012600261) ^ UINT64_C(3527539));
    if ((PyLiteHash)accumulator == PYLITE_HASH_ERROR) {
        return (PyLiteHash)UINT64_C(1546275796);
    }
    return (PyLiteHash)accumulator;
}

PyLiteHash
pylite_hash(const PyLiteObject *object)
{
    if (object == NULL) {
        pylite_error_set("cannot hash NULL");
        return PYLITE_HASH_ERROR;
    }
    if (object->hash_cached) {
        return object->cached_hash;
    }

    PyLiteHash hash;
    switch (object->type) {
        case PYLITE_NONE:
            hash = none_object.cached_hash;
            break;
        case PYLITE_INT:
            /* Exact for CPython compact integers; remap -1, its error sentinel. */
            hash = (PyLiteHash)object->as.integer;
            if (hash == PYLITE_HASH_ERROR) {
                hash = -2;
            }
            break;
        case PYLITE_STRING:
            hash = (PyLiteHash)siphash13(
                (const unsigned char *)object->as.string.bytes,
                object->as.string.length);
            if (hash == PYLITE_HASH_ERROR) {
                hash = -2;
            }
            break;
        case PYLITE_PAIR:
            hash = pair_hash(object);
            break;
        case PYLITE_COLLIDER:
            hash = 42;
            break;
        default:
            pylite_error_set("unknown object type");
            return PYLITE_HASH_ERROR;
    }
    ((PyLiteObject *)object)->cached_hash = hash;
    ((PyLiteObject *)object)->hash_cached = true;
    return hash;
}

bool
pylite_equal(const PyLiteObject *left, const PyLiteObject *right)
{
    if (left == right) {
        return true;
    }
    if (left == NULL || right == NULL || left->type != right->type) {
        return false;
    }
    switch (left->type) {
        case PYLITE_NONE:
            return true;
        case PYLITE_INT:
            return left->as.integer == right->as.integer;
        case PYLITE_STRING:
            return left->as.string.length == right->as.string.length &&
                memcmp(left->as.string.bytes, right->as.string.bytes,
                       left->as.string.length) == 0;
        case PYLITE_PAIR:
            return pylite_equal(left->as.pair.first, right->as.pair.first) &&
                pylite_equal(left->as.pair.second, right->as.pair.second);
        case PYLITE_COLLIDER:
            return left->as.collider_identity == right->as.collider_identity;
    }
    return false;
}

int
pylite_compare(const PyLiteObject *left, const PyLiteObject *right, int *result)
{
    if (left == NULL || right == NULL || result == NULL) {
        pylite_error_set("invalid comparison argument");
        return -1;
    }
    if (left->type != right->type ||
        (left->type != PYLITE_INT && left->type != PYLITE_STRING &&
         left->type != PYLITE_COLLIDER)) {
        pylite_error_set("objects are not order-comparable");
        return -1;
    }
    if (left->type == PYLITE_STRING) {
        int comparison = strcmp(left->as.string.bytes, right->as.string.bytes);
        *result = (comparison > 0) - (comparison < 0);
    }
    else {
        int64_t left_value = left->type == PYLITE_INT
            ? left->as.integer : left->as.collider_identity;
        int64_t right_value = right->type == PYLITE_INT
            ? right->as.integer : right->as.collider_identity;
        *result = (left_value > right_value) - (left_value < right_value);
    }
    return 0;
}

static void
print_quoted_string(const char *s, FILE *stream)
{
    fputc('\'', stream);
    for (; *s != '\0'; s++) {
        if (*s == '\\' || *s == '\'') {
            fputc('\\', stream);
        }
        fputc(*s, stream);
    }
    fputc('\'', stream);
}

void
pylite_object_print(const PyLiteObject *object, FILE *stream)
{
    if (stream == NULL) {
        stream = stdout;
    }
    if (object == NULL) {
        fputs("<missing>", stream);
        return;
    }
    switch (object->type) {
        case PYLITE_NONE:
            fputs("None", stream);
            break;
        case PYLITE_INT:
            fprintf(stream, "%" PRId64, object->as.integer);
            break;
        case PYLITE_STRING:
            print_quoted_string(object->as.string.bytes, stream);
            break;
        case PYLITE_PAIR:
            fputc('(', stream);
            pylite_object_print(object->as.pair.first, stream);
            fputs(", ", stream);
            pylite_object_print(object->as.pair.second, stream);
            fputc(')', stream);
            break;
        case PYLITE_COLLIDER:
            fprintf(stream, "Collider(%" PRId64 ")", object->as.collider_identity);
            break;
    }
}
