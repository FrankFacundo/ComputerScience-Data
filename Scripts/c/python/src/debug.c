#include "internal.h"

#include <stdio.h>

static bool tracing_enabled;
static char error_buffer[256];

const char *
pylite_last_error(void)
{
    return error_buffer[0] == '\0' ? "no error" : error_buffer;
}

void
pylite_clear_error(void)
{
    error_buffer[0] = '\0';
}

void
pylite_error_set(const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error_buffer, sizeof(error_buffer), format, arguments);
    va_end(arguments);
}

void
pylite_trace_set(bool enabled)
{
    tracing_enabled = enabled;
}

bool
pylite_trace_enabled(void)
{
    return tracing_enabled;
}

void
pylite_trace(const char *format, ...)
{
    if (!tracing_enabled) {
        return;
    }
    va_list arguments;
    va_start(arguments, format);
    fputs("[pylite] ", stderr);
    (void)vfprintf(stderr, format, arguments);
    fputc('\n', stderr);
    va_end(arguments);
}

bool
pylite_size_multiply(size_t left, size_t right, size_t *result)
{
    if (left != 0 && right > SIZE_MAX / left) {
        pylite_error_set("allocation size overflow");
        return false;
    }
    *result = left * right;
    return true;
}
