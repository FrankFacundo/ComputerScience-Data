CC ?= cc
AR ?= ar

CPPFLAGS := -Iinclude -D_POSIX_C_SOURCE=200809L
WARNINGS := -Wall -Wextra -Wpedantic -Wconversion -Wshadow
RELEASE_FLAGS := -std=c17 $(WARNINGS) -O3
DEBUG_FLAGS := -std=c17 $(WARNINGS) -O0 -g3 -fno-inline -fno-omit-frame-pointer \
	-DPYLITE_DEBUG
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SANITIZERS := -fsanitize=undefined
else
SANITIZERS := -fsanitize=address,undefined
endif
SANITIZE_FLAGS := -std=c17 $(WARNINGS) -O1 -g3 -fno-omit-frame-pointer \
	$(SANITIZERS)

LIB_SOURCES := src/debug.c src/object.c src/list.c src/set.c src/dict.c src/convert.c
RELEASE_OBJECTS := $(LIB_SOURCES:src/%.c=build/release/%.o)
DEBUG_OBJECTS := $(LIB_SOURCES:src/%.c=build/debug/%.o)
CPYTHON_VERSION := 3.14.7
CPYTHON_ARCHIVE := build/upstream/Python-$(CPYTHON_VERSION).tar.xz
CPYTHON_SHA256 := 3b48dac8fb59f62eaa67ac83c1eb12bda1b7a08406dd286e252c11a66be27f81

.PHONY: all release debug test check bench trace lldb sanitize upstream clean

all: release

release: build/release/playground build/release/complexity \
	build/release/test_objects

debug: build/debug/playground build/debug/test_objects

build/release/libpylite.a: $(RELEASE_OBJECTS)
	$(AR) rcs $@ $^

build/debug/libpylite.a: $(DEBUG_OBJECTS)
	$(AR) rcs $@ $^

build/release/%.o: src/%.c include/pylite.h src/internal.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) -c $< -o $@

build/debug/%.o: src/%.c include/pylite.h src/internal.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) -c $< -o $@

build/release/playground: examples/playground.c build/release/libpylite.a
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $< build/release/libpylite.a -o $@

build/release/complexity: benchmarks/complexity.c build/release/libpylite.a
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $< build/release/libpylite.a -o $@

build/release/test_objects: tests/test_objects.c build/release/libpylite.a
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $< build/release/libpylite.a -o $@

build/debug/playground: examples/playground.c build/debug/libpylite.a
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $< build/debug/libpylite.a -o $@

build/debug/test_objects: tests/test_objects.c build/debug/libpylite.a
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $< build/debug/libpylite.a -o $@

test: build/release/test_objects
	./build/release/test_objects

check: test sanitize

bench: build/release/complexity
	./build/release/complexity

trace: build/debug/playground
	./build/debug/playground --trace

lldb: build/debug/playground
	lldb ./build/debug/playground

upstream: build/upstream/Python-$(CPYTHON_VERSION)/Objects/listobject.c

build/upstream/Python-$(CPYTHON_VERSION)/Objects/listobject.c:
	@mkdir -p build/upstream
	curl -fsSL https://www.python.org/ftp/python/$(CPYTHON_VERSION)/Python-$(CPYTHON_VERSION).tar.xz \
		-o $(CPYTHON_ARCHIVE)
	cd build/upstream && echo "$(CPYTHON_SHA256)  Python-$(CPYTHON_VERSION).tar.xz" | \
		shasum -a 256 -c -
	tar -xJf $(CPYTHON_ARCHIVE) -C build/upstream

sanitize:
	@mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SANITIZE_FLAGS) $(LIB_SOURCES) tests/test_objects.c \
		-o build/sanitize/test_objects
	./build/sanitize/test_objects

clean:
	rm -rf build
