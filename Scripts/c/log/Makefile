CC=gcc
CFLAGS=-Wall

log: log.c
	$(CC) $(CFLAGS) log.c -o log -lm

assembly: log.c
	$(CC) $(CFLAGS) -S log.c -lm
