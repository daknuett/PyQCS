
objs= obj/bytecode.o \
	  obj/raw_state.o \
	  obj/linked_list.o \
	  obj/main.o \
	  obj/graph_operations.o

CC=gcc
CFLAG = -O -c -o

all: $(objs)

obj/%.o: src/%.c
	$(CC) $< $(CFLAG) $@

clean:
	-rm $(objs)
