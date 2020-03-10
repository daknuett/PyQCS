
objs= obj/bytecode.o \
	  obj/raw_state.o \
	  obj/linked_list.o \
	  obj/main.o \
	  obj/graph_operations.o \
	  obj/exec.o \
	  obj/sample.o

CC=gcc
CFLAG = -g -O -o

all: gqcs

gqcs: $(objs)
	gcc $(CFLAG) gqcs $(objs)

obj/%.o: src/%.c
	$(CC) $< -c $(CFLAG) $@

clean:
	-rm $(objs)
