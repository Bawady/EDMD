# define the C compiler to use
CC=gcc

# define any compile-time flags
CFLAGS=-Wall -Ofast -funroll-loops -march=native


# define any libraries to link into executable:
LIBS=-lm


2d: mdCell.c  mdCell.h
	$(CC) $(CFLAGS) mdCell.c -o $@ $(LIBS)

3d: mdCell.c  mdCell.h
	$(CC) $(CFLAGS) -DDIM_3D mdCell.c -o $@  $(LIBS)
