CC = gcc
CFLAGS = -O2 -fPIC
INCLUDES = -I/usr/include/python2.6
LIBS = -lz
TARGET_D = MPLn23d.so
TARGET_S = libMPLn23d.a
INSTALL_DIR = ..
OBJS = cut.o ln2d.o ln3d.o neigh2d.o neigh3d.o python.o rand.o

all:	$(TARGET_D) $(TARGET_S)

$(TARGET_D):	$(OBJS)
		$(CC) -shared -o $@ $(OBJS) $(LIBS)

$(TARGET_S):	$(OBJS)
		ar r $@ $(OBJS)

install:	$(TARGET_D)
		cp $(TARGET_D) $(INSTALL_DIR)

clean:
		rm $(TARGET_D) $(TARGET_S) $(OBJS)

.c.o:
		$(CC) $(CFLAGS) $(INCLUDES) -c $<

cut.c:	MPLn23d.h
ln2d.c:	MPLn23d.h
ln3d.c:	MPLn23d.h
neigh2d.c:	MPLn23d.h
neigh3d.c:	MPLn23d.h
python.c:	MPLn23d.h
rand.c:	MPLn23d.h
