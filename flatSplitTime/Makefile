# The libraries for pythia, fastjet, and root ( "LIB_TRI" )
#PY8PATH=/home/davidstewart/software/pythia8310
#LIB_PYTH=-I${PY8PATH}/include -L${PY8PATH}/lib -lpythia8
#LIB_FASTJET=`/home/davidstewart/software/fastjet-install/bin/fastjet-config --cxxflags --libs`
LIB_ROOT=`root-config --cflags --glibs`
#LIB_TRI=${LIB_PYTH} ${LIB_FASTJET} ${LIB_ROOT} -lRecursiveTools
LIB_TRI = ${LIB_ROOT} 

CC=g++
CFLAGS=-std=c++17 -O3 -Wno-deprecated

DICT_HDR = src/FlatDict_LinkDef.h
DICT_SRC = obj/FlatDict.cxx
DICT_OBJ = obj/FlatDict.o

bin/flat3d: obj/flat3d.o ${DICT_OBJ}
	${CC} ${CFLAGS} -o $@ $^ ${LIB_TRI}

bin/flat32d: obj/flat32d.o ${DICT_OBJ}
	${CC} ${CFLAGS} -o $@ $^ ${LIB_TRI}

bin/flat: obj/flat.o ${DICT_OBJ}
	${CC} ${CFLAGS} -o $@ $^ ${LIB_TRI}

bin/profiles: obj/profiles.o ${DICT_OBJ}
	${CC} ${CFLAGS} -o $@ $^ ${LIB_TRI}

bin/cutsplice: obj/cutsplice.o ${DICT_OBJ}
	${CC} ${CFLAGS} -o $@ $^ ${LIB_TRI}

bin/coarser: obj/coarser.o ${DICT_OBJ}
	${CC} ${CFLAGS} -o $@ $^ ${LIB_TRI}

bin/stats: obj/stats.o ${DICT_OBJ}
	${CC} ${CFLAGS} -o $@ $^ ${LIB_TRI}

${DICT_SRC}:
	rootcling -f $@ -c ${DICT_HDR}

${DICT_OBJ}: ${DICT_SRC}
	${CC} ${CFLAGS} ${LIB_ROOT} -c $< -o $@



# compilation option
obj/profiles.o: src/profiles.cc 
	${CC} ${CFLAGS} ${LIB_TRI} -c $< -o $@

obj/cutsplice.o: src/cutsplice.cc 
	${CC} ${CFLAGS} ${LIB_TRI} -c $< -o $@

obj/coarser.o: src/coarser.cc 
	${CC} ${CFLAGS} ${LIB_TRI} -c $< -o $@

obj/stats.o: src/stats.cc 
	${CC} ${CFLAGS} ${LIB_TRI} -c $< -o $@

obj/flat.o: src/flat.cc 
	${CC} ${CFLAGS} ${LIB_TRI} -c $< -o $@

obj/flat3d.o: src/flat3d.cc 
	${CC} ${CFLAGS} ${LIB_TRI} -c $< -o $@

obj/flat32d.o: src/flat32d.cc
	${CC} ${CFLAGS} ${LIB_TRI} -c $< -o $@
