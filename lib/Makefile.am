TOPSRCDIR=@top_srcdir@
TOPBUILDDIR=@top_builddir@
INCFLAGS= -I$(TOPSRCDIR)/include -I$(TOPBUILDDIR)/include -I@QMP_HOME@/include -I@QDPXX_HOME@/include
AM_CFLAGS = $(INCFLAGS) @CFLAGS@ @DEFS@
AM_CXXFLAGS = $(INCFLAGS) @CXXFLAGS@ @DEFS@
AM_CC  = $(CC)
AM_CXX = $(CXX)

.SUFFIXES=.c .h .cc .s. o

.c.s:
	$(CC) $(AM_CFLAGS) -S $< 

lib_LIBRARIES = libneondslash.a

libneondslash_a_SOURCES = dslash_table.cc \
	shift_table.cc \
	neon_dslash.cc \
	neon_dslash_impl.cc

