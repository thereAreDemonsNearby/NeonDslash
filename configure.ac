#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT(neon_wilson_dslash, 1.0, shiletong@outlook.com)
AC_CONFIG_SRCDIR([lib/dslash_table.cc])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_RANLIB

AM_INIT_AUTOMAKE

dnl check for QMP -- needed for parallel configures
AC_ARG_WITH(qmp,
  AC_HELP_STRING(
    [--with-qmp=DIR],
    [Build Assembler on top of QMP, where QMP is installed in DIR.]
  ),
  [ QMP_HOME="${with_qmp}"
    QMP_GIVEN="yes"
  ],
  [ QMP_GIVEN="no" ]
)

dnl Use OpenMP threading
AC_ARG_ENABLE(openmp,
   AC_HELP_STRING(
    [--enable-openmp],
    [Enable building of OpenMP dispatcher. Pass flags necessary for OpenMP on CFLAGS]
   ),
   [ omp_enabled="yes" ],
   [ omp_enabled="no" ]
)

AC_ARG_WITH(qdp,
  AC_HELP_STRING(
     [--with-qdp=DIR],
     [ Build Assembler with QDP++, where QDP++ is installed in DIR. ]
  ),
  [ QDPXX_HOME="${with_qdp}"
    QDPXX_GIVEN="yes"
  ],
  [ QDPXX_GIVEN="no" ]
)

if test "X${QDPXX_GIVEN}X" = "XyesX"; then
     dnl Find the QDP++ config program 
     if test "X${QDPXX_HOME}X" = "XX" ; 
     then
	AC_PATH_PROG(QDPXX_CONFIG, [qdp++-config], [])
     else
	AC_PATH_PROG(QDPXX_CONFIG, [qdp++-config], [], [${QDPXX_HOME}/bin:${PATH}])
     fi

     if test "X${QDPXX_CONFIG}X" = "XX" ; then
	AC_MSG_ERROR([QDP++ configuration program qdp++-config not found.])
     fi

     AC_MSG_NOTICE([Found QDP++ configuration program ${QDPXX_CONFIG}])
     AC_MSG_CHECKING([if QDP++ uses checkerboarded layout])
     layout=`${QDPXX_CONFIG} --layout`
     if test "X${layout}X" == "Xcb2X" -o "X${layout}X" == "Xcb3dX" ; then 
	AC_MSG_RESULT([yes])
     else
        AC_MSG_RESULT([no])
	AC_MSG_ERROR([This Dslash Operator only works for checkerboarded QDP++. Sorry])
     fi


     AC_SUBST(NEONXX_AUX_CFLAGS, "`${QDPXX_CONFIG} --cxxflags`")
     AC_MSG_NOTICE([QDP++ compile flags: ${NEONXX_AUX_CFLAGS}])

     AC_SUBST(NEONXX_AUX_LDFLAGS,  "`${QDPXX_CONFIG} --ldflags`")
     AC_MSG_NOTICE([QDP++ linking flags: ${NEONXX_AUX_LDFLAGS}])

     AC_SUBST(NEONXX_AUX_LIBS,     "`${QDPXX_CONFIG} --libs`")
     AC_MSG_NOTICE([QDP++ libraries flags: ${NEONXX_AUX_LIBS}])

     QDPXX_ND="`${QDPXX_CONFIG} --Nd`"
     AC_MSG_NOTICE([QDP++ spacetime dimension: ${QDPXX_ND}])
     
     QDPXX_NC="`${QDPXX_CONFIG} --Nc`"
     AC_MSG_NOTICE([QDP++ number of colors: ${QDPXX_NC}])

     QDPXX_NS="`${QDPXX_CONFIG} --Ns`"
     AC_MSG_NOTICE([QDP++ number of spin components: ${QDPXX_NS}])

     PARALLEL_ARCH="`${QDPXX_CONFIG} --parallel-arch`"

     AC_MSG_NOTICE([QDP++ has parallel arch: ${PARALLEL_ARCH} ] )

     dnl This function is defined in acinclude.m4
     dnl it tries to try to compile the program 
    dnl  PAC_QDPXX_LINK_CXX_FUNC(
    dnl 	${NEONXX_AUX_CFLAGS},
    dnl 	${NEONXX_AUX_LDFLAGS},
    dnl 	${NEONXX_AUX_LIBS},
    dnl 	,
    dnl 	,
    dnl 	[qdpxx_link_ok=yes],
    dnl 	[qdpxx_link_ok=no]
    dnl  )

    dnl  AC_MSG_CHECKING([if we can compile/link a simple QDP++ program])
    dnl  if test "X${qdpxx_link_ok}X" = "XyesX" ; then 
    dnl 	AC_MSG_RESULT(yes)
    dnl  else 
    dnl 	AC_MSG_RESULT(no)
    dnl 	AC_MSG_ERROR([Cannot compile/link a program with QDP++.
    dnl Use --with-qdp++=<dir> to select a working version.])
    dnl fi



    dnl *********************************************************
    dnl ***              end of section with QDP++            ***
    dnl *********************************************************
else 
     dnl ***************************************************************
     dnl *** Section without QDP++ -- Need to check for QMP possibly ***
     dnl ***************************************************************
     case "${PARALLEL_ARCH}" in
	  scalar)
		AC_MSG_NOTICE([ Scalar Build! Not checking for QMP ])
		AC_SUBST(NEONXX_AUX_CFLAGS)
		AC_SUBST(NEONXX_AUX_LDFLAGS)
		AC_SUBST(NEONXX_AUX_LIBS)
		;;

	  parscalar)
		AC_MSG_NOTICE([ Parscalar build! Checking for QMP ])

		dnl look for qmp-config program
		if test "X${QMP_HOME}X" = "XX" ; then
		   AC_PATH_PROG([QMP_CONFIG], [qmp-config], [])
		else
		   AC_PATH_PROG([QMP_CONFIG], [qmp-config], [], [${QMP_HOME}/bin:${PATH}])
		fi

		dnl look for flags needed for QMP
		if test "X${QMP_CONFIG}X" != "XX" ; then
		   AC_MSG_NOTICE([Found QMP configuration program ${QMP_CONFIG}])
		   AC_SUBST(NEONXX_AUX_CFLAGS,  "`${QMP_CONFIG} --cflags`")
		   AC_MSG_NOTICE([QMP compile flags: ${NEONXX_AUX_CFLAGS}])

		   AC_SUBST(NEONXX_AUX_LDFLAGS, "`${QMP_CONFIG} --ldflags`")
		   AC_MSG_NOTICE([QMP linking flags: ${NEONXX_AUX_LDFLAGS}])

		   AC_SUBST(NEONXX_AUX_LIBS,    "`${QMP_CONFIG} --libs`")
		   AC_MSG_NOTICE([QMP libraries flags: ${NEONXX_AUX_LIBS}])
		else 
		   AC_MSG_WARN([QMP configuration program qmp-config not found.])
		   AC_MSG_WARN([Set environment variables QMP_CFLAGS QMP_LDFAGS QMP_LIBS before configure])
		fi

		dnl verify we can compile and link against QMP, if needed
		AC_PROG_CXX
		dnl PAC_QMP_LINK_CC_FUNC(
		dnl 	[${NEONXX_AUX_CFLAGS}],
		dnl 	[${NEONXX_AUX_LDFLAGS}],
		dnl 	[${NEONXX_AUX_LIBS}],
		dnl 	,
		dnl 	,
		dnl 	[qmp_link_ok=yes],
		dnl 	[qmp_link_ok=no]
		dnl )
		
		dnl AC_MSG_CHECKING([if we can compile/link of a simple QMP C program])
		dnl if test "X${qmp_link_ok}X" = "XyesX" ; then
		dnl    AC_MSG_RESULT(yes)
		dnl else
		dnl    AC_MSG_RESULT(no)
		dnl    AC_MSG_ERROR([Cannot compile/link a basic QMP C program! Check QMP_CFLAGS, QMP_LDFLAGS, QMP_LIBS.])
		dnl fi
		;;
	*)
		AC_MSG_ERROR([ Unknown value for --enable-parallel-arch ])
		;;
	esac
fi

if test "X${omp_enabled}X" == "XyesX";
then
	AC_MSG_NOTICE([Configuring OpenMP Threading])
	if test "X${qmt_enabled}X" == "XyesX";
        then 
          AC_MSG_ERROR([Cannot have OpenMP and QMT threading defined simultaneously])
	fi
	  
	AC_DEFINE([DSLASH_USE_OMP_THREADS], [1], [ Use OpenMP Threads ])
fi


if test "X${QMP_GIVEN}X" == "XyesX";
then
	AC_MSG_NOTICE([QMP is given in ${QMP_HOME}, good])
	AC_SUBST(QMP_HOME, "${QMP_HOME}")
else
	AC_MSG_ERROR([QMP is not given, damn it])
fi

if test "X${QDPXX_GIVEN}X" == "XyesX";
then
	AC_SUBST(QDPXX_HOME, "${QDPXX_HOME}")
else
	AC_MSG_ERROR([QDPXX is not given, damn it])
fi

# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
AC_FUNC_MALLOC

AM_CONDITIONAL(BUILD_OMP, [test "x${omp_enabled}x" = "xyesx" ])
AC_CONFIG_FILES(Makefile)
AC_CONFIG_FILES(include/Makefile)
AC_CONFIG_FILES(lib/Makefile)
AC_OUTPUT
