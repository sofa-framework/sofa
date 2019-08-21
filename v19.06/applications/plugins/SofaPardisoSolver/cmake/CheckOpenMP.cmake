# Multi precision toolbox for Scilab
# Copyright (C) 2009 - Jonathan Blanchard
#
# This file must be used under the terms of the CeCILL.
# This source file is licensed as described in the file COPYING, which
# you should have received as part of this distribution.  The terms
# are also available at
# http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt

# Check if the compiler supports OpenMP and attempts to identify the required compiler flags.

FUNCTION(CHECK_OPENMP)

    IF( DEFINED OMP_C_FLAGS AND DEFINED OMP_LINK_FLAGS AND DEFINED OMP_LINK_LIBS AND DEFINED HAVE_OPENMP )
        RETURN()
    ENDIF()

    MESSAGE(STATUS "Detecting OpenMP configuration")

    INCLUDE(CheckIncludeFile)

    CHECK_INCLUDE_FILE(omp.h OMP_H)

    IF(NOT OMP_H)
        MESSAGE(STATUS "Compiler does not appear to support OpenMP")
        RETURN()
    ENDIF()

    MESSAGE(STATUS "Checking for the required compiler and linker flags")

    # Try to use GOMP the gnu OpenMP implementation of GCC.
    TRY_COMPILE(OMP_CHECK ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake_modules/ompcheck.c
                    CMAKE_FLAGS "-DCMAKE_C_FLAGS:STRING=-fopenmp" "-DCMAKE_EXE_LINKER_FLAGS:STRING=-fopenmp")
    IF(OMP_CHECK)
        SET(OMP_C_FLAGS "-fopenmp" CACHE STRING "Flags required to build OpenMP applications.")
        SET(OMP_LINK_FLAGS "-fopenmp" CACHE STRING "Flags required to link OpenMP applications.")
        SET(OMP_LINK_LIBS "" CACHE STRING "Libraries required to link OpenMP applications.")
        SET(HAVE_OPENMP 1 CACHE BOOL "Set if OpenMP is available.")
        MESSAGE(STATUS "OpenMP C compiler flags - '-fopenmp'")
        MESSAGE(STATUS "OpenMP C linker flags - '-fopenmp'")
        MESSAGE(STATUS "OpenMP link libraries - none")
    ENDIF()

    # Try to use GOMP the gnu OpenMP implementation of GCC with explicit linking of libgomp.
    IF(NOT OMP_CHECK)
        TRY_COMPILE(OMP_CHECK ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake_modules/ompcheck.c
                        CMAKE_FLAGS "-DCMAKE_C_FLAGS:STRING=-fopenmp" "-DLINK_LIBRARIES:STRING=gomp")
        IF(OMP_CHECK)
            SET(OMP_C_FLAGS "-fopenmp" CACHE STRING "Flags required to build OpenMP applications.")
            SET(OMP_LINK_FLAGS "" CACHE STRING "Flags required to link OpenMP applications.")
            SET(OMP_LINK_LIBS "gomp" CACHE STRING "Libraries required to link OpenMP applications.")
            SET(HAVE_OPENMP 1 CACHE BOOL "Set if OpenMP is available.")
            MESSAGE(STATUS "OpenMP C compiler flags - '-fopenmp'")
            MESSAGE(STATUS "OpenMP C linker flags - none")
            MESSAGE(STATUS "OpenMP link libraries - libgomp")
        ENDIF()
    ENDIF()

    # Try to use the Sun Studio OpenMP implementation.
    IF(NOT OMP_CHECK)
        TRY_COMPILE(OMP_CHECK ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake_modules/ompcheck.c
                        CMAKE_FLAGS "-DCMAKE_C_FLAGS:STRING='-xopenmp -xO4'" "-DCMAKE_EXE_LINKER_FLAGS:STRING=-xopenmp")
        IF(OMP_CHECK)
            SET(OMP_C_FLAGS "-xopenmp -xO4" CACHE STRING "Flags required to build OpenMP applications.")
            SET(OMP_LINK_FLAGS "-xopenmp" CACHE STRING "Flags required to link OpenMP applications.")
            SET(OMP_LINK_LIBS "" CACHE STRING "Libraries required to link OpenMP applications.")
            SET(HAVE_OPENMP 1 CACHE BOOL "Set if OpenMP is available.")
            MESSAGE(STATUS "OpenMP C compiler flags - '-xopenmp -xO4'")
            MESSAGE(STATUS "OpenMP C linker flags - '-xopenmp'")
            MESSAGE(STATUS "OpenMP link libraries - none")
        ENDIF()
    ENDIF()

    # Try to use the Intel Compiler OpenMP implementation.
    IF(NOT OMP_CHECK)
        TRY_COMPILE(OMP_CHECK ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake_modules/ompcheck.c
                        CMAKE_FLAGS "-DCMAKE_C_FLAGS:STRING=-openmp" "-DCMAKE_EXE_LINKER_FLAGS:STRING=-openmp")
        IF(OMP_CHECK)
            SET(OMP_C_FLAGS "-openmp" CACHE STRING "Flags required to build OpenMP applications.")
            SET(OMP_LINK_FLAGS "-openmp" CACHE STRING "Flags required to link OpenMP applications.")
            SET(OMP_LINK_LIBS "" CACHE STRING "Libraries required to link OpenMP applications.")
            SET(HAVE_OPENMP 1 CACHE BOOL "Set if OpenMP is available.")
            MESSAGE(STATUS "OpenMP C compiler flags - '-openmp'")
            MESSAGE(STATUS "OpenMP C linker flags - '-openmp'")
            MESSAGE(STATUS "OpenMP link libraries - none")
        ENDIF()
    ENDIF()

    IF(NOT OMP_CHECK)
        SET(HAVE_OPENMP 0 CACHE BOOL "Set if OpenMP is available.")
        SET(OMP_C_FLAGS "" CACHE STRING "Flags required to build OpenMP applications.")
        SET(OMP_LINK_FLAGS "" CACHE STRING "Flags required to link OpenMP applications.")
        SET(OMP_LINK_LIBS "" CACHE STRING "Libraries required to link OpenMP applications.")
        MESSAGE(STATUS "Unsupported compiler")
        RETURN()
    ENDIF()

ENDFUNCTION()
