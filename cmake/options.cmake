
# The option for any project in applications/ is OFF by default.
# Here, we override this during the preconfiguration step for projects everybody uses.
if(NOT PRECONFIGURE_DONE)
    macro(override_default name type default_value description)
        set(${name} "${default_value}" CACHE ${type} "${description}" FORCE)
        set(SOFA_OPTION_DEFAULT_VALUE_${name} "${default_value}" CACHE INTERNAL "Default value for ${name}")
    endmacro()

    if(NOT PS3)
        override_default(SOFA-APPLICATION_RUNSOFA BOOL ON "Build RunSofa application")
        override_default(SOFA-APPLICATION_MODELER BOOL ON "Build Modeler application")
        override_default(SOFA-APPLICATION_GENERATERIGID BOOL ON "Build GenerateRigid application")
    endif()
endif()

# CHG: disable as this breaks build of external applications
# set(CMAKE_INSTALL_PREFIX "${SOFA_BUILD_DIR}" CACHE INTERNAL "Sofa install path (not used yet)")

set(compilerDefines)

set(SOFA-EXTERNAL_INCLUDE_DIR ${SOFA-EXTERNAL_INCLUDE_DIR} CACHE PATH "Include path for pre-compiled dependencies outside of the Sofa directory")
set(SOFA-EXTERNAL_LIBRARY_DIR ${SOFA-EXTERNAL_LIBRARY_DIR} CACHE PATH "Library path for pre-compiled dependencies outside of the Sofa directory")

# extlibs
##CGoGN
sofa_option(SOFA-EXTERNAL_CGOGN_PATH PATH "${SOFA_EXTLIBS_DIR}/CGoGN" "Path to the CGoGN library")

## eigen
sofa_option(SOFA-EXTERNAL_EIGEN_PATH PATH "${SOFA_EXTLIBS_DIR}/eigen-3.2.1" "Path to the eigen header-only library")

## lua
sofa_option(SOFA-EXTERNAL_LUA_PATH PATH "${SOFA_EXTLIBS_DIR}/lua" "Path to the Lua library sources")

## metis
sofa_option(SOFA-EXTERNAL_METIS_PATH PATH "${SOFA_EXTLIBS_DIR}/metis-5.1.0" "Path to the metis library sources")

## verdandi
sofa_option(SOFA-EXTERNAL_VERDANDI_PATH PATH "${SOFA_EXTLIBS_DIR}/verdandi-1.5" "Path to the Verdandi library sources")

## Qt
set(QTDIR "$ENV{QTDIR}")
if(WIN32 AND QTDIR STREQUAL "")
    set(QTDIR "${SOFA_TOOLS_DIR}/qt4win")
endif()
if(NOT QTDIR STREQUAL "")
    if(WIN32)
        file(TO_CMAKE_PATH "${QTDIR}" QTDIR) # GLOB will fail with pathes containing backslashes.
    endif()
    file(GLOB QTDIR "${QTDIR}") # check if the QTDIR contains a correct path
endif()

### the ENV{QTDIR} MUST BE DEFINED in order to find Qt (giving a path in find_package does not work)
sofa_option(SOFA-EXTERNAL_QT_PATH PATH "${QTDIR}" "Qt dir path")
sofa_option(SOFA-EXTERNAL_QT5_PATH PATH "${QTDIR}" "Qt5 dir path")
list(APPEND compilerDefines SOFA_QT4)

## boost
set(MINIBOOST_PATH "${SOFA_EXTLIBS_DIR}/miniBoost")
sofa_option(SOFA-EXTERNAL_BOOST_PATH PATH "${MINIBOOST_PATH}" "Boost path, set to blank if you want to use the boost installed on your system or set a path if you want to use a compiled boost")
if(SOFA-EXTERNAL_BOOST_PATH STREQUAL "${MINIBOOST_PATH}")
    unset(SOFA-EXTERNAL_BOOST CACHE)
else()
    set(SOFA-EXTERNAL_BOOST 1 CACHE INTERNAL "Use the system / user compiled boost library instead of miniBoost" FORCE)
    list(APPEND compilerDefines SOFA_HAVE_BOOST)
endif()


## geometric tools
sofa_option(SOFA-EXTERNAL_GEOMETRIC_TOOLS_PATH PATH "" "Path to Geometric tools folder containing the cmake project")
if(SOFA-EXTERNAL_GEOMETRIC_TOOLS_PATH) # since the lib could be in the system path we cannot check the path with the EXISTS function
    set(SOFA-EXTERNAL_GEOMETRIC_TOOLS 1 CACHE INTERNAL "Build and use geometric tools" FORCE)
    # list(APPEND compilerDefines SOFA_HAVE_GEOMETRIC_TOOLS) # set this compiler defininition using RegisterProjectDependencies (to avoid the need of rebuilding everything if you change this option)
else()
    unset(SOFA-EXTERNAL_GEOMETRIC_TOOLS CACHE)
endif()

## tinyxml
set(TINYXML_PATH "${SOFA_EXTLIBS_DIR}/tinyxml")
sofa_option(SOFA-EXTERNAL_TINYXML_PATH PATH "${TINYXML_PATH}" "")
if(SOFA-EXTERNAL_TINYXML_PATH STREQUAL "${TINYXML_PATH}")
    unset(SOFA-EXTERNAL_TINYXML CACHE)
else()
    set(SOFA-EXTERNAL_TINYXML 1 CACHE INTERNAL "Use the system / user compiled tinyxml library instead of miniBoost" FORCE)
endif()

#sofa_option(SOFA-EXTERNAL_TINYXML_INCLUDE_DIR PATH "" "For pre-compiled tinyxml: library where headers are available")
#sofa_option(SOFA-EXTERNAL_TINYXML_LIBRARY PATH "" "For pre-compiled tinyxml: release-mode library name")
#sofa_option(SOFA-EXTERNAL_TINYXML_DEBUG_LIBRARY PATH "" "For pre-compiled tinyxml: debug-mode library name")
#mark_as_advanced(SOFA-EXTERNAL_TINYXML_INCLUDE_DIR)
#mark_as_advanced(SOFA-EXTERNAL_TINYXML_LIBRARY)
#mark_as_advanced(SOFA-EXTERNAL_TINYXML_DEBUG_LIBRARY)

## CGoGN
sofa_option(SOFA-EXTERNAL_CGOGN BOOL OFF "Use the CGoGN library")

## zlib
sofa_option(SOFA-EXTERNAL_ZLIB BOOL ON "Use the ZLib library")

## libpng
sofa_option(SOFA-EXTERNAL_PNG BOOL ON "Use the LibPNG library")

## freeglut
sofa_option(SOFA-EXTERNAL_FREEGLUT BOOL OFF "Use the FreeGLUT library (instead of regular GLUT)")

## glew
sofa_option(SOFA-EXTERNAL_GLEW BOOL ON "Use the GLEW library")

## ffmpeg
set(FFMPEG_PATH "")
sofa_option(SOFA-EXTERNAL_FFMPEG BOOL OFF "Use the FFMPEG library")
sofa_option(SOFA-EXTERNAL_FFMPEG_PATH PATH "${FFMPEG_PATH}" "")
if(SOFA-EXTERNAL_FFMPEG)
    list(APPEND compilerDefines SOFA_HAVE_FFMPEG)
endif()

## METIS
sofa_option(SOFA-EXTERNAL_METIS BOOL OFF "Use Metis")

## VERDANDI
sofa_option(SOFA-EXTERNAL_VERDANDI BOOL OFF "Use Verdandi")

## LUA
sofa_option(SOFA-EXTERNAL_LUA BOOL OFF "Use Lua")

## CSPARSE
set(CSPARSE_PATH "${SOFA_EXTLIBS_DIR}/csparse")
sofa_option(SOFA-EXTERNAL_CSPARSE BOOL OFF "Use CSparse")
sofa_option(SOFA-EXTERNAL_CSPARSE_PATH PATH "${CSPARSE_PATH}" "")

## FLOWVR
set(FLOWVR_PATH "${SOFA_EXTLIBS_DIR}/miniFlowVR")
sofa_option(SOFA-EXTERNAL_FLOWVR_PATH PATH "${FLOWVR_PATH}" "")
if(NOT SOFA-EXTERNAL_FLOWVR_PATH STREQUAL ${FLOWVR_PATH})
    set(SOFA-EXTERNAL_FLOWVR 1 CACHE INTERNAL "Use FlowVR" FORCE)
else()
    unset(SOFA-EXTERNAL_FLOWVR CACHE)
endif()

## OPENCASCADE
set(OPENCASCADE_PATH "")
sofa_option(SOFA-EXTERNAL_OPENCASCADE_PATH PATH "${OPENCASCADE_PATH}" "OpenCascade Path")
file(TO_CMAKE_PATH "${SOFA-EXTERNAL_OPENCASCADE_PATH}" OPENCASCADE_PATH)
set(SOFA-EXTERNAL_OPENCASCADE_PATH "${OPENCASCADE_PATH}" CACHE PATH "OpenCascade Path" FORCE)

# Miscellaneous features

sofa_option(SOFA-MISC_CMAKE_VERBOSE BOOL OFF "Print more information during the cmake configuration step.")

sofa_option(SOFA-MISC_SMP BOOL OFF "Use SMP")

sofa_option(SOFA-MISC_DOXYGEN BOOL OFF "Create targets to generate documentation with doxygen.")
sofa_option(SOFA-MISC_DOXYGEN_COMPONENT_LIST BOOL OFF "When generating the documentation of SOFA, generate a page with the list of the available components. This is a separate option because it requires compiling SOFA.")
if(SOFA-MISC_DOXYGEN_COMPONENT_LIST)
    set(SOFA-LIB_COMPONENT_COMPONENT_MAIN ON CACHE BOOL "" FORCE)
endif()

## no opengl
sofa_option(SOFA-MISC_NO_OPENGL BOOL OFF "Disable OpenGL")
if(SOFA-MISC_NO_OPENGL)
    list(APPEND compilerDefines SOFA_NO_OPENGL)
    set(SOFA_VISUAL_LIB SofaBaseVisual)
else()
    set(SOFA_VISUAL_LIB SofaOpenglVisual)
endif()

## SOFA_NO_UPDATE_BBOX
sofa_option(SOFA-MISC_NO_UPDATE_BBOX BOOL OFF "No BBOX update")
if(SOFA-MISC_NO_UPDATE_BBOX)
    list(APPEND compilerDefines SOFA_NO_UPDATE_BBOX)
endif()

sofa_option(SOFA-MISC_DEV BOOL OFF "Compile SOFA_DEV code")
if(SOFA-MISC_DEV)
    list(APPEND compilerDefines SOFA_DEV)
endif()

sofa_option(SOFA-MISC_DUMP_VISITOR_INFO BOOL OFF "Compile with performance analysis")
if(SOFA-MISC_DUMP_VISITOR_INFO)
    list(APPEND compilerDefines SOFA_DUMP_VISITOR_INFO)
endif()


## tutorials
if(PS3)
    set(tutorial_default OFF)
else()
    set(tutorial_default ON)
endif()
# The first ones are disabled by default because they depend on the SceneCreator plugin - Marc
sofa_option(SOFA-TUTORIAL_CHAIN_HYBRID BOOL OFF "Build the \"Chain Hybrid\" tutorial")
sofa_option(SOFA-TUTORIAL_HOUSE_OF_CARDS BOOL OFF "Build the \"House of Cards\" tutorial")
sofa_option(SOFA-TUTORIAL_COMPOSITE_OBJECT BOOL ${tutorial_default} "Build the \"Composite Object\" tutorial")
sofa_option(SOFA-TUTORIAL_MIXED_PENDULUM BOOL ${tutorial_default} "Build the \"Mixed Pendulum\" tutorial")
sofa_option(SOFA-TUTORIAL_ONE_PARTICLE BOOL ${tutorial_default} "Build the \"One Particle\" tutorial")
#sofa_option(SOFA-TUTORIAL_ONE_PARTICLE_WITH_SOFA_TYPEDEFS BOOL ${tutorial_default} "Build the \"One Particle with sofa typedefs\" tutorial")
sofa_option(SOFA-TUTORIAL_ONE_TETRAHEDRON BOOL ${tutorial_default} "Build the \"One Tetrahedron\" tutorial")
#sofa_option(SOFA-TUTORIAL_ANATOMY_MODELLING BOOL ${tutorial_default} "Build the \"Anatomy Modelling\" tutorial")

## framework
sofa_option(SOFA-LIB_CORE BOOL ON "")
sofa_option(SOFA-LIB_DEFAULTTYPE BOOL ON "")
sofa_option(SOFA-LIB_HELPER BOOL ON "")

# component
sofa_option(SOFA-LIB_COMPONENT_BASE_ANIMATION_LOOP BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_BASE_COLLISION BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_BASE_LINEAR_SOLVER BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_BASE_MECHANICS BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_BASE_TOPOLOGY BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_BASE_VISUAL BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_BOUNDARY_CONDITION BOOL ON "")

sofa_option(SOFA-LIB_COMPONENT_COMPONENT_ADVANCED BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_COMPONENT_COMMON BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_COMPONENT_GENERAL BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_COMPONENT_MISC BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_COMPONENT_BASE BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_COMPONENT_MAIN BOOL ON "")

sofa_option(SOFA-LIB_COMPONENT_CONSTRAINT BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_DEFORMABLE BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_DENSE_SOLVER BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_EIGEN2_SOLVER BOOL ON "")

sofa_option(SOFA-LIB_COMPONENT_ENGINE BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_EULERIAN_FLUID BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_EXPLICIT_ODE_SOLVER BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_EXPORTER BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_GRAPH_COMPONENT BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_HAPTICS BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_IMPLICIT_ODE_SOLVER BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_LOADER BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_MESH_COLLISION BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_MISC BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_MISC_COLLISION BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_MISC_ENGINE BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_MISC_FEM BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_MISC_FORCEFIELD BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_MISC_MAPPING BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_MISC_SOLVER BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_MISC_TOPOLOGY BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_NON_UNIFORM_FEM BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_OBJECT_INTERACTION BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_OPENGL_VISUAL BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_PARDISO_SOLVER BOOL OFF "")
sofa_option(SOFA-LIB_COMPONENT_RIGID BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_SIMPLE_FEM BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_SPARSE_SOLVER BOOL OFF "")

sofa_option(SOFA-LIB_COMPONENT_PRECONDITIONER BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_SPH_FLUID BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_TAUCS_SOLVER BOOL OFF "")
sofa_option(SOFA-LIB_COMPONENT_TOPOLOGY_MAPPING BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_USER_INTERACTION BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_VALIDATION BOOL ON "")
sofa_option(SOFA-LIB_COMPONENT_VOLUMETRIC_DATA BOOL ON "")

# i don't know if we mark default components as advanced or not
# it would enhance readability but thinking to look for
# advanced options is not really obvious
if(false)
    mark_as_advanced(SOFA-LIB_COMPONENT_BASE_ANIMATION_LOOP)
    mark_as_advanced(SOFA-LIB_COMPONENT_BASE_COLLISION)
    mark_as_advanced(SOFA-LIB_COMPONENT_BASE_LINEAR_SOLVER)
    mark_as_advanced(SOFA-LIB_COMPONENT_BASE_MECHANICS)
    mark_as_advanced(SOFA-LIB_COMPONENT_BASE_TOPOLOGY)
    mark_as_advanced(SOFA-LIB_COMPONENT_BASE_VISUAL)
    mark_as_advanced(SOFA-LIB_COMPONENT_BOUNDARY_CONDITION)

    mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_ADVANCED)
    mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_COMMON)
    mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_GENERAL)
    mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_MISC)
    mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_BASE)
    mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_MAIN)

    mark_as_advanced(SOFA-LIB_COMPONENT_CONSTRAINT)
    mark_as_advanced(SOFA-LIB_COMPONENT_DEFORMABLE)
    mark_as_advanced(SOFA-LIB_COMPONENT_DENSE_SOLVER)
    mark_as_advanced(SOFA-LIB_COMPONENT_EIGEN2_SOLVER)

    mark_as_advanced(SOFA-LIB_COMPONENT_ENGINE)
    mark_as_advanced(SOFA-LIB_COMPONENT_EULERIAN_FLUID)
    mark_as_advanced(SOFA-LIB_COMPONENT_EXPLICIT_ODE_SOLVER)
    mark_as_advanced(SOFA-LIB_COMPONENT_EXPORTER)
    mark_as_advanced(SOFA-LIB_COMPONENT_GRAPH_COMPONENT)
    mark_as_advanced(SOFA-LIB_COMPONENT_HAPTICS)
    mark_as_advanced(SOFA-LIB_COMPONENT_IMPLICIT_ODE_SOLVER)
    mark_as_advanced(SOFA-LIB_COMPONENT_LOADER)
    mark_as_advanced(SOFA-LIB_COMPONENT_MESH_COLLISION)
    mark_as_advanced(SOFA-LIB_COMPONENT_MISC)
    mark_as_advanced(SOFA-LIB_COMPONENT_MISC_COLLISION)
    mark_as_advanced(SOFA-LIB_COMPONENT_MISC_ENGINE)
    mark_as_advanced(SOFA-LIB_COMPONENT_MISC_FEM)
    mark_as_advanced(SOFA-LIB_COMPONENT_MISC_FORCEFIELD)
    mark_as_advanced(SOFA-LIB_COMPONENT_MISC_MAPPING)
    mark_as_advanced(SOFA-LIB_COMPONENT_MISC_SOLVER)
    mark_as_advanced(SOFA-LIB_COMPONENT_MISC_TOPOLOGY)
    mark_as_advanced(SOFA-LIB_COMPONENT_NON_UNIFORM_FEM)
    mark_as_advanced(SOFA-LIB_COMPONENT_OBJECT_INTERACTION)
    mark_as_advanced(SOFA-LIB_COMPONENT_OPENGL_VISUAL)
    #mark_as_advanced(SOFA-LIB_COMPONENT_PARDISO_SOLVER)
    mark_as_advanced(SOFA-LIB_COMPONENT_RIGID)
    mark_as_advanced(SOFA-LIB_COMPONENT_SIMPLE_FEM)
    #mark_as_advanced(SOFA-LIB_COMPONENT_SPARSE_SOLVER)

    mark_as_advanced(SOFA-LIB_PRECONDITIONER)
    mark_as_advanced(SOFA-LIB_SPH_FLUID)
    #mark_as_advanced(SOFA-LIB_TAUCS_SOLVER)
    mark_as_advanced(SOFA-LIB_TOPOLOGY_MAPPING)
    #mark_as_advanced(SOFA-LIB_USER_INTERACTION)
    mark_as_advanced(SOFA-LIB_VALIDATION)
    mark_as_advanced(SOFA-LIB_VOLUMETRIC_DATA)
endif()

# simulation
sofa_option(SOFA-LIB_SIMULATION_GRAPH_DAG BOOL ON "Directed acyclic graph")

# Qt GUI
if (NOT SOFA-MISC_NO_OPENGL AND NOT SOFA-MISC_NO_QT AND NOT PS3)
    sofa_option(SOFA-LIB_GUI_QT BOOL ON "Use QT interface")
    sofa_option(SOFA-LIB_GUI_QTVIEWER BOOL ON "Use QT Viewer")
    sofa_option(SOFA-LIB_GUI_QGLVIEWER BOOL OFF "Use QGLViewer")
    sofa_option(SOFA-LIB_GUI_INTERACTION BOOL OFF "Enable interaction mode")
else()
    unset(SOFA-LIB_GUI_QT CACHE)
    unset(SOFA-LIB_GUI_QTVIEWER CACHE)
    unset(SOFA-LIB_GUI_QGLVIEWER CACHE)
    unset(SOFA-LIB_GUI_INTERACTION CACHE)
endif()

# GLUT GUI
if (NOT SOFA-MISC_NO_OPENGL AND NOT PS3)
    sofa_option(SOFA-LIB_GUI_GLUT BOOL ON "Use GLUT interface")
else()
    unset(SOFA-LIB_GUI_GLUT CACHE)
endif()

# unit tests
sofa_option(SOFA-MISC_TESTS BOOL OFF "Build and use all the unit tests, including the tests of the activated plugins")
if(SOFA-MISC_TESTS)
    sofa_option(SOFA-MISC_BUILD_GTEST BOOL ON "Build google test framework")
endif()

# use external template
sofa_option(SOFA-MISC_EXTERN_TEMPLATE BOOL ON "Use extern template")
if(NOT SOFA-MISC_EXTERN_TEMPLATE)
    list(APPEND compilerDefines SOFA_NO_EXTERN_TEMPLATE)
endif()

# float / double or both
sofa_option(SOFA-MISC_USE_FLOAT BOOL OFF "Use single precision floating point (float)")
sofa_option(SOFA-MISC_USE_DOUBLE BOOL OFF "Use double precision floating point (double)")
if(SOFA-MISC_USE_FLOAT AND NOT SOFA-MISC_USE_DOUBLE)
    list(APPEND compilerDefines SOFA_FLOAT)
elseif(SOFA-MISC_USE_DOUBLE AND NOT SOFA-MISC_USE_FLOAT)
    list(APPEND compilerDefines SOFA_DOUBLE)
elseif(SOFA-MISC_USE_DOUBLE AND SOFA-MISC_USE_FLOAT)
    message(FATAL_ERROR "You can't enable both SOFA-MISC_USE_FLOAT and SOFA-MISC_USE_DOUBLE")
endif()

# use OpenMP multithreading
sofa_option(SOFA-MISC_OPENMP BOOL OFF "Use OpenMP multithreading")
if(SOFA-MISC_OPENMP )
    sofa_option(OpenMP_DEFAULT_NUM_THREADS_EIGEN_SPARSE_DENSE_PRODUCT INT 1 "Default number of threads for Eigen Sparse x Dense Matrix product (this number must not be too large for Matrix-Vector products where the MT overhead is hard to compensate)")
    list(APPEND compilerDefines OMP_DEFAULT_NUM_THREADS_EIGEN_SPARSE_DENSE_PRODUCT=${OpenMP_DEFAULT_NUM_THREADS_EIGEN_SPARSE_DENSE_PRODUCT})
endif()


# OS-specific
if(XBOX)
    if(SOFA-EXTERNAL_BOOST)
        # we use SOFA-EXTERNAL_BOOST_PATH but don't have the full boost and thus can't compile the code this normally enables.
        unset(SOFA-EXTERNAL_BOOST CACHE)
        list(REMOVE_ITEM compilerDefines SOFA_HAVE_BOOST)
    endif()

    # eigen - cpuid identification code does not exist on the platform, it's cleaner to disable it here.
    list(APPEND compilerDefines EIGEN_NO_CPUID)
endif()

if(MSVC)
    sofa_option(SOFA-MISC_VECTORIZE BOOL OFF "Enable the use of SSE2 instructions by the compiler. (MSVC only)")
endif()

sofa_option(SOFA-MISC_NO_EXCEPTIONS BOOL OFF "Disable the use of exceptions")
sofa_option(SOFA-MISC_STATIC_LINK_BOOST BOOL OFF "Use static library version of boost")

##############
#### CUDA ####
##############
sofa_option(SOFA-CUDA_VERBOSE_PTXAS BOOL OFF "SOFA-CUDA_VERBOSE_PTXAS")
if(SOFA-CUDA_VERBOSE_PTXAS)
    set(VERBOSE_PTXAS --ptxas-options=-v)
endif()

#Option to activate double-precision support in CUDA (requires GT200+ GPU and -arch sm_13 flag)
sofa_option(SOFA-CUDA_DOUBLE BOOL OFF "SOFA-CUDA_DOUBLE")
if(SOFA-CUDA_DOUBLE)
    add_definitions("-DSOFA_GPU_CUDA_DOUBLE")
    AddCompilerDefinitions("SOFA_GPU_CUDA_DOUBLE")
endif()

#Option to use IEEE 754-compliant floating point operations
sofa_option(SOFA-CUDA_PRECISE BOOL OFF "SOFA-CUDA_PRECISE")
if(SOFA-CUDA_PRECISE)
    add_definitions("-DSOFA_GPU_CUDA_PRECISE")
    AddCompilerDefinitions("SOFA_GPU_CUDA_PRECISE")
endif()

# Option to get double-precision for sqrt/div...
# (requires compute capability >= 2 and CUDA_VERSION > 3.0)
# (with SOFA_GPU_CUDA_PRECISE and SOFA_GPU_CUDA_DOUBLE you get IEEE 754-compliant floating point
#  operations for addition and multiplication only)
sofa_option(SOFA-CUDA_DOUBLE_PRECISE BOOL OFF "SOFA-CUDA_DOUBLE_PRECISE")
if(SOFA-CUDA_DOUBLE_PRECISE)
    add_definitions("-DSOFA_GPU_CUDA_DOUBLE_PRECISE")
    AddCompilerDefinitions("SOFA_GPU_CUDA_DOUBLE_PRECISE")
endif()

# Option to activate cublas support in CUDA (requires SOFA_GPU_CUDA_DOUBLE)
sofa_option(SOFA-CUDA_CUBLAS BOOL OFF "SOFA-CUDA_CUBLAS")
if(SOFA-CUDA_CUBLAS)
    add_definitions("-DSOFA_GPU_CUBLAS")
    AddCompilerDefinitions("SOFA_GPU_CUBLAS")
endif()

# Option to activate CUDPP (for RadixSort)
sofa_option(SOFA-CUDA_CUDPP BOOL OFF "SOFA-CUDA_CUDPP")
if(SOFA-CUDA_CUDPP)
    add_definitions("-DSOFA_GPU_CUDPP")
    AddCompilerDefinitions("SOFA_GPU_CUDPP")
    if(SOFA-EXTERNAL_CUDPP)
        AddLinkerDependencies(cudpp)
    endif()
endif()

# Option to activate THRUST (for RadixSort)
# Note: THRUST is included in CUDA SDK 4.0+, it is recommended to use it if available
sofa_option(SOFA-CUDA_THRUST BOOL ON "SOFA-CUDA_THRUST")
if(SOFA-CUDA_THRUST)
    add_definitions("-DSOFA_GPU_THRUST")
    AddCompilerDefinitions("SOFA_GPU_THRUST")
    if(SOFA-EXTERNAL_THRUST)
        AddLinkerDependencies(thrust)
    endif()
endif()

# GPU architecture for which CUDA code will be compiled.
sofa_option(SOFA-CUDA_SM STRING "20" "GPU architecture; it will translate to the following option for nvcc: -arch sm_<value>")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch sm_${SOFA-CUDA_SM})
if(NOT WIN32)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fPIC)
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler)
endif()

# TODO   activate it automatically
sofa_option(SOFA-CUDA_GREATER_THAN_GCC44 BOOL OFF "SOFA-CUDA_GREATER_THAN_GCC44")
if(SOFA-CUDA_GREATER_THAN_GCC44)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--compiler-options -fno-inline)
endif()

# nvcc uses a "host code compiler" to compile CPU code, specified by CUDA_HOST_COMPILER.
# With some versions of CMake, CUDA_HOST_COMPILER defaults to CMAKE_C_COMPILER,
# but few host compilers are actually supported. Workarounds should go here.
if (${CUDA_HOST_COMPILER} MATCHES "ccache$")
    set(CUDA_HOST_COMPILER "gcc" CACHE STRING "Host side compiler used by NVCC" FORCE)
endif()

## in debug mode, enforce cuda to compile host code in debug (the same could be done for device code with -G)
set(CUDA_NVCC_FLAGS_DEBUG "-g" CACHE STRING "Semi-colon delimit multiple arguments" FORCE)
## in release mode, enforce optimizations for host code
set(CUDA_NVCC_FLAGS_RELEASE "-DNDEBUG" CACHE STRING "Semi-colon delimit multiple arguments" FORCE)


# plugins (auto-search)
set(SOFA_PROJECT_FOLDER "SofaPlugin")
RetrieveDependencies("${SOFA_APPLICATIONS_PLUGINS_DIR}" "SOFA-PLUGIN_" "Enable plugin" "SOFA_HAVE_PLUGIN_" RECURSIVE)

# dev-plugins (auto-search)
set(SOFA_PROJECT_FOLDER "SofaDevPlugin")
RetrieveDependencies("${SOFA_APPLICATIONS_DEV_PLUGINS_DIR}" "SOFA-DEVPLUGIN_" "Enable dev plugin" "SOFA_HAVE_DEVPLUGIN_" RECURSIVE)

# projects (auto-search)
set(SOFA_PROJECT_FOLDER "SofaApplication")
RetrieveDependencies("${SOFA_APPLICATIONS_PROJECTS_DIR}" "SOFA-APPLICATION_" "Enable application" "SOFA_HAVE_APPLICATION_")

# dev-projects (auto-search)
set(SOFA_PROJECT_FOLDER "SofaDevApplication")
RetrieveDependencies("${SOFA_APPLICATIONS_DEV_PROJECTS_DIR}" "SOFA-DEVAPPLICATION_" "Enable dev application" "SOFA_HAVE_DEVAPPLICATION_")

set(SOFA_PROJECT_FOLDER "")
# configurable paths to use pre-compiled dependencies outside of the Sofa directory

set(GLOBAL_COMPILER_DEFINES ${GLOBAL_COMPILER_DEFINES} ${compilerDefines} CACHE INTERNAL "Global Compiler Defines" FORCE)

