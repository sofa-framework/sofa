# This Finder allow to find SOFA files for build a project with it.
# In order to use this cmake module, you have to call the find_package(SOFA) command in your CMakeLists.txt
#
# This module defines for use :
# SOFA_LIBRARIES wich contain all LIBRARIES variables in absolute path of Sofa
# SOFA_FOUND, if false, Sofa not found
# SOFA_INCLUDE_DIRS, where to find all the headers
#
# $SOFA_DIR is the entry point of this find package.
# $SOFA_DIR can be set by an environment variable path or in command line
#
# Header files are presumed to be included like
# #include <sofa/defaulttype/Vec.h>
# #include <sofa/defaulttype/Quat.h>
#
# To maintain this script, you just have to :
#  * update if necessary the header file and path to search it to find the framwork/moules/application dirs
#  * update if necessary the include dirs for extlibs
#  * update if necessary name and cmake name of libraries in the corresponding section
#  * update if necessary the paths to search the libs to the find_lib macro
#
# If you have some problem to include this cmake files in your CMake project, be sur you appended CMAKE_MODULE_PATH to the cmake dir :
# list(APPEND CMAKE_MODULE_PATH "${SOFA_DIR}/cmake")
#
## TODO :
# In order to have a more flexible FindSOFA.cmake, may be we can allow to use some additional CMAKE variables as input of this cmake module
# to find other specific SOFA lib/include...
#
# Created by Jean-Christophe Lombardo and Jerome Esnault.



## ###########################################################################################
## VERBOSITY SETTINGS
## ###########################################################################################
option(VERBOSE_SOFA "Do you want cmake to be verbose during project searching?" false)
message(STATUS "VERBOSE_SOFA = ${VERBOSE_SOFA}")


## ###########################################################################################
## DEFINE SOFA_DIR root path
## ###########################################################################################
if (NOT SOFA_DIR)
    set(SOFA_DIR "$ENV{SOFA_DIR}" CACHE PATH "Sofa root directory")
endif (NOT SOFA_DIR)
## Make sur the provided path is a cmake style path with unix /
file(TO_CMAKE_PATH ${SOFA_DIR} SOFA_DIR)

if(SOFA_DIR AND VERBOSE_SOFA)
    message( STATUS "\nSOFA_DIR = ${SOFA_DIR}" )
    message( STATUS "" )
endif(SOFA_DIR AND VERBOSE_SOFA)

if(NOT SOFA_DIR)
    message( "" )
    message( "Set SOFA_DIR because it : ${SOFA_DIR}" )
    message( "" )
endif(NOT SOFA_DIR)

## Already in cache, be silent
IF (SOFA_INCLUDE_DIRS)
    set(SOFA_FIND_QUIETLY TRUE)
    if (VERBOSE_SOFA)
        message( STATUS "SOFA_INCLUDE_DIRS already in cache : ${SOFA_INCLUDE_DIRS}" )
    endif (VERBOSE_SOFA)
ENDIF (SOFA_INCLUDE_DIRS)


## ###########################################################################################
## FIND INCLUDE SOFA DIRS
## ###########################################################################################
if (VERBOSE_SOFA)
    message(STATUS "----SOFA INCLUDE DIRS LIST : ")
endif (VERBOSE_SOFA)


## ===================== SOFA_INCLUDE_FRAMEWORK_DIR
find_path(SOFA_INCLUDE_FRAMEWORK_DIR 
   NAME
        sofa/core/core.h #use a file .h looks like important file to find the path directory
   PATHS
        ${SOFA_DIR}/framework
        ## comment to allow CMake to search in system environment variables and
        ## in cmake cache/environment/defined variables
        #NO_DEFAULT_PATH
)
if (VERBOSE_SOFA)
    message( STATUS "SOFA_INCLUDE_FRAMEWORK_DIR = ${SOFA_INCLUDE_FRAMEWORK_DIR}" )
endif (VERBOSE_SOFA)


## ===================== SOFA_INCLUDE_MODULES_DIR
find_path(SOFA_INCLUDE_MODULES_DIR 
   NAME
        sofa/component/init.h #use a file .h looks like important file to find the path directory
   PATHS
        ${SOFA_DIR}/modules
        ## comment to allow CMake to search in system environment variables
        ## and in cmake cache/environment/defined variables
        #NO_DEFAULT_PATH
)
if (VERBOSE_SOFA)
    message( STATUS "SOFA_INCLUDE_MODULES_DIR = ${SOFA_INCLUDE_MODULES_DIR}" )
endif (VERBOSE_SOFA)


## ===================== SOFA_INCLUDE_APPLICATIONS_DIR
find_path(SOFA_INCLUDE_APPLICATIONS_DIR
   NAME
        sofa/gui/SofaGUI.h #use a file .h looks like important file to find the path directory
   PATHS
        ${SOFA_DIR}/applications
        ## comment to allow CMake to search in system environment variables and
        ## in cmake cache/environment/defined variables
        #NO_DEFAULT_PATH
)
if (VERBOSE_SOFA)
    message( STATUS "SOFA_INCLUDE_APPLICATIONS_DIR = ${SOFA_INCLUDE_APPLICATIONS_DIR}" )
endif (VERBOSE_SOFA)


## ===================== SOFA_INCLUDE_OTHER_DIRS
set(SOFA_INCLUDE_EXTLIBS "${SOFA_DIR}/extlibs")
list(APPEND SOFA_INCLUDE_OTHER_DIRS
    ${SOFA_INCLUDE_EXTLIBS}
    ${SOFA_INCLUDE_EXTLIBS}/ARTrack
    ${SOFA_INCLUDE_EXTLIBS}/CImg
    ${SOFA_INCLUDE_EXTLIBS}/colladadom
    ${SOFA_INCLUDE_EXTLIBS}/csparse
    ${SOFA_INCLUDE_EXTLIBS}/cudpp
    ${SOFA_INCLUDE_EXTLIBS}/eigen-3.1.1
    ${SOFA_INCLUDE_EXTLIBS}/ffmpeg
    ${SOFA_INCLUDE_EXTLIBS}/fftpack
    ${SOFA_INCLUDE_EXTLIBS}/fishpack
    ${SOFA_INCLUDE_EXTLIBS}/indexedmap
    ${SOFA_INCLUDE_EXTLIBS}/libQGLViewer-2.3.3
    ${SOFA_INCLUDE_EXTLIBS}/LML
    ${SOFA_INCLUDE_EXTLIBS}/metis
    ${SOFA_INCLUDE_EXTLIBS}/miniBoost
    ${SOFA_INCLUDE_EXTLIBS}/miniFlowVR/include
    ${SOFA_INCLUDE_EXTLIBS}/MKL
    ${SOFA_INCLUDE_EXTLIBS}/muparser
    ${SOFA_INCLUDE_EXTLIBS}/newmat
    ${SOFA_INCLUDE_EXTLIBS}/PML
    #${SOFA_INCLUDE_EXTLIBS}/qwt-5.2.0/src
    ${SOFA_INCLUDE_EXTLIBS}/qwt-6.0.1/src
    ${SOFA_INCLUDE_EXTLIBS}/self-ccd-1.0
#    ${SOFA_INCLUDE_EXTLIBS}/SLC
    ${SOFA_INCLUDE_EXTLIBS}/taucs
    ${SOFA_INCLUDE_EXTLIBS}/taucs_mt
    ${SOFA_INCLUDE_EXTLIBS}/taucs-svn
    ${SOFA_INCLUDE_EXTLIBS}/tinyxml
    ${SOFA_INCLUDE_EXTLIBS}/VRPN
    ${SOFA_INCLUDE_EXTLIBS}/wiiuse
)
if(MSVC)
    list(APPEND SOFA_INCLUDE_OTHER_DIRS ${SOFA_DIR}/include)
endif(MSVC)


list(LENGTH SOFA_INCLUDE_OTHER_DIRS otherIncludeCount)
math(EXPR count "${otherIncludeCount}-1")
foreach( i RANGE 0 ${count})
    list(GET SOFA_INCLUDE_OTHER_DIRS ${i} VALUE)
    if(EXISTS ${VALUE})
        if(VERBOSE_SOFA)
            message(STATUS "SOFA_INCLUDE_OTHER_DIRS_${i} = ${VALUE}")
        endif()
        set(SOFA_INCLUDE_OTHER_DIRS_${i} ${VALUE} CACHE STRING "other include dirs needed by sofa")
    else()
        message(WARNING 
                "It seems the SOFA_INCLUDE_OTHER_DIRS_${i} doest not exist : ${VALUE}.
                Please check it in FindSOFA.cmake")
    endif()
endforeach()
if(VERBOSE_SOFA)
    message( STATUS "----set this variable : SOFA_INCLUDE_DIRS with all include directories found" )
    message(STATUS "")
endif()


set(SOFA_INCLUDE_DIRS
    ${SOFA_INCLUDE_FRAMEWORK_DIR}
    ${SOFA_INCLUDE_MODULES_DIR}
    ${SOFA_INCLUDE_APPLICATIONS_DIR}
    ${SOFA_INCLUDE_OTHER_DIRS}
)

## ###########################################################################################
## FIND LIBRARIES
##
## SOFA group the components by functionality and maturity state.
## 50 new groups are contained in 5 different categories:
## BASE, COMMON, GENERAL, ADVANCED and MISC.
##
## 1- collect all library name to search in the SOFA_LIBS_NAME list splitted into 5 parts
##    * the SOFA LIBS BASE LIST
##    * the SOFA COMMON LIST
##    * THE SOFA GENERAL LIST
##    * THE SOFA ADVANCED LIST
##    * THE SOFA MISC LIST
## 2- for each library :
##    * get it CMAKE_SOFA_LIB_NAME and it associate REAL_SOFA_LIB_NAME
##    * find library and set SOFA_LIBRARIES
## ###########################################################################################
## Put the name of the library SOFA CORE to search and put it associate CMakeName
list(APPEND SOFA_LIB_NAME
    sofagui             SOFA_LIB_GUI            
    sofatree            SOFA_LIB_TREE            
    sofacore            SOFA_LIB_CORE              
    sofaguiqt           SOFA_LIB_GUI_QT         
    sofahelper          SOFA_LIB_HELPER         
    sofaguiglut         SOFA_LIB_GUI_GLUT             
    sofaguimain         SOFA_LIB_GUI_MAIN           
    sofamodeler         SOFA_LIB_MODELER         
    sofasimulation      SOFA_LIB_SIMULATION      
    sofaobjectcreator   SOFA_OBJECT_CREATOR     
    sofadefaulttype     SOFA_LIB_DEFAULT_TYPE   
    sofagraph           SOFA_LIB_GRAPH
)
list(LENGTH SOFA_LIB_NAME sofaLibList)
math(EXPR passToComponent "${sofaLibList}/2")


## Put the name of the library SOFA COMPONENT to search and put it associate CMakeName
list(APPEND SOFA_LIB_COMPONENT_NAME
    sofa_component              SOFA_LIB_COMPONENT                  
    sofa_component_dev          SOFA_LIB_COMPONENT_DEV              
    sofa_component_base         SOFA_LIB_COMPONENT_BASE             
    sofa_component_misc         SOFA_LIB_COMPONENT_MISC             
    sofa_component_common       SOFA_LIB_COMPONENT_COMMON           
    sofa_component_general      SOFA_LIB_COMPONENT_GENERAL          
    sofa_component_misc_dev     SOFA_LIB_COMPONENT_MISC_DEV                      
    sofa_component_advanced     SOFA_LIB_COMPONENT_ADVANCED         
    sofa_component_advanced_dev SOFA_LIB_COMPONENT_ADVANCED_DEV         
)
list(LENGTH SOFA_LIB_COMPONENT_NAME sofaLibComponentList)
math(EXPR passToBase "${sofaLibComponentList}/2+${passToComponent}")


## Put the name of the library SOFA BASE to search and put it associate CMakeName
list(APPEND SOFA_LIB_BASE_NAME
    sofa_base_animation_loop    SOFA_LIB_BASE_ANIMATION_LOOP    
    sofa_base_collision         SOFA_LIB_BASE_COLLISION         
    sofa_base_linear_solver     SOFA_LIB_BASE_LINEAR_SOLVER     
    sofa_base_mechanics         SOFA_LIB_BASE_MECHANICS         
    sofa_base_topology          SOFA_LIB_BASE_TOPOLOGY          
    sofa_base_visual            SOFA_LIB_BASE_VISUAL            
)
list(LENGTH SOFA_LIB_BASE_NAME sofaLibBaseList)
math(EXPR passToCommon "${sofaLibBaseList}/2+${passToBase}")


## Put the name of the library SOFA COMMON to search and put it associate CMakeName
list(APPEND SOFA_LIB_COMMON_NAME
    sofa_deformable             SOFA_LIB_DEFORMABLE          
    sofa_explicit_ode_solver    SOFA_LIB_ODE_SOLVER          
    sofa_implicit_ode_solver    SOFA_LIB_IMPLICIT_ODE_SOLVER 
    sofa_loader                 SOFA_LIB_LOADER              
    sofa_mesh_collision         SOFA_LIB_MESH_COLLISION      
    sofa_rigid                  SOFA_LIB_RIGID               
    sofa_simple_fem             SOFA_LIB_SIMPLE_FEM          
    sofa_object_interaction     SOFA_LIB_OBJECT_INTERACTION  
)
list(LENGTH SOFA_LIB_COMMON_NAME sofaCommonLibList)
math(EXPR passToGeneralLib "${sofaCommonLibList}/2+${passToCommon}")


## Put the name of the library SOFA GENERAL to search and put it associate CMakeName
list(APPEND SOFA_LIB_GENERAL_NAME
    sofa_boundary_condition     SOFA_LIB_BOUNDARY_CONDITION     
    sofa_constraint             SOFA_LIB_CONSTRAINT             
    sofa_dense_solver           SOFA_LIB_DENSE_SOLVER           
    sofa_engine                 SOFA_LIB_ENGINE                 
    sofa_exporter               SOFA_LIB_EXPORTER               
    sofa_graph_component        SOFA_LIB_GRAPH_COMPONENT        
    sofa_opengl_visual          SOFA_LIB_OPENGL_VISUAL          
    sofa_preconditioner         SOFA_LIB_PRECONDITIONER         
    sofa_topology_mapping       SOFA_LIB_TOPOLOGY_MAPPING        
    sofa_user_interaction       SOFA_LIB_USER_INTERACTION        
    sofa_validation             SOFA_LIB_VALIDATION              
    sofa_haptics                SOFA_LIB_HAPTICS                
)
list(LENGTH SOFA_LIB_GENERAL_NAME sofaGeneralLibList)
math(EXPR passToAdvancedLib "${sofaGeneralLibList}/2+${passToGeneralLib}")


## Put the name of the library SOFA ADVANCED to search and put it associate CMakeName
list(APPEND SOFA_LIB_ADVANCED_NAME
    sofa_advanced_constraint    SOFA_LIB_ADVANCED_CONSTRAINT        
    sofa_advanced_fem           SOFA_LIB_ADVANCED_FEM               
    sofa_advanced_interaction   SOFA_LIB_ADVANCED_INTERACTION       
    sofa_eigen2_solver          SOFA_LIB_EIGEN2_SOLVER                
    sofa_eulerian_fluid         SOFA_LIB_EULERIAN_FUILD             
    sofa_mjed_fem               SOFA_LIB_MJED_FEM                   
    sofa_non_uniform_fem        SOFA_LIB_NON_UNIFORM_FEM            
    sofa_non_uniform_fem_dev    SOFA_LIB_NON_UNIFORM_FEM_DEV        
    sofa_sph_fluid              SOFA_LIB_SPH_FUILD                   
    sofa_volumetric_data        SOFA_LIB_VOLUMETRIC_DATA             
)
list(LENGTH SOFA_LIB_GENERAL_NAME sofaAdvancedLibList)
math(EXPR passToMiscLib "${sofaAdvancedLibList}/2+${passToAdvancedLib}")


## Put the name of the library SOFA MISC to search and put it associate CMakeName
list(APPEND SOFA_LIB_MISC_NAME
    sofa_misc                   SOFA_LIB_MISC                   
    sofa_misc_collision         SOFA_LIB_MISC_COLLISION         
    sofa_misc_collision_dev     SOFA_LIB_MISC_COLLISION_DEV     
    sofa_misc_fem               SOFA_LIB_MISC_FEM               
    sofa_misc_dev               SOFA_LIB_MISC_DEV               
    sofa_misc_fem_dev           SOFA_LIB_MISC_FEM_DEV           
    sofa_misc_forcefield        SOFA_LIB_MISC_FORCEFIELD        
     sofa_misc_forcefield_dev   SOFA_LIB_MISC_FORCEFIELD_DEV   
    sofa_misc_mapping           SOFA_LIB_MISC_MAPPING           
      SofaMiscMappingDev        SOFA_LIB_MISC_MAPPING_DEV       
    sofa_misc_solver            SOFA_LIB_MISC_SOLVER            
    sofa_misc_solver_dev        SOFA_LIB_MISC_SOLVER_DEV        
    sofa_misc_topology          SOFA_LIB_MISC_TOPOLOGY          
    SofaMiscTopologyDev         SOFA_LIB_MISC_TOPOLOGY_DEV      
    sofa_misc_engine            SOFA_LIB_MISC_ENGINE            
)
list(LENGTH SOFA_LIB_MISC_NAME sofaMiscLibList)
math(EXPR passToExtLib "${sofaMiscLibList}/2+${passToMiscLib}")


## Put the name of the library EXT to search and put it associate CMakeName
if(MSVC)
    list(APPEND SOFA_LIB_MSVC
        iconv    SOFAWIN_ICONV
        glew32   SOFAWIN_GLEW
        glut32   SOFAWIN_GLUT
        libxml2  SOFAWIN_LIBXML2
    )    
else()
    set(SOFA_LIB_MSVC)
endif()

list(APPEND SOFA_LIB_EXT_NAME
    qwt         SOFA_LIB_QWT            
    miniFlowVR  SOFA_LIB_MINI_FLOWVR     
    newmat      SOFA_LIB_NEWMAT          
    tinyxml     SOFA_LIB_TINYXML         
    ${SOFA_LIB_MSVC}
)


## Collect all list of libs names together in one list
list(APPEND SOFA_LIBS_NAME 
    ${SOFA_LIB_NAME}
    ${SOFA_LIB_COMPONENT_NAME}
    ${SOFA_LIB_BASE_NAME}
    ${SOFA_LIB_COMMON_NAME}
    ${SOFA_LIB_GENERAL_NAME}
    ${SOFA_LIB_ADVANCED_NAME}
    ${SOFA_LIB_MISC_NAME}
    ${SOFA_LIB_EXT_NAME}
)

# To use VERBOSE macro (print only if VAR or default VERBOSE_CMAKE is set to true
include(find_lib)
if (WIN32) 
  if (CMAKE_CL_64) 
      set(ARCH_DIR "win32") 
  else() 
      set(ARCH_DIR "win64") 
  endif() 
endif()
list(APPEND SEARCH_LIB_PATHS
    ${SOFA_DIR}/lib
    ${SOFA_DIR}/lib/linux
    ${SOFA_DIR}/lib/linux/sofa-plugins
    ${SOFA_DIR}/lib/${ARCH_DIR}
    ${SOFA_DIR}/lib/${ARCH_DIR}/Common
    ${SOFA_DIR}/lib/${ARCH_DIR}/ReleaseVC7
    ${SOFA_DIR}/lib/${ARCH_DIR}/ReleaseVC8
    ${SOFA_DIR}/lib/${ARCH_DIR}/ReleaseVC9
    ${SOFA_DIR}/lib/${ARCH_DIR}/ReleaseVC10
    #${SOFA_DIR}/bin
)

list(APPEND CMAKE_LIBRARY_PATH ${SEARCH_LIB_PATHS}) # first default path the find_library command will use

## Take the name of the library to found in the list[n],
## found it full path and place it into the variable of the list[n+1]
list(LENGTH SOFA_LIBS_NAME lengthList)
math(EXPR count "${lengthList}/2-1") #because of the foreach expression to take namesLib by pair
if(VERBOSE_SOFA)
    message(STATUS "")
    message(STATUS "----SOFA LIBS LIST : ")
endif()
foreach(index RANGE 0 ${count} 1)
    
    ## Indices to display list in 3 parts
    if(VERBOSE_SOFA)
        if(index EQUAL passToComponent)
            message(STATUS "")
            message(STATUS "----SOFA COMPONENT LIBS LIST : ")
        endif()
        if(index EQUAL passToBase)
            message(STATUS "")
            message(STATUS "----SOFA BASE LIBS LIST : ")
        endif()
        if(index EQUAL passToCommon)
            message(STATUS "")
            message(STATUS "----SOFA COMMON LIBS LIST : ")
        endif()
        if(index EQUAL passToGeneralLib)
            message(STATUS "")
            message(STATUS "----SOFA GENERAL LIBS LIST : ")
        endif()
        if(index EQUAL passToAdvancedLib)
            message(STATUS "")
            message(STATUS "----SOFA ADVANCED LIBS LIST : ")
        endif()
        if(index EQUAL passToMiscLib)
            message(STATUS "")
            message(STATUS "----SOFA MISC LIBS LIST : ")
        endif()
        if(index EQUAL passToExtLib)
            message(STATUS "")
            message(STATUS "----SOFA EXT LIBS LIST : ")
        endif()
    endif(VERBOSE_SOFA)

    ## Get the valueName of the searched library
    math(EXPR indexValue "${index}*2")
    list(GET SOFA_LIBS_NAME ${indexValue} REAL_SOFA_LIB_NAME)
    
    ## Get the variableName of the coresponding searched library
    math(EXPR indexName "${index}*2+1")
    list(GET SOFA_LIBS_NAME ${indexName} CMAKE_SOFA_LIB_NAME)
    
    ## Allow to select library
#    option(SOFA_USE_${CMAKE_SOFA_LIB_NAME} "Build ${CMAKE_SOFA_LIB_NAME}" true)
#    if(SOFA_USE_${CMAKE_SOFA_LIB_NAME})
    
        ## Use the MACRO defined above to find the library with it full path
        FIND_LIB(${CMAKE_SOFA_LIB_NAME} ${REAL_SOFA_LIB_NAME}
                PATHSLIST_DEBUG
                    ${SEARCH_LIB_PATHS}
                    ${SOFA_DIR}/lib/Debug
                    ${SOFA_DIR}/lib64/Debug
                PATHSLIST_RELEASE
                    ${SEARCH_LIB_PATHS}
                    ${SOFA_DIR}/lib/Release
                    ${SOFA_DIR}/lib64/Release
                VERBOSE         ${VERBOSE_SOFA}
                FORCE_DEBUG     true
                FORCE_RELEASE   true
                NO_DEFAULT_PATH # SOFA provide all extlib it need
        )
        
        if(NOT EXISTS ${${CMAKE_SOFA_LIB_NAME}_DEBUG} OR NOT EXISTS ${${CMAKE_SOFA_LIB_NAME}})
            message(WARNING 
                "It seems the ${CMAKE_SOFA_LIB_NAME} does not exist : ${${CMAKE_SOFA_LIB_NAME}}.
                Please, check it in FindSOFA.cmake.")
        endif()
    
        ## Add all libraries (release and then debug) find to one variable
        if(${CMAKE_SOFA_LIB_NAME}_DEBUG)
            list(APPEND SOFA_LIBRARIES
                    optimized ${${CMAKE_SOFA_LIB_NAME}}
                    debug     ${${CMAKE_SOFA_LIB_NAME}_DEBUG}
            )
        else()
            list(APPEND SOFA_LIBRARIES ${${CMAKE_SOFA_LIB_NAME}})
        endif()

#    endif(SOFA_USE_${CMAKE_SOFA_LIB_NAME})
    
endforeach(index)

if(VERBOSE_SOFA)
    message(STATUS "----set this variable : SOFA_LIBRARIES with all libraries found")
    message(STATUS "")
endif()


## ###########################################################################################
## FINALISE AND CHECK
## ###########################################################################################
## handle the QUIETLY and REQUIRED arguments and set SOFA_FOUND to TRUE if 
## all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SOFA DEFAULT_MSG
        SOFA_INCLUDE_FRAMEWORK_DIR
        SOFA_INCLUDE_MODULES_DIR
        SOFA_INCLUDE_APPLICATIONS_DIR        
        SOFA_LIB_CORE
        SOFA_LIB_HELPER
        SOFA_LIB_DEFAULT_TYPE
        SOFA_LIB_SIMULATION
        SOFA_LIB_COMPONENT
)


## ###########################################################################################
## FIND SOFA DEFINES
## ###########################################################################################
if(EXISTS ${SOFA_DIR}/sofaDefines.cfg)
    file(READ ${SOFA_DIR}/sofaDefines.cfg SOFA_DEFINES_CONFIG_FILE_CONTENTS)
    ## get a list of lines
    string(REGEX REPLACE "\r?\n" ";" SOFA_DEFINES_CONFIG_FILE_CONTENTS "${SOFA_DEFINES_CONFIG_FILE_CONTENTS}") 
    if(VERBOSE_SOFA)
        message(STATUS "Get Sofa defines in ${SOFA_DIR}/sofaDefines.cfg : ")
    endif()

    ## Add "-D" for compatibility with CMake
    foreach( define ${SOFA_DEFINES_CONFIG_FILE_CONTENTS})
        list(APPEND SOFA_DEFINES "-D${define}")
    endforeach( define ${SOFA_DEFINES_CONFIG_FILE_CONTENTS})

    add_definitions(${SOFA_DEFINES})

    if(VERBOSE_SOFA)
    message(STATUS "Add SOFA definition into CMake project : ")
        foreach( define ${SOFA_DEFINES})
            message(STATUS "add DEFINITION : ${define}")
        endforeach( define ${SOFA_DEFINES})
    endif(VERBOSE_SOFA)

else()
    message("WARNING : sofaDefines.cfg not found...Sofa defines will not be added to the CMake project.")
endif()

