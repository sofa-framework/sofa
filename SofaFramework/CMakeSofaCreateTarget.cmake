# - Create an imported target from a library path and an include dir path.
#   Handle the special case where LIBRARY_PATH is in fact an existing target.
#   Handle the case where LIBRARY_PATH contains the following syntax supported by cmake: 
#                                      "optimized /usr/lib/foo.lib debug /usr/lib/foo_d.lib"
#
# sofa_create_imported_target(TARGETNAME LIBRARY_PATH INCLUDE_DIRS)
#  TARGETNAME_Target  - (output) variable which contains the name of the created target. 
#                       It is usually contains TARGETNAME with one notable exception. 
#                       If LIBRARY_PATH is an existing target, TARGETNAME_Target
#                       contains LIBRARY_PATH instead.
#  TARGETNAME         - (input) the name of the target to create.
#  NAMESPACE          - (input) the namespace where the target is put.
#  LIBRARY_PATH       - (input) the path to the library ( .so or .lib depending on the platform)
#  INCLUDE_DIRS       - (input) include directories associated with the library, 
#                       which are added as INTERFACE_INCLUDE_DIRECTORIES for the target.
#
# The typical usage scenario is to convert the absolute paths to a system library that cmake return 
# after a find_package call into an imported target. By using the cmake target mechanism, it is 
# easier to redistribute a software that depends on system libraries, whose locations are not 
# known before hand on the consumer system.
#
# For further reference about this subject :
# http://public.kitware.com/pipermail/cmake-developers/2014-March/009983.html
# Quoted from https://github.com/Kitware/CMake/blob/master/Help/manual/cmake-packages.7.rst
# "Note that it is not advisable to populate any properties which may contain paths, 
#  such as :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` and :prop_tgt:`INTERFACE_LINK_LIBRARIES`, 
#  with paths relevnt to dependencies. That would hard-code into installed packages the 
#  include directory or library paths for dependencies as found on the machine the package 
#  was made on."
#
# Example: 
#
# add_library( SHARED myLib )
# find_package(PNG REQUIRED)
# sofa_create_target( PNG MyNamespace "${PNG_LIBRARY}" "${PNG_INCLUDE_DIRS}" ) 
# target_link_libraries( myLib PUBLIC ${PNG_Target} )
#

include(CMakeParseLibraryList)

macro(sofa_create_target TARGETNAME NAMESPACE LIBRARY_PATH INCLUDE_DIRS)
    # message("TARGETNAME ${TARGETNAME}")
    set(NAMESPACE_TARGETNAME "${NAMESPACE}::${TARGETNAME}")
    # message("LIBRARY_PATH ${LIBRARY_PATH}")
    parse_library_list( "${LIBRARY_PATH}" FOUND LIB_FOUND DEBUG LIB_DEBUG OPT LIB_OPT GENERAL LIB_GEN )
    
    message("FOUND ${LIB_FOUND} DEBUG: ${LIB_DEBUG} OPT: ${LIB_OPT} GEN: ${LIB_GEN}")
    if(${LIB_FOUND} ) 
        if(NOT TARGET ${TARGETNAME} )
            set(${TARGETNAME}_Target ${NAMESPACE_TARGETNAME} )
            if(NOT TARGET ${NAMESPACE_TARGETNAME} )
                add_library( ${NAMESPACE_TARGETNAME} UNKNOWN IMPORTED )
                set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}" )
                if( NOT ${LIB_DEBUG} STREQUAL "") 
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION_DEBUG "${LIB_DEBUG}" )
                endif()
                if( NOT ${LIB_OPT} STREQUAL "")
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION "${LIB_OPT}" )
                elseif( NOT ${LIB_GEN} STREQUAL "" )
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION "${LIB_GEN}" )
                endif()
            endif()
        else()
            message( SEND_ERROR "sofa_create_target error. ${TARGETNAME} is an already an existing TARGET.\ 
                                 Choose a different name.")
        endif()
    else()
        if(NOT TARGET "${LIBRARY_PATH}" )
           # message("${LIBRARY_PATH} is not a TARGET")
            if(NOT TARGET ${TARGETNAME} )
               # message("${TARGETNAME} is not a TARGET")
                set(${TARGETNAME}_Target ${NAMESPACE_TARGETNAME} )
                if(NOT TARGET ${NAMESPACE_TARGETNAME} )
                   # message("${NAMESPACE_TARGETNAME} is not a TARGET")
                    add_library( ${NAMESPACE_TARGETNAME} UNKNOWN IMPORTED )
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION "${LIBRARY_PATH}" )
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}" )    
                endif()
            else()
                message( SEND_ERROR "sofa_create_target error. ${TARGETNAME} is an already an existing TARGET.\ 
                                     Choose a different name.")
            endif()
        else()
    #        message("${LIBRARY_PATH} is a TARGET")
            set(${TARGETNAME}_Target ${LIBRARY_PATH} )
        endif()
    
    endif()
endmacro()
