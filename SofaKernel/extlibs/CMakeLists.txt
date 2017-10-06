cmake_minimum_required(VERSION 3.1)

set(EXTLIBS_TARGETS
    newmat
    tinyxml
    json
)

## Necessary ones
add_subdirectory(eigen-3.2.7 eigen)
add_subdirectory(newmat)
add_subdirectory(tinyxml)
add_subdirectory(json)

foreach(target ${EXTLIBS_TARGETS})
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "_d")
    set_target_properties(${target} PROPERTIES FOLDER "SofaExtlibs")
endforeach()

# C++11
set(CPP11_EXTLIBS_TARGETS ${EXTLIBS_TARGETS})
if(NOT WIN32)
    if(SOFA_USE_C++11)
        foreach(target ${CPP11_EXTLIBS_TARGETS})
            target_compile_options(${target} PRIVATE "$<$<STREQUAL:$<TARGET_PROPERTY:LINKER_LANGUAGE>,CXX>:${CXX11_FLAG}>")
            if ("${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}" MATCHES ".*clang") # Wno-deprecated-register does not exist in GCC
                target_compile_options(${target} PRIVATE "$<$<STREQUAL:$<TARGET_PROPERTY:LINKER_LANGUAGE>,CXX>:-Wno-deprecated-register>")
            endif()
        endforeach()
    endif()
endif()
