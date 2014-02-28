include(${SOFA_CMAKE_DIR}/preProject.cmake)

find_package(LibXml2 REQUIRED)
include_directories("${LIBXML2_INCLUDE_DIR}")

if(${CMAKE_COMPILER_IS_GNUCC})
    AddCompilerFlags("-fPIC")
endif()

AddCompilerDefinitions("use_namespace")

file(GLOB SOURCE_FILES *.cpp)
file(GLOB HEADER_FILES *.h)
add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)

