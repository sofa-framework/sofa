include(${SOFA_CMAKE_DIR}/preProject.cmake)

file(GLOB SOURCE_FILES *.cpp PhysicalProperties/*.cpp)
file(GLOB HEADER_FILES *.h PhysicalProperties/*.h)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

find_package(LibXml2 REQUIRED)
include_directories("${LIBXML2_INCLUDE_DIR}" "PhysicalProperties/")

if(${CMAKE_COMPILER_IS_GNUCC})
    AddCompilerFlags("-fPIC")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

