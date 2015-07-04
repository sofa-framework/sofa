include(${SOFA_CMAKE_DIR}/preProject.cmake)

# only take the librairy headers, not test headers and other stuff
file(GLOB_RECURSE HEADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Eigen/*.h ${CMAKE_CURRENT_SOURCE_DIR}/unsupported/Eigen/*.h)

add_custom_target(${PROJECT_NAME} SOURCES ${HEADER_FILES})

#add_library(${PROJECT_NAME} SHARED ${HEADER_FILES}) # it contains only headers but it is still generating an empty lib
#SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX) # otherwise cmake cannot recognize the language

include(${SOFA_CMAKE_DIR}/postProject.cmake)

