cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    xmlvisitor.h

    )

set(SOURCE_FILES

    main.cpp
    xmlvisitor.cpp

    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

#set(LINKER_DEPENDENCIES  )

include(${SOFA_CMAKE_DIR}/postProject.cmake)

