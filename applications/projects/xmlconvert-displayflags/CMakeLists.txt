include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    xmlvisitor.h

    )

set(SOURCE_FILES

    main.cpp
    xmlvisitor.cpp

    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

AddLinkerDependencies(tinyxml SofaCore)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

