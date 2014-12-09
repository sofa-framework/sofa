cmake_minimum_required(VERSION 2.8.12)

project(Eigen)

set(EIGEN_INSTALL_INCLUDE_DIR include    CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Eigen
        COMPONENT Eigen_headers
        DESTINATION "${EIGEN_INSTALL_INCLUDE_DIR}"
        PATTERN ".svn" EXCLUDE
        PATTERN ".git" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        PATTERN ".cxx" EXCLUDE
        PATTERN ".cpp" EXCLUDE
        )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unsupported/Eigen
        COMPONENT Eigen_headers
        DESTINATION "${EIGEN_INSTALL_INCLUDE_DIR}"
        PATTERN ".svn" EXCLUDE
        PATTERN ".git" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        PATTERN ".cxx" EXCLUDE
        PATTERN ".cpp" EXCLUDE
        )
