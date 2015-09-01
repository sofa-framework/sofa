cmake_minimum_required(VERSION 2.8.12)

project(Eigen)


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Eigen
        COMPONENT Eigen_headers
        DESTINATION include
        PATTERN ".svn" EXCLUDE
        PATTERN ".git" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        PATTERN ".cxx" EXCLUDE
        PATTERN ".cpp" EXCLUDE)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unsupported/Eigen
        COMPONENT Eigen_headers
        DESTINATION include
        PATTERN ".svn" EXCLUDE
        PATTERN ".git" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        PATTERN ".cxx" EXCLUDE
        PATTERN ".cpp" EXCLUDE)
