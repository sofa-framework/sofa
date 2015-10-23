cmake_minimum_required(VERSION 2.8.12)

message("Using miniBoost, to use an installed Boost, set BOOST_ROOT to the install directory")

set(BOOST_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Directory for Boost find_package")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION include
        COMPONENT Boost_headers
        FILES_MATCHING PATTERN "*.hpp")
