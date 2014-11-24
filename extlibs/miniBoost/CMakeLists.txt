cmake_minimum_required(VERSION 2.8.12)

project(Boost)

set(BOOST_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Directory for Boost find_package")

set(BOOST_INSTALL_INCLUDE_DIR include/sofa/    CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

file( GLOB_RECURSE BOOST_HEADERS 
    *.hxx
    *.h
    *.hpp
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ 
        DESTINATION "${BOOST_INSTALL_INCLUDE_DIR}"
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )
