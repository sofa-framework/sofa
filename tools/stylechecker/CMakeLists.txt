cmake_minimum_required(VERSION 3.1)

project(stylecheck.exe)

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  message(STATUS "Stylecheck out-of-tree configuration")

  if( NOT DEFINED LLVM_PATH )
    message(FATAL_ERROR "LLVM_PATH must be provided using -DLLVM_PATH=<path to llvm package root>")
  endif()

  link_directories(${LLVM_PATH}/lib)
  include_directories(${LLVM_PATH}/include)

  add_definitions(
    -D__STDC_LIMIT_MACROS
    -D__STDC_CONSTANT_MACROS
  )

  if( CMAKE_SYSTEM_NAME MATCHES "Darwin" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
  endif()
else()
  message(STATUS "Stylecheck in-tree configuration")
endif()

add_executable(stylecheck.exe
  stylecheck.cpp
  utilsllvm.cpp
  fileutils.cpp
)

if( MSVC )
  # Disable warnings for Stylecheck
  add_definitions(
    -wd4722 # Suppress ''destructor'' : destructor never returns, potential memory leak
  )

  # Put project in solution folder
  set_target_properties(stylecheck
    PROPERTIES FOLDER "Clang executables"
  )
else()
  # Disable RTTI to be compatible with LLVM/Clang libraries.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -fno-rtti")
endif()

# Clang dependencies.
target_link_libraries(stylecheck.exe
  clangTooling 
  clangFrontend
  clangSerialization
  clangDriver
  clangParse
  clangSema
  clangAnalysis
  clangAST
  clangBasic
  clangEdit
  clangLex
  clangASTMatchers 
  clang
)

# LLVM dependencies.
target_link_libraries(stylecheck.exe
  #LLVMSupport
  #LLVMTransformUtils
  #LLVMMC # Object, Support
  #LLVMMCParser # MC, Support
  #LLVMObject # Support
  #LLVMCore # Support
  #LLVMBitReader
  #LLVMScalarOpts
  LLVMSupport
  LLVMOption
)

# Platform dependencies.
if( WIN32 )
  target_link_libraries(stylecheck.exe
    shlwapi
  )
else()
  target_link_libraries(stylecheck.exe
    pthread
    dl
    ncurses
    z
  )
endif()

install(TARGETS stylecheck.exe
  RUNTIME DESTINATION bin)
