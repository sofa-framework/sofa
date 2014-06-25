CXX = g++
CFLAGS = -fno-rtti

LLVM_SRC_PATH = /home/dmarchal/project/2013/metaprogramming/clang/llvm2/llvm
LLVM_BUILD_PATH = /home/dmarchal/project/2013/metaprogramming/clang/build/

LLVM_BIN_PATH = $(LLVM_BUILD_PATH)/Debug+Asserts/bin
LLVM_LIBS=all

LLVM_CONFIG_COMMAND = llvm-config --cxxflags --ldflags --libs $(LLVM_LIBS) 
LLVM_CONFIG_CFLAG = llvm-config --cxxflags 
%CLANG_BUILD_FLAGS = -I$(LLVM_SRC_PATH)/tools/clang/include \
%                                      -I$(LLVM_BUILD_PATH)/tools/clang/include

CLANGFLAGS =  -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS  -std=c++11 -fPIC -rdynamic 

CLANGLIBS = -ldl \
	-lclangFrontendTool -lclangFrontend -lclangDriver \
	-lclangSerialization -lclangCodeGen -lclangParse \
	-lclangSema -lclangStaticAnalyzerFrontend \
	-lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore \
 	-lclangAnalysis -lclangARCMigrate  -lclangRewriteCore \
	-lclangEdit -lclangAST -lclangLex -lclangBasic -lclang \
	-lclangTooling -lclangASTMatchers  -lclangBasic  -lpthread -ldl \
  -lclangFrontendTool -lclangFrontend -lclangDriver \
        -lclangSerialization -lclangCodeGen -lclangParse \
        -lclangSema -lclangStaticAnalyzerFrontend \
        -lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore \
        -lclangAnalysis -lclangARCMigrate  -lclangRewriteCore \
        -lclangEdit -lclangAST -lclangLex -lclangBasic -lclang \
        -lclangTooling -lclangASTMatchers -lclangBasic -rdynamic 

all: stylecheck

stylecheck: stylecheck.cpp utilsllvm.cpp fileutils.o
	$(CXX) stylecheck.cpp utilsllvm.cpp fileutils.o $(CFLAGS) -o stylecheck.exe $(CLANGLIBS)  $(CLANGFLAGS) `$(LLVM_CONFIG_COMMAND)` `$(LLVM_CONFIG_COMMAND)` -ldl -lncurses  -lz


clean:
	rm -rf *.o *.ll llvmtest
