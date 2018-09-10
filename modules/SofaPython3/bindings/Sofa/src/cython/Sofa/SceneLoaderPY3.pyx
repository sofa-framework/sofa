# -*- coding: ASCII -*-
import os

from libcpp.vector cimport vector as std_vector
from libcpp.string cimport string as std_string
from libcpp cimport bool
from .Node cimport Node, create as createNode
from Sofa.cpp.sofa.simulation.Node_wrap cimport Node as cpp_Node, SPtr as NodeSPtr

cdef extern from "<sofa/simulation/SceneLoaderFactory.h>" namespace "sofa::simulation":
    cdef cppclass SceneLoader:
        SceneLoader() except +

    cdef cppclass SceneLoaderFactory:
        @staticmethod
        SceneLoaderFactory* getInstance()

        void addEntry(SceneLoader*)

cdef public cppclass SceneLoaderPY3(SceneLoader):
    bool canLoadFileExtension(const char *extension) with gil:
        return <bytes>extension in [b"py", b"py3", b"pyscn", b"py3scn"]

    bool canWriteFileExtension(const char *extension) with gil:
        return False

    void loadSceneWithArguments(const char *filename,
                                const std_vector[std_string]& arguments,
                                NodeSPtr* root_out) with gil:
        print("Load scene" + <bytes>filename)

    bool loadTestWithArguments(const char *filename,
                               const std_vector[std_string]& arguments) with gil:
        print("A")

    NodeSPtr load(const char* filename) with gil:
        f=open(filename,"r")
        if f is not None:
            code = compile(f.read(), filename, 'exec')
            locals={}
            exec(code, globals(), locals)
            if "createScene" in locals:
                root = createNode("root")
                locals["createScene"](root)
                return (<Node>root).ptr
        return NodeSPtr()

    std_string getFileTypeDesc() with gil:
        return str("Load python scenes")

    void getExtensionList(std_vector[std_string]* list) with gil:
        list.push_back(str(".py"))

cpdef public registerLoader():
    SceneLoaderFactory.getInstance().addEntry(new SceneLoaderPY3())

