from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "../src/SofaEditor/BaseSofaEditor.h" namespace "sofaeditor":
        cdef cppclass SofaEditorState:
                const vector[string]& getSelection();
 
        cdef cppclass SofaEditor:
                @staticmethod 
                size_t createId(const SofaEditorState*);

                @staticmethod
                const SofaEditorState* getState(size_t);

