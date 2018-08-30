from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.memory cimport shared_ptr

cdef extern from "../src/SofaEditor/BaseSofaEditor.h" namespace "sofaeditor":
        cdef cppclass SofaEditorState:
                string editorname;
                const vector[string]& getSelection();


        cdef cppclass SofaEditor:
                ctypedef size_t ID

                @staticmethod 
                size_t createIdAndAttachState(shared_ptr[SofaEditorState]);

                @staticmethod
                bool attachState(ID editorId, shared_ptr[SofaEditorState]& s);

                @staticmethod
                shared_ptr[SofaEditorState] getState(size_t);

                @staticmethod
                ID getIdFromEditorName(const string& s);

cdef extern from "../src/SofaEditor/BaseSofaEditor.h" namespace "sofaeditor::SofaEditor":
        cdef SofaEditor.ID InvalidID;
