# distutils: language=c++
from cpp_SofaEditor cimport SofaEditor, SofaEditorState
                
def createId():
        return SofaEditor.createId(NULL)

def getSelection():
        cdef const SofaEditorState* state = SofaEditor.getState(0)
        if state == NULL:
            return []
        return state.getSelection()
