# distutils: language=c++
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string
from cython.operator cimport dereference as deref
from cpp_SofaEditor cimport SofaEditor as cppSofaEditor
from cpp_SofaEditor cimport SofaEditorState as cppSofaEditorState
from cpp_SofaEditor cimport InvalidID

cdef class SofaEditorState(object):
        """
        Holds the state of an editor in sofa.

        Eg:
            /// To retrive the default editor
            e = SofaEditor.getState()
            e.getSelection()

            /// To add a new editor from python
            e = SofaEditorState("newEditor")
            i = SofaEditor.createId(e)
            ee = SofaEditor.getState(i)
        """
        cdef shared_ptr[cppSofaEditorState] sptr

        def __cinit__(self, str name=None):
            if name is not None:
                self.sptr = make_shared[cppSofaEditorState](<string>name)
            else:
                self.sptr = make_shared[cppSofaEditorState]()

        cdef lateInitFromC(self, const shared_ptr[cppSofaEditorState]& state):
            self.sptr = state
            return self

        def getSelection(self):
            """
            Returns the current selection in this editor
            """
            return deref(self.sptr).getSelection()

        def __str__(self):
            return "SofaEditorState('"+str(self.editorname)+"')"

        def __eq__(self, other):
            if not isinstance(other, SofaEditorState):
                return False

            cdef SofaEditorState tother = other
            return self.sptr.get() == tother.sptr.get()

        @property
        def editorname(self):
            return deref(self.sptr).editorname

        @editorname.setter
        def editorname(self, str newname not None):
            deref(self.sptr).editorname = newname

def createId(SofaEditorState state not None):
        """
        Create a new id for the provided SofaEditorState.
        Returns the SofaEditor::ID on success.

        Examples:
        """
        return cppSofaEditor.createId(state.sptr)

def getIdFromEditorName(str editorname not None):
        """
        Returns the ID associated with the provided editor's name
        Returns the SofaEditor::ID on success.
        Returns InvalidId on failure.

        Examples:
            id2 = SofaEditor.getIdFromEditorName("pyQtEditor")
            sel = SofaEditor.getState(id2)
            print(str(sel.getSelection()))
        """
        cdef cppSofaEditor.ID i = cppSofaEditor.getIdFromEditorName(editorname)
        if i == InvalidID:
            return None
        return i

def getState(id=0):
        """
        Returns a SofaEditorState that is associated to the given editor's 'id'.
        If not 'id' is provided, the zero'th editor is used.
        If the editor's id does not existst it returns None.

        Examples:
            sel = SofaEditor.getState(0)
            print(str(sel.getSelection())
        """
        cdef shared_ptr[cppSofaEditorState] state = cppSofaEditor.getState(<cppSofaEditor.ID>id)

        if state.get() == NULL:
            return None
        return SofaEditorState().lateInitFromC(state)

def getSelection(id=0):
        """
        Returns the current selection.

        This is a shorter version of:
        SofaEditor.getState().getSelection()
        """
        return getState(id).getSelection()
