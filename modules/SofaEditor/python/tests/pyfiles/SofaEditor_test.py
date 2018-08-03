import unittest
from SofaTest import *
from SofaEditor import SofaEditor, SofaEditorState

class SofaEditorState_test(unittest.TestCase):
    def test_constructor(self):
        anEditor = SofaEditorState("editor1")
        EXPECT_EQ("editor1", anEditor.editorname)

    def test_property(self):
        anEditor = SofaEditorState("editor1")
        EXPECT_EQ("editor1", anEditor.editorname)

        anEditor.editorname = "newName"
        EXPECT_EQ("newName", anEditor.editorname)

class SofaEditor_test(unittest.TestCase):
    def test_createId(self):
        e1 = SofaEditorState("editor1")
        e2 = SofaEditorState("editor2")

        s1 = SofaEditor.createId(e1)
        s2 = SofaEditor.createId(e2)

        as1 = SofaEditor.getIdFromEditorName("editor1")
        as2 = SofaEditor.getIdFromEditorName("editor2")

        EXPECT_EQ(s1, as1)
        EXPECT_EQ(s2, as2)

    def test_getState(self):
        s1 = SofaEditor.getState(as1)
        s2 = SofaEditor.getState(as2)
        EXPECT_EQ(e1, s1)
        EXPECT_EQ(e2, s2)

    def test_getSelection(self):
        s1 = SofaEditor.getSelection(as1)
        s2 = SofaEditor.getSelection(as2)
        EXPECT_EQ([], s1)
        EXPECT_EQ([], s2)

def run():
    unittest.main()
    return True
