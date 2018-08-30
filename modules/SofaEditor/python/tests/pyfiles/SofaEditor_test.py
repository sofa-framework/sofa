# -*- coding: utf-8 -*
import unittest
from SofaEditor import SofaEditor, SofaEditorState

class SofaEditorState_test(unittest.TestCase):
    def test_constructor(self):
        anEditor = SofaEditorState("editor1")
        self.assertEqual("editor1", anEditor.editorname)

    def test_property(self):
        anEditor = SofaEditorState("editor1")
        self.assertEqual("editor1", anEditor.editorname)

        anEditor.editorname = "newName"
        self.assertEqual("newName", anEditor.editorname)

class SofaEditor_test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ## Create two states and register them.
        cls.s1 = SofaEditorState("editor1")
        cls.s2 = SofaEditorState("editor2")
        cls.id1 = SofaEditor.createIdAndAttachState(cls.s1)
        cls.id2 = SofaEditor.createIdAndAttachState(cls.s2)

    def test_getIdFromEditorname(self):
        id1 = SofaEditor.getIdFromEditorName("editor1")
        id2 = SofaEditor.getIdFromEditorName("editor2")

        self.assertEqual(SofaEditor_test.id1, id1)
        self.assertEqual(SofaEditor_test.id2, id2)

    def test_getState(self):
        id1 = SofaEditor.getIdFromEditorName("editor1")
        id2 = SofaEditor.getIdFromEditorName("editor2")

        s1 = SofaEditor.getState(id1)
        s2 = SofaEditor.getState(id2)

        self.assertEqual(SofaEditor_test.s1, s1)
        self.assertEqual(SofaEditor_test.s2, s2)

    def test_getSelection(self):
        id1 = SofaEditor.getIdFromEditorName("editor1")
        id2 = SofaEditor.getIdFromEditorName("editor2")

        s1 = SofaEditor.getSelection(id1)
        s2 = SofaEditor.getSelection(id2)

        self.assertEqual([], s1)
        self.assertEqual([], s2)

def run():
    for testcase in [SofaEditorState_test, SofaEditor_test]:
        suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(suite)
    return True
