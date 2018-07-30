from SofaTest import *

def run():
    from SofaEditor import SofaEditor, SofaEditorState

    anEditor = SofaEditorState("editor1")
    EXPECT_EQ("editor1", anEditor.editorname)

    anEditor.editorname = "newName"
    EXPECT_EQ("newName", anEditor.editorname)

    e1 = SofaEditorState("editor1")
    e2 = SofaEditorState("editor2")

    s1 = SofaEditor.createId(e1)
    s2 = SofaEditor.createId(e2)

    as1 = SofaEditor.getIdFromEditorName("editor1")
    as2 = SofaEditor.getIdFromEditorName("editor2")

    EXPECT_EQ(s1, as1)
    EXPECT_EQ(s2, as2)

    s1 = SofaEditor.getState(as1)
    s2 = SofaEditor.getState(as2)
    EXPECT_EQ(e1, s1)
    EXPECT_EQ(e2, s2)

    s1 = SofaEditor.getSelection(as1)
    s2 = SofaEditor.getSelection(as2)
    EXPECT_EQ([], s1)
    EXPECT_EQ([], s2)

    return True

def createScene(root):
    run()
