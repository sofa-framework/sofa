#include <string>
#include <vector>
#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest;

#include <SofaEditor/BaseSofaEditor.h>
using sofaeditor::SofaEditor;
using sofaeditor::SofaEditorState;

namespace
{

TEST(SofaEditorState, checkSetEmpty)
{
    SofaEditorState editor;

    ASSERT_EQ( editor.getSelection(), std::vector<std::string>{} );
}

TEST(SofaEditorState, checkSetValue)
{
    SofaEditorState editor;

    ASSERT_EQ( editor.getSelection(), std::vector<std::string>{} );

    editor.setSelectionFromPath({"one", "two", "three", "four"});
    ASSERT_EQ( editor.getSelection(), std::vector<std::string>({"one", "two", "three", "four"}) );
}

TEST(SofaEditorState, checkGetItFromEditorName)
{
    /// Insure that different calls are not returning the same ID.
    SofaEditor::ID a = SofaEditor::createId() ;
    SofaEditor::ID b = SofaEditor::createId() ;
    SofaEditor::ID c = SofaEditor::createId() ;

    ASSERT_NE(a, b);
    ASSERT_NE(a, c);
    ASSERT_NE(b, c);

    SofaEditorState editor1 {"one"} ;
    SofaEditorState editor2 {"two"} ;

    SofaEditor::attachState(a, nullptr);
    SofaEditor::attachState(b, &editor1);
    SofaEditor::attachState(c, &editor2);

    ASSERT_EQ(SofaEditor::getIdFromEditorName("one"), b);
    ASSERT_EQ(SofaEditor::getIdFromEditorName("two"), c);
    ASSERT_EQ(SofaEditor::getIdFromEditorName("invalidName"), SofaEditor::InvalidID);
}

TEST(SofaEditor, test_createId)
{
    /// Insure that different calls are not returning the same ID.
    SofaEditor::ID a = SofaEditor::createId() ;
    SofaEditor::ID b = SofaEditor::createId() ;
    SofaEditor::ID c = SofaEditor::createId() ;

    ASSERT_NE(a, b);
    ASSERT_NE(a, c);
    ASSERT_NE(b, c);
}

TEST(SofaEditor, test_getSetId)
{
    /// Insure that different calls are not returning the same ID.
    SofaEditor::ID a = SofaEditor::createId() ;
    SofaEditor::ID b = SofaEditor::createId() ;

    SofaEditorState sa, sb;

    ASSERT_TRUE(SofaEditor::attachState(a, &sa));
    ASSERT_TRUE(SofaEditor::attachState(b, &sb));

    ASSERT_EQ(&sa, SofaEditor::getState(a));
    ASSERT_EQ(&sb, SofaEditor::getState(b));
}

TEST(SofaEditor, getInvalid)
{
    SofaEditor::ID a = SofaEditor::createId();

    ASSERT_EQ(SofaEditor::getState(a), nullptr);
    ASSERT_EQ(SofaEditor::getState((SofaEditor::ID)10000), nullptr);
}

TEST(SofaEditor, setInvalid)
{
    SofaEditorState sa;
    ASSERT_FALSE(SofaEditor::attachState((SofaEditor::ID)10000, &sa));
}


}


