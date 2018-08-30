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
    std::shared_ptr<SofaEditorState> editor1 = std::make_shared<SofaEditorState>("one") ;
    std::shared_ptr<SofaEditorState> editor2 = std::make_shared<SofaEditorState>("two") ;
    std::shared_ptr<SofaEditorState> editor3 = std::make_shared<SofaEditorState>("three") ;

    /// Insure that different calls are not returning the same ID.
    SofaEditor::ID a = SofaEditor::createIdAndAttachState(editor1) ;
    SofaEditor::ID b = SofaEditor::createIdAndAttachState(editor2) ;
    SofaEditor::ID c = SofaEditor::createIdAndAttachState(editor3) ;

    ASSERT_NE(a, b);
    ASSERT_NE(a, c);
    ASSERT_NE(b, c);

    SofaEditor::attachState(c, editor1);
    SofaEditor::attachState(b, editor2);
    SofaEditor::attachState(a, editor3);

    ASSERT_EQ(SofaEditor::getIdFromEditorName("one"), c);
    ASSERT_EQ(SofaEditor::getIdFromEditorName("two"), b);
    ASSERT_EQ(SofaEditor::getIdFromEditorName("three"), a);

    ASSERT_EQ(SofaEditor::getIdFromEditorName("invalidName"), SofaEditor::InvalidID);
}

TEST(SofaEditor, test_createIdAndAttachState)
{
    std::shared_ptr<SofaEditorState> editor1 = std::make_shared<SofaEditorState>("one") ;
    std::shared_ptr<SofaEditorState> editor2 = std::make_shared<SofaEditorState>("two") ;
    std::shared_ptr<SofaEditorState> editor3 = std::make_shared<SofaEditorState>("three") ;

    /// Insure that different calls are not returning the same ID.
    SofaEditor::ID a = SofaEditor::createIdAndAttachState(editor1) ;
    SofaEditor::ID b = SofaEditor::createIdAndAttachState(editor2) ;
    SofaEditor::ID c = SofaEditor::createIdAndAttachState(editor3) ;

    ASSERT_NE(a, b);
    ASSERT_NE(a, c);
    ASSERT_NE(b, c);
}

TEST(SofaEditor, test_getSetId)
{
    std::shared_ptr<SofaEditorState> sa;
    std::shared_ptr<SofaEditorState> sb;

    /// Insure that different calls are not returning the same ID.
    SofaEditor::ID a = SofaEditor::createIdAndAttachState(sa) ;
    SofaEditor::ID b = SofaEditor::createIdAndAttachState(sb) ;

    ASSERT_TRUE(SofaEditor::attachState(a, sb));
    ASSERT_TRUE(SofaEditor::attachState(b, sa));

    ASSERT_EQ(sa, SofaEditor::getState(b));
    ASSERT_EQ(sb, SofaEditor::getState(a));
}

TEST(SofaEditor, getInvalid)
{
    ASSERT_EQ(SofaEditor::getState((SofaEditor::ID)10000), nullptr);
}

TEST(SofaEditor, setInvalid)
{
    std::shared_ptr<SofaEditorState> sa;
    ASSERT_FALSE(SofaEditor::attachState((SofaEditor::ID)10000, sa));
}


}


