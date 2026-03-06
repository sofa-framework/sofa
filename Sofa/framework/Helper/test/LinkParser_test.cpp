#include <sofa/helper/LinkParser.h>
#include <gtest/gtest.h>

namespace sofa
{

TEST(LinkParser, empty)
{
    sofa::helper::LinkParser parser("");
    EXPECT_EQ(parser.getLink(), "");
}

TEST(LinkParser, hasPrefixTrue)
{
    sofa::helper::LinkParser parser("@dsaf");
    EXPECT_TRUE(parser.hasPrefix());
}

TEST(LinkParser, hasPrefixFalse)
{
    sofa::helper::LinkParser parser("dsaf");
    EXPECT_FALSE(parser.hasPrefix());
}

TEST(LinkParser, leadingSpace)
{
    sofa::helper::LinkParser parser("  example");
    EXPECT_EQ(parser.cleanLink().getLink(), "example");
}

TEST(LinkParser, separator)
{
    sofa::helper::LinkParser parser("example\\separator");
    EXPECT_EQ(parser.cleanLink().getLink(), "example/separator");
}

TEST(LinkParser, doubleSeparator)
{
    sofa::helper::LinkParser parser("example\\\\separator");
    EXPECT_EQ(parser.cleanLink().getLink(), "example/separator");
}

TEST(LinkParser, split)
{
    sofa::helper::LinkParser parser("@/root/node1/node2/component");
    const auto decomposition = parser.split();
    ASSERT_EQ(decomposition.size(), 4);
    EXPECT_EQ(decomposition[0], "root");
    EXPECT_EQ(decomposition[1], "node1");
    EXPECT_EQ(decomposition[2], "node2");
    EXPECT_EQ(decomposition[3], "component");
}

TEST(LinkParser, splitWithSpace)
{
    sofa::helper::LinkParser parser("@/root/node 1/node 2/component  ");
    const auto decomposition = parser.split();
    ASSERT_EQ(decomposition.size(), 4);
    EXPECT_EQ(decomposition[0], "root");
    EXPECT_EQ(decomposition[1], "node 1");
    EXPECT_EQ(decomposition[2], "node 2");
    EXPECT_EQ(decomposition[3], "component  ");
}

TEST(LinkParser, validateInvalidBrackets)
{
    sofa::helper::LinkParser parser("@/root/[invalid/component");
    parser.validate();
    const auto errors = parser.getErrors();
    ASSERT_FALSE(errors.empty());
    EXPECT_NE(errors[0].find("it starts with '['"), std::string::npos);
    EXPECT_NE(errors[0].find("does not end with ']'"), std::string::npos);
}

TEST(LinkParser, validBrackets)
{
    sofa::helper::LinkParser parser("@/root/[valid]/component");
    const auto decomposition = parser.split();
    ASSERT_EQ(decomposition.size(), 3);
    EXPECT_EQ(decomposition[1], "[valid]");
}

TEST(LinkParser, cleanLinkMore)
{
    sofa::helper::LinkParser parser("  @root\\\\node//component");
    parser.cleanLink();
    EXPECT_EQ(parser.getLink(), "@root/node/component");
}

TEST(LinkParser, multipleBackslashes)
{
    sofa::helper::LinkParser parser("@root\\\\\\\\node");
    parser.cleanLink();
    EXPECT_EQ(parser.getLink(), "@root/node");
}

}
