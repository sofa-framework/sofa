#include <sofa/helper/system/Utils.h>
#include <gtest/gtest.h>

using namespace sofa::helper::system;

#ifdef WIN32

TEST(UtilsTest, string_to_widestring_to_string)
{
    std::string ascii_chars;
    for (char c = 32 ; c <= 126 ; c++)
        ascii_chars.push_back(c);
    EXPECT_EQ(ascii_chars, Utils::ws2s(Utils::s2ws(ascii_chars)));
}

#endif
