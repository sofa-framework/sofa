#include <sofa/helper/system/Utils.h>
#include <gtest/gtest.h>

using namespace sofa::helper::system;

TEST(UtilsTest, string_to_widestring_to_string)
{
    std::string ascii_chars;
    for (char c = 32 ; c <= 126 ; c++)
        ascii_chars.push_back(c);
    EXPECT_EQ(ascii_chars, Utils::ws2s(Utils::s2ws(ascii_chars)));

    const std::string s("chaîne de test avec des caractères accentués");
    EXPECT_EQ(s, Utils::ws2s(Utils::s2ws(s)));
}

TEST(UtilsTest, widestring_to_string_to_widestring)
{
    const std::string s("chaîne de test avec des caractères accentués");
    const std::wstring ws = Utils::s2ws(s);
    EXPECT_EQ(ws, Utils::s2ws(Utils::ws2s(ws)));
}
