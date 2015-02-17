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

TEST(UtilsTest, getExecutablePath)
{
    EXPECT_FALSE(Utils::getExecutablePath().empty());
}

TEST(UtilsTest, readBasicIniFile_nonexistentFile)
{
    std::map<std::string, std::string> values = Utils::readBasicIniFile("this-file-does-not-exist");
    EXPECT_TRUE(values.empty());
}

TEST(UtilsTest, readBasicIniFile)
{
    const std::string path = std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/UtilsTest.ini";
    std::map<std::string, std::string> values = Utils::readBasicIniFile(path);
    EXPECT_EQ(3, values.size());
    EXPECT_EQ(1, values.count("a"));
    EXPECT_EQ("b again", values["a"]);
    EXPECT_EQ(1, values.count("someKey"));
    EXPECT_EQ("someValue", values["someKey"]);
    EXPECT_EQ(1, values.count("foo bar baz"));
    EXPECT_EQ("qux 42", values["foo bar baz"]);
}
