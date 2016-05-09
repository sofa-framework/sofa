#include <sofa/helper/Utils.h>
#include <gtest/gtest.h>

#include <sofa/helper/system/FileSystem.h>
#include <SofaTest/TestMessageHandler.h>

using sofa::helper::Utils;
using sofa::helper::system::FileSystem;

TEST(UtilsTest, string_to_widestring_to_string)
{
    std::string ascii_chars;
    for (char c = 32 ; c <= 126 ; c++)
        ascii_chars.push_back(c);
    EXPECT_EQ(ascii_chars, Utils::narrowString(Utils::widenString(ascii_chars)));

    const std::string s("chaîne de test avec des caractères accentués");
    EXPECT_EQ(s, Utils::narrowString(Utils::widenString(s)));
}

TEST(UtilsTest, widestring_to_string_to_widestring)
{
    const std::string s("chaîne de test avec des caractères accentués");
    const std::wstring ws = Utils::widenString(s);
    EXPECT_EQ(ws, Utils::widenString(Utils::narrowString(ws)));
}

TEST(UtilsTest, downcaseString)
{
    EXPECT_EQ("abcdef", Utils::downcaseString("abcdef"));
    EXPECT_EQ("abcdef", Utils::downcaseString("ABCDEF"));
    EXPECT_EQ("abcdef", Utils::downcaseString("AbCdEf"));
    EXPECT_EQ("abcdef", Utils::downcaseString("ABCDEF"));
}

TEST(UtilsTest, upcaseString)
{
    EXPECT_EQ("ABCDEF", Utils::upcaseString("abcdef"));
    EXPECT_EQ("ABCDEF", Utils::upcaseString("ABCDEF"));
    EXPECT_EQ("ABCDEF", Utils::upcaseString("AbCdEf"));
    EXPECT_EQ("ABCDEF", Utils::upcaseString("ABCDEF"));
}

TEST(UtilsTest, getExecutablePath)
{
    const std::string path = Utils::getExecutablePath();
    EXPECT_TRUE(path.find("SofaFramework_test") != std::string::npos);
}

TEST(UtilsTest, getExecutableDirectory)
{
    const std::string path = Utils::getExecutableDirectory();
    EXPECT_TRUE(path.find("bin") != std::string::npos);
}

TEST(UtilsTest, getSofaPathPrefix)
{
    const std::string prefix = Utils::getSofaPathPrefix();
    EXPECT_TRUE(FileSystem::exists(prefix + "/bin"));
}

TEST(UtilsTest, readBasicIniFile_nonexistentFile)
{
    // this test will raise an error on purpose
    sofa::helper::logging::ScopedDeactivatedTestMessageHandler scopedDeactivatedTestMessageHandler;
    std::map<std::string, std::string> values = Utils::readBasicIniFile("this-file-does-not-exist");
    EXPECT_TRUE(values.empty());
}

TEST(UtilsTest, readBasicIniFile)
{
    const std::string path = std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/UtilsTest.ini";
    std::map<std::string, std::string> values = Utils::readBasicIniFile(path);
    EXPECT_EQ(3u, values.size());
    EXPECT_EQ(1u, values.count("a"));
    EXPECT_EQ("b again", values["a"]);
    EXPECT_EQ(1u, values.count("someKey"));
    EXPECT_EQ("someValue", values["someKey"]);
    EXPECT_EQ(1u, values.count("foo bar baz"));
    EXPECT_EQ("qux 42", values["foo bar baz"]);
}
