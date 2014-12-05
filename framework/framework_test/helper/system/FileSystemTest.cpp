
#include <sofa/helper/system/FileSystem.h>
#include <gtest/gtest.h>
#include <exception>
#include <algorithm>

using namespace sofa::helper::system;

static std::string getPath(std::string s) {
    return std::string(FRAMEWORK_TEST_RESOURCES_DIR) + std::string("/") + s;
}

// Mmmh, we can't have empty directories in git...
// TEST(FileSystemTest, listDirectory_empty)
// {
//     std::vector<std::string> fileList;
//     FileSystem::listDirectory(getPath("empty-directory"), fileList);
// 	EXPECT_TRUE(fileList.empty());
// }

TEST(FileSystemTest, listDirectory_nonEmpty)
{
    std::vector<std::string> fileList;
    FileSystem::listDirectory(getPath("non-empty-directory"), fileList);
    // Workaround: svn adds a '.svn' directory in each subdirectory
    if (std::find(fileList.begin(), fileList.end(), ".svn") != fileList.end())
        EXPECT_EQ(fileList.size(), 4u);
    else
        EXPECT_EQ(fileList.size(), 3u);
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileA.txt") != fileList.end());
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileB.txt") != fileList.end());
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileC.so") != fileList.end());
}

TEST(FileSystemTest, listDirectory_nonEmpty_trailingSlash)
{
    std::vector<std::string> fileList;
    FileSystem::listDirectory(getPath("non-empty-directory/"), fileList);
    // Workaround: svn adds a '.svn' directory in each subdirectory
    if (std::find(fileList.begin(), fileList.end(), ".svn") != fileList.end())
        EXPECT_EQ(fileList.size(), 4u);
    else
        EXPECT_EQ(fileList.size(), 3u);
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileA.txt") != fileList.end());
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileB.txt") != fileList.end());
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileC.so") != fileList.end());
}

TEST(FileSystemTest, listDirectory_withExtension_multipleMatches)
{
    std::vector<std::string> fileList;
    FileSystem::listDirectory(getPath("non-empty-directory/"), fileList, "txt");
    EXPECT_EQ(fileList.size(), 2u);
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileA.txt") != fileList.end());
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileB.txt") != fileList.end());
}

TEST(FileSystemTest, listDirectory_withExtension_oneMatch)
{
    std::vector<std::string> fileList;
    FileSystem::listDirectory(getPath("non-empty-directory/"), fileList, "so");
    EXPECT_EQ(fileList.size(), 1u);
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileC.so") != fileList.end());
}

TEST(FileSystemTest, listDirectory_withExtension_noMatch)
{
    std::vector<std::string> fileList;
    FileSystem::listDirectory(getPath("non-empty-directory/"), fileList, "h");
    EXPECT_TRUE(fileList.empty());
}

TEST(FileSystemTest, exists_yes)
{
    EXPECT_TRUE(FileSystem::exists(getPath("non-empty-directory/fileA.txt")));
}

TEST(FileSystemTest, exists_yes_directory)
{
    EXPECT_TRUE(FileSystem::exists(getPath("non-empty-directory")));
}

TEST(FileSystemTest, exists_yes_directory_trailingSlash)
{
    EXPECT_TRUE(FileSystem::exists(getPath("non-empty-directory/")));
}

TEST(FileSystemTest, exists_nope)
{
    EXPECT_FALSE(FileSystem::exists(getPath("thisFileDoesNotExist.txt")));
}

TEST(FileSystemTest, isDirectory_yes)
{
    EXPECT_TRUE(FileSystem::isDirectory(getPath("non-empty-directory")));
}

TEST(FileSystemTest, isDirectory_yes_trailingSlash)
{
    EXPECT_TRUE(FileSystem::isDirectory(getPath("non-empty-directory/")));
}

TEST(FileSystemTest, isDirectory_nope)
{
    EXPECT_FALSE(FileSystem::isDirectory(getPath("non-empty-directory/fileA.txt")));
}
