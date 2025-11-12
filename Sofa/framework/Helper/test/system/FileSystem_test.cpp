/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/testing/config.h>

#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Utils.h>
#include <gtest/gtest.h>
#include <sofa/helper/logging/MessageDispatcher.h>
#include <exception>
#include <algorithm>
#include <fstream>
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;



using sofa::helper::system::FileSystem;

static std::string getPath(std::string s) {
    return std::string(SOFA_TESTING_RESOURCES_DIR) + std::string("/") + s;
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
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    EXPECT_MSG_NOEMIT(Error) ;

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
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    EXPECT_MSG_NOEMIT(Error) ;

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
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    EXPECT_MSG_NOEMIT(Error) ;

    std::vector<std::string> fileList;
    FileSystem::listDirectory(getPath("non-empty-directory/"), fileList, "txt");
    EXPECT_EQ(fileList.size(), 2u);
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileA.txt") != fileList.end());
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileB.txt") != fileList.end());
}

TEST(FileSystemTest, listDirectory_withExtension_oneMatch)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    EXPECT_MSG_NOEMIT(Error) ;

    std::vector<std::string> fileList;
    FileSystem::listDirectory(getPath("non-empty-directory/"), fileList, "so");
    EXPECT_EQ(fileList.size(), 1u);
    EXPECT_TRUE(std::find(fileList.begin(), fileList.end(), "fileC.so") != fileList.end());
}

TEST(FileSystemTest, listDirectory_withExtension_noMatch)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    EXPECT_MSG_NOEMIT(Error) ;

    std::vector<std::string> fileList;
    FileSystem::listDirectory(getPath("non-empty-directory/"), fileList, "h");
    EXPECT_TRUE(fileList.empty());
}

TEST(FileSystemTest, createDirectory)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    EXPECT_MSG_NOEMIT(Error) ;

    EXPECT_FALSE(FileSystem::createDirectory("createDirectoryTestDir"));
    EXPECT_TRUE(FileSystem::exists("createDirectoryTestDir"));
    EXPECT_TRUE(FileSystem::isDirectory("createDirectoryTestDir"));
    // Cleanup
    FileSystem::removeDirectory("createDirectoryTestDir");
}

TEST(FileSystemTest, createDirectory_alreadyExists)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    EXPECT_MSG_NOEMIT(Error) ;

    EXPECT_FALSE(FileSystem::createDirectory("createDirectoryTestDir"));
    EXPECT_TRUE(FileSystem::exists("createDirectoryTestDir"));
    EXPECT_TRUE(FileSystem::isDirectory("createDirectoryTestDir"));
    EXPECT_FALSE(FileSystem::createDirectory("createDirectoryTestDir"));

    // Cleanup
    FileSystem::removeDirectory("createDirectoryTestDir");

}

TEST(FileSystemTest, removeDirectory)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    EXPECT_MSG_NOEMIT(Error) ;

    FileSystem::createDirectory("removeDirectoryTestDir");
    EXPECT_FALSE(FileSystem::removeDirectory("removeDirectoryTestDir"));
    EXPECT_FALSE(FileSystem::exists("removeDirectoryTestDir"));
}

TEST(FileSystemTest, removeDirectory_doesNotExists)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    {
        // this test will raise an error on purpose
        EXPECT_MSG_EMIT(Error) ;
        EXPECT_TRUE(FileSystem::removeDirectory("removeDirectoryTestDir"));
    }
    {
        EXPECT_MSG_NOEMIT(Error) ;
        EXPECT_FALSE(FileSystem::exists("removeDirectoryTestDir"));
    }
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

TEST(FileSystemTest, isFile_yes)
{
    // Absolute path
    EXPECT_TRUE(FileSystem::isFile(getPath("non-empty-directory/fileA.txt")));

    // Relative path
    std::ofstream ofs ("FileSystemTest_isFile_yes.txt", std::ofstream::out);
    ofs.close();
    EXPECT_TRUE(FileSystem::isFile("FileSystemTest_isFile_yes.txt"));
    std::remove("FileSystemTest_isFile_yes.txt");
    EXPECT_FALSE(FileSystem::isFile("FileSystemTest_isFile_yes.txt"));

}

TEST(FileSystemTest, isFile_nope)
{
    // Absolute path
    EXPECT_FALSE(FileSystem::isFile(getPath("non-empty-directory")));

    // Relative path
    FileSystem::createDirectory("FileSystemTest_isFile_no");
    EXPECT_FALSE(FileSystem::isFile("FileSystemTest_isFile_no"));
    FileSystem::removeDirectory("FileSystemTest_isFile_no");
}

TEST(FileSystemTest, isAbsolute)
{
    EXPECT_FALSE(FileSystem::isAbsolute(""));
    EXPECT_FALSE(FileSystem::isAbsolute("abc"));
    EXPECT_FALSE(FileSystem::isAbsolute("abc/def"));
    EXPECT_TRUE(FileSystem::isAbsolute("/"));
    EXPECT_TRUE(FileSystem::isAbsolute("/abc"));
    EXPECT_TRUE(FileSystem::isAbsolute("/abc/"));
    EXPECT_TRUE(FileSystem::isAbsolute("/abc/def"));
    EXPECT_TRUE(FileSystem::isAbsolute("A:/"));
    EXPECT_TRUE(FileSystem::isAbsolute("B:/abc"));
    EXPECT_TRUE(FileSystem::isAbsolute("C:/abc/"));
    EXPECT_TRUE(FileSystem::isAbsolute("D:/abc/def"));
}

TEST(FileSystemTest, cleanPath)
{
    EXPECT_EQ("", FileSystem::cleanPath(""));
    EXPECT_EQ("/abc/def/ghi/jkl/mno", FileSystem::cleanPath("/abc/def//ghi/jkl///mno"));
    EXPECT_EQ("C:/abc/def/ghi/jkl/mno", FileSystem::cleanPath("C:\\abc\\def\\ghi/jkl///mno"));
}

TEST(FileSystemTest, convertBackSlashesToSlashes)
{
    EXPECT_EQ("", FileSystem::convertBackSlashesToSlashes(""));
    EXPECT_EQ("abc/def/ghi", FileSystem::convertBackSlashesToSlashes("abc/def/ghi"));
    EXPECT_EQ("abc/def/ghi", FileSystem::convertBackSlashesToSlashes("abc/def\\ghi"));
    EXPECT_EQ("abc/def/ghi", FileSystem::convertBackSlashesToSlashes("abc\\def\\ghi"));
    EXPECT_EQ("C:/abc/def/ghi", FileSystem::convertBackSlashesToSlashes("C:\\abc\\def\\ghi"));
    EXPECT_EQ("C:/abc/def/ghi", FileSystem::convertBackSlashesToSlashes("C:\\abc\\def/ghi"));
}

TEST(FileSystemTest, removeExtraSlashes)
{
    EXPECT_EQ("", FileSystem::removeExtraSlashes(""));
    EXPECT_EQ("/", FileSystem::removeExtraSlashes("/"));
    EXPECT_EQ("/", FileSystem::removeExtraSlashes("//"));
    EXPECT_EQ("/", FileSystem::removeExtraSlashes("///"));
    EXPECT_EQ("/", FileSystem::removeExtraSlashes("////"));
    EXPECT_EQ("/", FileSystem::removeExtraSlashes("/////"));
    EXPECT_EQ("/", FileSystem::removeExtraSlashes("//////"));
    EXPECT_EQ("/abc/def", FileSystem::removeExtraSlashes("/abc/def"));
    EXPECT_EQ("/abc/def/", FileSystem::removeExtraSlashes("/abc/def/"));
    EXPECT_EQ("/abc/def/ghi/jkl/", FileSystem::removeExtraSlashes("/abc//def//ghi/jkl///"));
}

TEST(FileSystemTest, getParentDirectory)
{
    EXPECT_EQ("/abc/def", FileSystem::getParentDirectory("/abc/def/ghi"));
    EXPECT_EQ("/abc/def", FileSystem::getParentDirectory("/abc/def/ghi/"));
    EXPECT_EQ("/", FileSystem::getParentDirectory("/abc/"));
    EXPECT_EQ("/", FileSystem::getParentDirectory("/abc"));
    EXPECT_EQ("/", FileSystem::getParentDirectory("/"));
    EXPECT_EQ(".", FileSystem::getParentDirectory("."));
    EXPECT_EQ(".", FileSystem::getParentDirectory(""));

    EXPECT_EQ("abc/def", FileSystem::getParentDirectory("abc/def/ghi"));
    EXPECT_EQ("abc/def", FileSystem::getParentDirectory("abc/def/ghi/"));
    EXPECT_EQ("abc/def", FileSystem::getParentDirectory("abc/def/ghi//"));
    EXPECT_EQ("abc/def", FileSystem::getParentDirectory("abc/def/ghi///"));
    EXPECT_EQ("abc/def", FileSystem::getParentDirectory("abc/def/ghi////"));
    EXPECT_EQ("abc", FileSystem::getParentDirectory("abc/def"));
    EXPECT_EQ(".", FileSystem::getParentDirectory("abc"));

    EXPECT_EQ("C:/abc/def", FileSystem::getParentDirectory("C:/abc/def/ghi"));
    EXPECT_EQ("C:/abc/def", FileSystem::getParentDirectory("C:/abc/def/ghi/"));
    EXPECT_EQ("C:/", FileSystem::getParentDirectory("C:/abc/"));
    EXPECT_EQ("C:/", FileSystem::getParentDirectory("C:/abc"));
    EXPECT_EQ("C:/", FileSystem::getParentDirectory("C:///"));
    EXPECT_EQ("C:/", FileSystem::getParentDirectory("C://"));
    EXPECT_EQ("C:/", FileSystem::getParentDirectory("C:/"));
}

TEST(FileSystemTest, stripDirectory)
{
    EXPECT_EQ("", FileSystem::stripDirectory(""));
    EXPECT_EQ("/", FileSystem::stripDirectory("/"));
    EXPECT_EQ("/", FileSystem::stripDirectory("C:/"));
    EXPECT_EQ("abc", FileSystem::stripDirectory("abc"));
    EXPECT_EQ("abc", FileSystem::stripDirectory("/abc"));
    EXPECT_EQ("abc", FileSystem::stripDirectory("/abc/"));
    EXPECT_EQ("abc", FileSystem::stripDirectory("C:/abc"));
    EXPECT_EQ("abc", FileSystem::stripDirectory("C:/abc/"));
    EXPECT_EQ("def", FileSystem::stripDirectory("abc/def"));
    EXPECT_EQ("def", FileSystem::stripDirectory("/abc/def"));
    EXPECT_EQ("def", FileSystem::stripDirectory("/abc/def/"));
    EXPECT_EQ("def", FileSystem::stripDirectory("C:/abc/def"));
    EXPECT_EQ("def", FileSystem::stripDirectory("C:/abc/def/"));
    EXPECT_EQ("ghi", FileSystem::stripDirectory("/abc/def/ghi"));
    EXPECT_EQ("ghi", FileSystem::stripDirectory("/abc/def/ghi/"));
    EXPECT_EQ("ghi", FileSystem::stripDirectory("abc/def/ghi"));
    EXPECT_EQ("ghi", FileSystem::stripDirectory("abc/def/ghi/"));
    EXPECT_EQ("ghi", FileSystem::stripDirectory("C:/abc/def/ghi"));
    EXPECT_EQ("ghi", FileSystem::stripDirectory("C:/abc/def/ghi/"));
}

TEST(FileSystemTest, append)
{
    EXPECT_EQ(FileSystem::append("C:/", "folder"), "C:/folder");
    EXPECT_EQ(FileSystem::append("C:", "folder"), "C:/folder");
    EXPECT_EQ(FileSystem::append("C:\\", "folder"), "C:/folder");

    EXPECT_EQ(FileSystem::append("", "folder"), "/folder");

    EXPECT_EQ(FileSystem::append("a/b/c/d", ""), "a/b/c/d");
    EXPECT_EQ(FileSystem::append("a/b/c/d", "/folder"), "a/b/c/d/folder");
    EXPECT_EQ(FileSystem::append("a/b/c/d/", "/folder"), "a/b/c/d/folder");
    EXPECT_EQ(FileSystem::append("a/b/c/d//", "/folder"), "a/b/c/d/folder");

    EXPECT_EQ(FileSystem::append("a/b/c/d", "e", "f", "g"), "a/b/c/d/e/f/g");
    EXPECT_EQ(FileSystem::append("a/b/c/d/", "e", "f", "g"), "a/b/c/d/e/f/g");
    EXPECT_EQ(FileSystem::append("a/b/c/d/", "e", "/f", "g"), "a/b/c/d/e/f/g");
    EXPECT_EQ(FileSystem::append("a/b/c/d/", "e", "/f", "/g"), "a/b/c/d/e/f/g");
    EXPECT_EQ(FileSystem::append("a/b/c/d/", "/e", "/f", "/g"), "a/b/c/d/e/f/g");
}

TEST(FileSystemTest, ensureFolderExists)
{
    ASSERT_TRUE(FileSystem::isDirectory(sofa::helper::Utils::getSofaPathPrefix()));

    const auto parentDir = FileSystem::append(sofa::helper::Utils::getSofaPathPrefix(), "test_folder");
    const auto dir = FileSystem::append(parentDir, "another_layer");

    //the folders don't exist yet
    EXPECT_FALSE(FileSystem::isDirectory(parentDir));
    EXPECT_FALSE(FileSystem::isDirectory(dir));

    FileSystem::ensureFolderExists(dir);

    EXPECT_TRUE(FileSystem::isDirectory(parentDir));
    EXPECT_TRUE(FileSystem::isDirectory(dir));

    //cleanup
    EXPECT_FALSE(FileSystem::removeDirectory(dir));
    EXPECT_FALSE(FileSystem::removeDirectory(parentDir));
}

TEST(FileSystemTest, ensureFolderForFileExists_fileAndFolderDontExist)
{
    ASSERT_TRUE(FileSystem::isDirectory(sofa::helper::Utils::getSofaPathPrefix()));

    const auto parentDir = FileSystem::append(sofa::helper::Utils::getSofaPathPrefix(), "test_folder");
    const auto dir = FileSystem::append(parentDir, "another_layer");

    //the folder does not exist yet
    EXPECT_FALSE(FileSystem::isDirectory(dir));

    const auto file = FileSystem::append(dir, "file.txt");
    FileSystem::ensureFolderForFileExists(file);

    EXPECT_TRUE(FileSystem::isDirectory(dir));
    EXPECT_FALSE(FileSystem::isDirectory(file));

    EXPECT_FALSE(FileSystem::exists(file));

    //cleanup
    EXPECT_FALSE(FileSystem::removeDirectory(dir));
    EXPECT_FALSE(FileSystem::removeDirectory(parentDir));
}

TEST(FileSystemTest, ensureFolderForFileExists_fileExist)
{
    ASSERT_TRUE(FileSystem::isDirectory(sofa::helper::Utils::getSofaPathPrefix()));

    const auto file = FileSystem::append(sofa::helper::Utils::getSofaPathPrefix(), "file.txt");
    EXPECT_FALSE(FileSystem::exists(file));

    std::ofstream fileStream;
    fileStream.open(file);
    fileStream << "Hello";
    fileStream.close();

    EXPECT_TRUE(FileSystem::exists(file));

    FileSystem::ensureFolderForFileExists(file);

    EXPECT_TRUE(FileSystem::exists(file));

    //cleanup
    EXPECT_TRUE(FileSystem::removeFile(file));
}