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

#include <sofa/helper/Utils.h>
#include <gtest/gtest.h>

#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/system/Locale.h>

using sofa::helper::Utils;
using sofa::helper::system::FileSystem;

TEST(UtilsTest, getExecutablePath)
{
    const std::string path = Utils::getExecutablePath();
    EXPECT_TRUE(path.find("Sofa.Helper_test") != std::string::npos);
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

// Following tests can fail (or not really relevant)
// if the user has a custom/non-standard home directory
// (moreso if the user does not have a home directory or is being disabled for security reason)
bool testGetUserLocalDirectory()
{
    bool result = true;

    const std::string path = Utils::getUserLocalDirectory();
#if defined(WIN32)
    result = result && (path.find("AppData") != std::string::npos);
    result = result && (path.find("Local") != std::string::npos);
#elif defined (__APPLE__)
    result = result && (path.find("Library") != std::string::npos);
    result = result && (path.find("Application Support") != std::string::npos);
#else // Linux
    result = result && (path.find(".config") != std::string::npos);
#endif

    return result;
}

TEST(UtilsTest, getUserLocalDirectory)
{
    EXPECT_TRUE(testGetUserLocalDirectory());
}

TEST(UtilsTest, getSofaUserLocalDirectory)
{
    const std::string path = Utils::getSofaUserLocalDirectory();
    EXPECT_TRUE(testGetUserLocalDirectory());
    EXPECT_TRUE(path.find("SOFA") != std::string::npos);
}

TEST(UtilsTest, readBasicIniFile_nonexistentFile)
{
    // this test will raise an error on purpose
    const std::map<std::string, std::string> values = Utils::readBasicIniFile("this-file-does-not-exist");
    EXPECT_TRUE(values.empty());
}

TEST(UtilsTest, readBasicIniFile)
{
    const std::string path = std::string(SOFA_TESTING_RESOURCES_DIR) + "/UtilsTest.ini";
    std::map<std::string, std::string> values = Utils::readBasicIniFile(path);
    EXPECT_EQ(3u, values.size());
    EXPECT_EQ(1u, values.count("a"));
    EXPECT_EQ("b again", values["a"]);
    EXPECT_EQ(1u, values.count("someKey"));
    EXPECT_EQ("someValue", values["someKey"]);
    EXPECT_EQ(1u, values.count("foo bar baz"));
    EXPECT_EQ("qux 42", values["foo bar baz"]);
}
