/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/helper/system/FileRepository.h>
#include <gtest/gtest.h>

using sofa::helper::system::FileRepository;




struct FileRepository_test: public ::testing::Test
{
    FileRepository fileRepository;

    void SetUp()
    {
        fileRepository.addFirstPath( FRAMEWORK_TEST_RESOURCES_DIR );
    }
};



TEST_F(FileRepository_test, findFile )
{
    std::string filename = "UtilsTest.ini";
    ASSERT_TRUE( fileRepository.findFile(filename) );
}

TEST_F(FileRepository_test, findFileSubDir )
{
    std::string filename = "non-empty-directory/fileA.txt";
    ASSERT_TRUE( fileRepository.findFile(filename) );
}


TEST_F(FileRepository_test, findFileWithSpaces )
{
    std::string filename = "file with spaces.txt";
    ASSERT_TRUE( fileRepository.findFile(filename) );
    ASSERT_TRUE( filename == std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/file with spaces.txt" );

    filename = "dir with spaces/file.txt";
    ASSERT_TRUE( fileRepository.findFile(filename) );
    ASSERT_TRUE( filename == std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/dir with spaces/file.txt" );

    fileRepository.addFirstPath( std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/dir with spaces" );
    filename = "file.txt";
    ASSERT_TRUE( fileRepository.findFile(filename) );
    ASSERT_TRUE( filename == std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/dir with spaces/file.txt" );
}

TEST_F(FileRepository_test, findFileWithAccents )
{
    std::string filename = "file è with é accents à.txt";
    ASSERT_TRUE( fileRepository.findFile(filename) );
    ASSERT_TRUE( filename == std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/file è with é accents à.txt" );

    filename = "dir_é_with_è_accents_à/file.txt";
    ASSERT_TRUE( fileRepository.findFile(filename) );
    ASSERT_TRUE( filename == std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/dir_é_with_è_accents_à/file.txt" );

    fileRepository.addFirstPath( std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/dir_é_with_è_accents_à" );
    filename = "file.txt";
    ASSERT_TRUE( fileRepository.findFile(filename) );
    ASSERT_TRUE( filename == std::string(FRAMEWORK_TEST_RESOURCES_DIR) + "/dir_é_with_è_accents_à/file.txt" );
}
