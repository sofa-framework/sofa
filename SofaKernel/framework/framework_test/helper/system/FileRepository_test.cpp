
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
