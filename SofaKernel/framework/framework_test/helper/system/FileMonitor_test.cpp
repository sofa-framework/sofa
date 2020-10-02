 #include <gtest/gtest.h>
#include <exception>
#include <algorithm>

#include <vector>
using std::vector ;

#include <fstream>
using std::ofstream ;

#include <string>
using std::string ;

#include <sofa/helper/system/FileMonitor.h>
using sofa::helper::system::FileEventListener ;
using sofa::helper::system::FileMonitor ;

#ifdef WIN32
#include <windows.h>
#endif

static std::string getPath(std::string s) {
    return std::string(FRAMEWORK_TEST_RESOURCES_DIR) + std::string("/") + s;
}

void createAFilledFile(const string filename, unsigned int rep){
    ofstream file1 ;
    file1.open(filename.c_str(), ofstream::out) ;

    //throw_when(!file1.is_open()) ;

    string sample = "#include<TODOD> int main(int argc...){ ... }\n}" ;
    for(unsigned int i=0;i<rep;i++){
        file1.write(sample.c_str(), sample.size()) ;
    }
    file1.close();
}

void waitForFileEvents()
{
	// on windows we use file date, which resoution is assumed (by us) to be below this value in ms
#ifdef WIN32
	Sleep(100);
#endif
}

class MyFileListener : public FileEventListener
{
public:
    vector<string> m_files ;

    virtual void fileHasChanged(const std::string& filename){
        //std::cout << "FileHasChanged: " << filename << std::endl ;
        m_files.push_back(filename) ;
    }
};

TEST(FileMonitor, addFileNotExist_test)
{
    MyFileListener listener ;

    // Should refuse to add a file that does not exists
     EXPECT_EQ( FileMonitor::addFile(getPath("nonexisting.txt"), &listener), -1 ) ;

     FileMonitor::removeListener(&listener) ;
}

TEST(FileMonitor, addFileNotExist2_test)
{
    MyFileListener listener ;

    // Should refuse to add a file that does not exists
     EXPECT_EQ( FileMonitor::addFile(getPath(""),"nonexisting.txt", &listener), -1 ) ;

     FileMonitor::removeListener(&listener) ;
}

TEST(FileMonitor, addFileExist_test)
{
    MyFileListener listener ;

    // Add an existing file.It should work.
    EXPECT_EQ( FileMonitor::addFile(getPath("existing.txt"), &listener), 1 ) ;

    FileMonitor::removeListener(&listener) ;
}

TEST(FileMonitor, addFileTwice_test)
{
    MyFileListener listener ;

    // Add an existing file.It should work.
    FileMonitor::addFile(getPath("existing.txt"), &listener);

    // Retry to add an existing file. It should fail.
    EXPECT_EQ( FileMonitor::addFile(getPath("existing.txt"), &listener), 1 ) ;

    // change the file content..
    createAFilledFile(getPath("existing.txt"), 10) ;

    waitForFileEvents();
    FileMonitor::updates(2) ;

    // The listener should be notified 1 times with the same event.
    EXPECT_EQ( listener.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener) ;
}

TEST(FileMonitor, noUpdate_test)
{
    MyFileListener listener ;

    // Add an existing file.It should work.
    FileMonitor::addFile(getPath("existing.txt"), &listener) ;
    EXPECT_EQ( listener.m_files.size(), 0u) ;

    FileMonitor::removeListener(&listener) ;
}

TEST(FileMonitor, updateNoChange_test)
{
    MyFileListener listener ;

    FileMonitor::addFile(getPath("existing.txt"), &listener) ;
    waitForFileEvents();
    FileMonitor::updates(2) ;
    EXPECT_EQ( listener.m_files.size(), 0u) ;

    FileMonitor::removeListener(&listener) ;
}

TEST(FileMonitor, fileChange_test)
{
    MyFileListener listener ;

    FileMonitor::addFile(getPath("existing.txt"), &listener) ;
    //waitForFileEvents();
    //FileMonitor::updates(2) ;

    // change the file content..
    createAFilledFile(getPath("existing.txt"), 10) ;
    waitForFileEvents();
    FileMonitor::updates(2) ;
    EXPECT_EQ( listener.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener) ;
}

TEST(FileMonitor, fileChangeTwice_test)
{
    MyFileListener listener ;

    FileMonitor::addFile(getPath("existing.txt"), &listener) ;
    //FileMonitor::updates(2) ;

    // change the file content 2x to test if the events are coalesced.
    listener.m_files.clear() ;
    createAFilledFile(getPath("existing.txt"), 100) ;
    createAFilledFile(getPath("existing.txt"), 200) ;

    waitForFileEvents();
    FileMonitor::updates(2) ;
    EXPECT_EQ( listener.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener) ;
}

TEST(FileMonitor, fileListenerRemoved_test)
{
    MyFileListener listener1 ;
    MyFileListener listener2 ;

    FileMonitor::addFile(getPath("existing.txt"), &listener1) ;
    FileMonitor::addFile(getPath("existing.txt"), &listener2) ;
    //FileMonitor::updates(2) ;

    // change the file content 2x to test if the events are coalesced.
    listener1.m_files.clear() ;
    listener2.m_files.clear() ;
    createAFilledFile(getPath("existing.txt"), 200) ;

    FileMonitor::removeFileListener(getPath("existing.txt"), &listener1) ;

    waitForFileEvents();
    FileMonitor::updates(2) ;
    EXPECT_EQ( listener1.m_files.size(), 0u) ;
    EXPECT_EQ( listener2.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener1) ;
    FileMonitor::removeListener(&listener2) ;
}

TEST(FileMonitor, listenerRemoved_test)
{
    MyFileListener listener1 ;
    MyFileListener listener2 ;

    FileMonitor::addFile(getPath("existing.txt"), &listener1) ;
    FileMonitor::addFile(getPath("existing.txt"), &listener2) ;
    //FileMonitor::updates(2) ;

    // change the file content 2x to test if the events are coalesced.
    listener1.m_files.clear() ;
    listener2.m_files.clear() ;
    createAFilledFile(getPath("existing.txt"), 200) ;

    FileMonitor::removeListener(&listener1) ;

    waitForFileEvents();
    FileMonitor::updates(2) ;
    EXPECT_EQ( listener1.m_files.size(), 0u) ;
    EXPECT_EQ( listener2.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener1) ;
    FileMonitor::removeListener(&listener2) ;
}

TEST(FileMonitor, fileChange2_test)
{
    MyFileListener listener ;

    FileMonitor::addFile(getPath(""),"existing.txt", &listener) ;
    //waitForFileEvents();
    //FileMonitor::updates(2) ;

    // change the file content..
    createAFilledFile(getPath("existing.txt"), 10) ;

    waitForFileEvents();
    FileMonitor::updates(2) ;
    EXPECT_EQ( listener.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener) ;
}
