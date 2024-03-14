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
#include <gtest/gtest.h>
#include <exception>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cstdio>

#include <vector>
using std::vector ;

#include <fstream>
using std::ofstream ;

#include <string>
using std::string ;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem ;

#include <sofa/helper/system/FileMonitor.h>
using sofa::helper::system::FileEventListener ;
using sofa::helper::system::FileMonitor ;

#include <cmath>


#include <sofa/helper/system/thread/CTime.h>
using sofa::helper::system::thread::CTime ;
using sofa::helper::system::thread::ctime_t ;


#ifdef WIN32
#include <windows.h>
#endif

static std::string getPath(std::string s) {
    return std::string(SOFA_TESTING_RESOURCES_DIR) + std::string("/") + s;
}

void createAFilledFile(const string filename, unsigned int rep, bool resetFileMonitor=true){
    ofstream file1 ;
    file1.open(filename.c_str(), ofstream::out) ;

    const string sample = "#include<TODOD> int main(int argc...){ ... }\n}" ;
    for(unsigned int i=0;i<rep;i++){
        file1.write(sample.c_str(), sample.size()) ;
    }
    file1.flush();
    file1.close();

    // dirty fix to avoid interferences between successive tests using the same file
    if (resetFileMonitor)
        FileMonitor::updates(1);
}

void waitUntilFileExists(const std::string& filename, double timeout)
{
    const ctime_t time = CTime::getTime() ;
    while( !FileSystem::exists(filename)
           && CTime::toSecond(CTime::getTime()-time) < timeout ){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void waitABit()
{
    // on windows we use file date, which resoution is assumed (by us) to be below this value in ms
#ifdef WIN32
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
#endif
#ifdef __APPLE__
    // std::this_thread::sleep_for(1);
#endif
#ifdef __linux__
    //  std::this_thread::sleep_for(1);
#endif
}

class MyFileListener : public FileEventListener
{
public:
    vector<string> m_files ;

    void fileHasChanged(const std::string& filename) override
    {
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

    // create the file
    const std::string filepath = getPath("existing.txt");
    createAFilledFile(filepath, 1) ;
    waitABit();

    // Add an existing file.It should work.
    EXPECT_EQ( FileMonitor::addFile(filepath, &listener), 1 ) ;

    FileMonitor::removeListener(&listener) ;

    // remove file
    std::remove(filepath.c_str());
}

TEST(FileMonitor, addFileTwice_test)
{
    MyFileListener listener ;

    // create the file
    const std::string filepath = getPath("existing.txt");
    createAFilledFile(filepath, 1) ;
    waitABit();

    // Add an existing file.It should work.
    FileMonitor::addFile(filepath, &listener);

    // Retry to add an existing file. It should fail.
    EXPECT_EQ( FileMonitor::addFile(filepath, &listener), 1 ) ;

    // change the file content..
    createAFilledFile(filepath, 10) ;
    FileMonitor::updates(2) ;

    // The listener should be notified 1 times with the same event.
    EXPECT_EQ( listener.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener) ;

    // remove file
    std::remove(filepath.c_str());
}

TEST(FileMonitor, noUpdate_test)
{
    MyFileListener listener ;
    const std::string filepath = getPath("existing.txt");

    // create the file
    createAFilledFile(filepath, 1) ;
    waitABit();

    // Add an existing file.It should work.
    FileMonitor::addFile(filepath, &listener) ;
    EXPECT_EQ( listener.m_files.size(), 0u) ;

    FileMonitor::removeListener(&listener) ;

    // remove file
    std::remove(filepath.c_str());
}

TEST(FileMonitor, updateNoChange_test)
{
    MyFileListener listener ;

    // create the file
    const std::string filepath = getPath("existing.txt");
    createAFilledFile(filepath, 1) ;
    waitABit();

    FileMonitor::addFile(filepath, &listener) ;
    FileMonitor::updates(1) ;

    EXPECT_EQ( listener.m_files.size(), 0u) ;

    FileMonitor::removeListener(&listener) ;

    // remove file
    std::remove(filepath.c_str());
}

TEST(FileMonitor, fileChange_test)
{
    MyFileListener listener ;

    // create the file
    const std::string filepath = getPath("existing.txt");
    createAFilledFile(filepath, 1) ;
    waitABit();

    FileMonitor::addFile(filepath, &listener) ;

    // change the file content..
    createAFilledFile(filepath, 10,false) ;
    FileMonitor::updates(2) ;
    EXPECT_EQ( listener.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener) ;

    // remove file
    std::remove(filepath.c_str());
}

TEST(FileMonitor, fileChangeTwice_test)
{
    MyFileListener listener ;

    // create the file
    const std::string filepath = getPath("existing.txt");
    createAFilledFile(filepath, 1) ;
    waitABit();

    FileMonitor::addFile(filepath, &listener) ;

    // change the file content 2x to test if the events are coalesced.
    listener.m_files.clear() ;
    createAFilledFile(filepath, 100,false) ;
    createAFilledFile(filepath, 200,false) ;

    FileMonitor::updates(2) ;
    EXPECT_EQ( listener.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener) ;

    // remove file
    std::remove(filepath.c_str());
}

TEST(FileMonitor, fileListenerRemoved_test)
{
    MyFileListener listener1 ;
    MyFileListener listener2 ;

    // create the file
    const std::string filepath = getPath("existing.txt");
    createAFilledFile(filepath, 1) ;
    waitABit();

    FileMonitor::addFile(filepath, &listener1) ;
    FileMonitor::addFile(filepath, &listener2) ;

    // change the file content 2x to test if the events are coalesced.
    listener1.m_files.clear() ;
    listener2.m_files.clear() ;
    createAFilledFile(filepath, 200, false) ;

    FileMonitor::removeFileListener(filepath, &listener1) ;

    FileMonitor::updates(2) ;
    EXPECT_EQ( listener1.m_files.size(), 0u) ;
    EXPECT_EQ( listener2.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener1) ;
    FileMonitor::removeListener(&listener2) ;

    // remove file
    std::remove(filepath.c_str());
}

TEST(FileMonitor, listenerRemoved_test)
{
    MyFileListener listener1 ;
    MyFileListener listener2 ;

    // create the file
    const std::string filepath = getPath("existing.txt");
    createAFilledFile(filepath, 1) ;
    waitABit();

    FileMonitor::addFile(filepath, &listener1) ;
    FileMonitor::addFile(filepath, &listener2) ;

    // change the file content 2x to test if the events are coalesced.
    listener1.m_files.clear() ;
    listener2.m_files.clear() ;
    createAFilledFile(filepath, 200,false) ;

    FileMonitor::removeListener(&listener1) ;

    FileMonitor::updates(2) ;
    EXPECT_EQ( listener1.m_files.size(), 0u) ;
    EXPECT_EQ( listener2.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener1) ;
    FileMonitor::removeListener(&listener2) ;

    // remove file
    std::remove(filepath.c_str());
}

TEST(FileMonitor, fileChange2_test)
{
    MyFileListener listener ;

    // create the file
    const std::string filepath = getPath("existing.txt");
    createAFilledFile(filepath, 1) ;
    waitABit();

    FileMonitor::addFile(getPath(""), "existing.txt", &listener) ;

    // change the file content..
    createAFilledFile(filepath, 10,false) ;

    FileMonitor::updates(2) ;
    EXPECT_EQ( listener.m_files.size(), 1u) ;

    FileMonitor::removeListener(&listener) ;

    // remove file
    std::remove(filepath.c_str());
}
