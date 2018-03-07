/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "FileMonitor.h"

//TIP: use GetFileAttributesEx for windows ?

#include "FileSystem.h"
using sofa::helper::system::FileSystem;

using namespace std;

#include <sofa/helper/system/thread/CTime.h>
using sofa::helper::system::thread::CTime ;
using sofa::helper::system::thread::ctime_t ;

//////////////////// C++ Header ///////////////////////////////////////////////
#include <list>
#include <string>
#include <map>

//////////////////// Windows Header ///////////////////////////////////////////////
#include <windows.h>
#include <FileAPI.h>

namespace sofa
{

namespace helper
{

namespace system
{

static unsigned int getFileHashTimeSize(const string& filename)
{
    if (filename.length() >= MAX_PATH)
        return 0;	// erreur !

    // convert filename from ascii to wchar...
    WCHAR unicodeFilename[MAX_PATH];
    MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, filename.c_str(), -1, unicodeFilename, MAX_PATH);

    WIN32_FILE_ATTRIBUTE_DATA fileInfo;
    if (!GetFileAttributesExW(unicodeFilename, GetFileExInfoStandard, &fileInfo))
    {
        return 0;	// erreur !
    }

    SYSTEMTIME st;
    if (!FileTimeToSystemTime(&fileInfo.ftLastWriteTime, &st))
    {
        return 0;	// erreur !
    }

    unsigned int hash = st.wMilliseconds
            + 1000 * st.wSecond
            + 60 * 000 * st.wMinute
            + 60 * 60 * 000 * st.wHour
            + 24 * 60 * 60 * 000 * st.wDay
            + 31 * 24 * 60 * 60 * 000 * st.wMonth
            + 12 * 31 * 24 * 60 * 60 * 000 * st.wYear
            ;

    return hash; // temp

}


class MonitoredFile
{
public:
    MonitoredFile(const string &filename, FileEventListener* listener)
    {
        m_filename = filename;
        m_hashTimeSize = getFileHashTimeSize(filename);
        m_listener = listener;
    }

    // update hash; returns FALSE if file changed
    bool update()
    {
        unsigned int oldHash = m_hashTimeSize;
        m_hashTimeSize = getFileHashTimeSize(m_filename);
        return (oldHash == m_hashTimeSize);
    }

private:
    unsigned int        m_hashTimeSize; // first criteria
    //    unsigned int        m_hashContent;  // subsidiary (slow) criteria
public:
    FileEventListener   *m_listener;
    string              m_filename;
};

typedef list<MonitoredFile> ListOfMonitors;
ListOfMonitors monitors;

void FileMonitor::removeFileListener(const string& filename,
                                     FileEventListener *listener)
{
    for (ListOfMonitors::iterator it_monitor = monitors.begin(); it_monitor != monitors.end(); )
        if (it_monitor->m_listener == listener && it_monitor->m_filename == filename)
            it_monitor = monitors.erase(it_monitor);
        else
            it_monitor++;
}


void FileMonitor::removeListener(FileEventListener *listener)
{
    for (ListOfMonitors::iterator it_monitor = monitors.begin(); it_monitor != monitors.end(); )
        if (it_monitor->m_listener == listener)
            it_monitor = monitors.erase(it_monitor);
        else
            it_monitor++;
}

int FileMonitor::addFile(const std::string& filepath, FileEventListener* listener)
{
    if (!FileSystem::exists(filepath))
        return -1;

    for (ListOfMonitors::iterator it_monitor = monitors.begin(); it_monitor != monitors.end(); it_monitor++)
        if (it_monitor->m_listener == listener && it_monitor->m_filename == filepath)
            return 1;

    monitors.push_back(MonitoredFile(filepath, listener));

    return 1;
}

int FileMonitor::addFile(const std::string& directoryname, const std::string& filename, FileEventListener* listener)
{
    return addFile(directoryname + filename, listener);
}

int FileMonitor::updates(int timeout)
{
    bool hadEvent = false ;
    ctime_t t = CTime::getTime() ;

    while(!hadEvent && CTime::toSecond(CTime::getRefTime()-t) < 1.0*timeout ){
        for (ListOfMonitors::iterator it_monitor = monitors.begin(); it_monitor != monitors.end(); it_monitor++)
        {
            if (!it_monitor->update())
            {
                it_monitor->m_listener->fileHasChanged(it_monitor->m_filename);
                hadEvent = true ;
            }
        }
        if(!hadEvent)
            CTime::sleep(0.05);
    }

    return 0;
}


} // system
} // helper
} // sofa
