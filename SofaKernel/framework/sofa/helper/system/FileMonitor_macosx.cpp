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
// TO BE DONE USING THE SPECIFIC API... OR FIND A CROSS PLATEFORM LIBRARY.

// let's use FSEvent API for this...
// https://developer.apple.com/library/mac/documentation/Darwin/Reference/FSEvents_Ref/
// https://developer.apple.com/library/content/documentation/Darwin/Conceptual/FSEvents_ProgGuide/UsingtheFSEventsFramework/UsingtheFSEventsFramework.html

#include "FileSystem.h"
using sofa::helper::system::FileSystem ;

#include "FileMonitor.h"

using namespace std ;


//////////////////// C++ Header ///////////////////////////////////////////////
#include <list>
#include <string>
#include <map>
#include <sys/time.h>

//////////////////// OSX Header ///////////////////////////////////////////////
//#include <sys/attr.h>
//#include <unistd.h>
//#include <sys/vnode.h>
//#include <ctime>

#include <CoreServices/CoreServices.h>
#include <iostream>

namespace sofa
{

namespace helper
{

namespace system
{

class MonitoredFile
{
public:
    MonitoredFile(const string &filename, FileEventListener* listener)
    {
        m_filename = filename;
        m_listener = listener;

        FSEventStreamContext context = {0};
        context.info = this;

        CFStringRef filename_str = CFStringCreateWithCString(
                                kCFAllocatorDefault,
                                filename.c_str(),
                                kCFStringEncodingUTF8);
        CFArrayRef paths = CFArrayCreate(NULL, (const void**)&filename_str, 1, NULL);
/*
- The application creates a stream by calling FSEventStreamCreate or FSEventStreamCreateRelativeToDevice.
The application schedules the stream on the run loop by calling FSEventStreamScheduleWithRunLoop.
The application tells the file system events daemon to start sending events by calling FSEventStreamStart.
The application services events as they arrive. The API posts events by calling the callback function specified in step 1.
The application tells the daemon to stop sending events by calling FSEventStreamStop.
If the application needs to restart the stream, go to step 3.
The application unschedules the event from its run loop by calling FSEventStreamUnscheduleFromRunLoop.
The application invalidates the stream by calling FSEventStreamInvalidate.
The application releases its reference to the stream by calling FSEventStreamRelease.
These steps are explained in more detail in the sections that follow.
 */
        m_eventStream = FSEventStreamCreate(NULL, // CFAllocatorRef allocator
                                          &eventCallback, // FSEventStreamCallback callback
                                          &context, // FSEventStreamContext *context
                                          paths, //CFArrayRef pathsToWatch
                                          kFSEventStreamEventIdSinceNow, //FSEventStreamEventId sinceWhen
                                          0.2, // CFTimeInterval latency
                                          kFSEventStreamCreateFlagFileEvents // FSEventStreamCreateFlags flags
                                          );
        FSEventStreamScheduleWithRunLoop(m_eventStream,CFRunLoopGetCurrent(), kCFRunLoopDefaultMode);
        FSEventStreamStart(m_eventStream);
    }

    ~MonitoredFile()
    {
        FSEventStreamStop(m_eventStream);
        FSEventStreamInvalidate(m_eventStream);
        FSEventStreamRelease(m_eventStream);
    }

    // update hash; returns FALSE if file changed
    bool update()
    {
        bool changed = m_changed;
        m_changed = false;
        return (!changed);
    }

private:
    static void eventCallback(ConstFSEventStreamRef streamRef, void *clientCallBackInfo, size_t numEvents, void *eventPaths, const FSEventStreamEventFlags eventFlags[], const FSEventStreamEventId eventIds[])
    {
        MonitoredFile *mf = (MonitoredFile*)clientCallBackInfo;
        mf->m_changed=true;

        //printf("#################################################  eventCallback file=%s\n",mf->m_filename.c_str());
        //printf("... numEvent %d\n", numEvents );
        //fflush(stdout);
    }

    FSEventStreamRef    m_eventStream ;
    bool                m_changed {false} ;

public:
    FileEventListener   *m_listener {nullptr} ;
    string              m_filename ;
};

typedef list<MonitoredFile*> ListOfMonitors ;
ListOfMonitors monitors;

void FileMonitor::removeFileListener(const string& filename,
                                     FileEventListener *listener)
{
    for(ListOfMonitors::iterator it_monitor=monitors.begin(); it_monitor!=monitors.end(); )
        if ((*it_monitor)->m_listener==listener && (*it_monitor)->m_filename==filename)
        {
            delete(*it_monitor);
            it_monitor = monitors.erase(it_monitor);
        }
        else
            it_monitor++;
}


void FileMonitor::removeListener(FileEventListener *listener)
{
    for(ListOfMonitors::iterator it_monitor=monitors.begin(); it_monitor!=monitors.end(); )
        if ((*it_monitor)->m_listener==listener)
        {
            delete(*it_monitor);
            it_monitor = monitors.erase(it_monitor);
        }
        else
            it_monitor++;
}

int FileMonitor::addFile(const std::string& filepath, FileEventListener* listener)
{
    if(!FileSystem::exists(filepath))
        return -1 ;

    for(ListOfMonitors::iterator it_monitor=monitors.begin(); it_monitor!=monitors.end(); it_monitor++)
        if ((*it_monitor)->m_listener==listener && (*it_monitor)->m_filename==filepath)
            return 1;

    monitors.push_back(new MonitoredFile(filepath,listener));

    return 1;
}

int FileMonitor::addFile(const std::string& directoryname, const std::string& filename, FileEventListener* listener)
{
    return addFile(directoryname+filename,listener);
}

/* This flag controls termination of the main loop. */
volatile sig_atomic_t keep_going = 1;

/* The signal handler just clears the flag and re-enables itself. */
void catch_alarm (int sig)
{
//    printf("TIMEOUT!!!!!!!!!!!!!!!!!!\n");
    keep_going = 0;
}

int FileMonitor::updates(int timeout)
{

    if (timeout>0)
    {
        keep_going = 1;
        /* Establish a handler for SIGALRM signals. */
        signal (SIGALRM, catch_alarm);

        /* Set an alarm to go off in a little while. */
        alarm (timeout);
    }
    else
    {
        keep_going = 0;
    }
    /* Check the flag once in a while to see when to quit. */
    // check file changes
    do{
        // update FSEventStreams
        //printf("update streams...\n");
        //fflush(stdout);
        CFRunLoopRunInMode(kCFRunLoopDefaultMode,
                           0,
                           false);
        for(ListOfMonitors::iterator it_monitor=monitors.begin(); it_monitor!=monitors.end(); it_monitor++)
        {
            if (!(*it_monitor)->update())
            {
                //printf("FileListener::fileHasChanged(%s) called...\n",(*it_monitor)->m_filename.c_str());
                (*it_monitor)->m_listener->fileHasChanged((*it_monitor)->m_filename);
                keep_going = 0; // we're done
            }
        }
        if (keep_going) usleep(10000);
    }
    while (keep_going);

    alarm(0);
    return 0;
}


} // system
} // helper
} // sofa
