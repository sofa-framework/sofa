// TO BE DONE USING THE SPECIFIC API... OR FIND A CROSS PLATEFORM LIBRARY.

// let's use FSEvent API for this...
// https://developer.apple.com/library/mac/documentation/Darwin/Reference/FSEvents_Ref/

//#include <errno.h>       // for errno
//#include <fcntl.h>       // for O_RDONLY
//#include <stdio.h>       // for fprintf()
//#include <stdlib.h>      // for EXIT_SUCCESS
//#include <string.h>      // for strerror()
//#include <sys/event.h>   // for kqueue() etc.
//#include <unistd.h>      // for close()
#include <CoreServices/CoreServices.h>



#include "FileSystem.h"
using sofa::helper::system::FileSystem ;

#include "FileMonitor.h"

namespace sofa
{

namespace helper
{

namespace system
{

FSEventStreamRef streamRef = 0;

FSEventStreamRef FileMonitor_init()
{
    if (streamRef!=0)
        return streamRef;

 //   streamRef = FSEventStreamCreate();

    return streamRef;
}


int FileMonitor::addFile(const std::string& filepath, FileEventListener* listener)
{
    if(!FileSystem::exists(filepath))
        return -1 ;

    // TODO
    return 0;
}

int FileMonitor::addFile(const std::string& directoryname, const std::string& filename, FileEventListener* listener)
{
    // TODO
    return 0;
}

int FileMonitor::updates(int timeout)
{
    // TODO
    return 0;
}


} // system
} // helper
} // sofa
