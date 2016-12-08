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

#include "FileSystem.h"
using sofa::helper::system::FileSystem ;

#include "FileMonitor.h"

using namespace std ;


//////////////////// C++ Header ///////////////////////////////////////////////
#include <vector>
#include <string>
#include <map>



namespace sofa
{

namespace helper
{

namespace system
{

typedef vector<string> ListOfFiles ;
typedef vector<FileEventListener*> ListOfListeners ;
map<string, ListOfFiles> dir2files ;
map<int, string> fd2fn ;
map<string, ListOfListeners> file2listener ;

void FileMonitor::removeFileListener(const string& filename,
                                     FileEventListener *listener)
{

}


void FileMonitor::removeListener(FileEventListener *listener)
{

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
    if(!FileSystem::exists(directoryname+filename))
        return -1 ;

    // TODO
    return 0;
}

int FileMonitor::updates(int timeout)
{
    printf("*************** FileMonitor::updates timeout=%d\n",timeout);

    // TODO
    return 0;
}


} // system
} // helper
} // sofa
