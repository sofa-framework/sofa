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
#include <list>
#include <string>
#include <map>

//////////////////// OSX Header ///////////////////////////////////////////////
#include <sys/attr.h>
#include <unistd.h>
#include <sys/vnode.h>
#include <ctime>

namespace sofa
{

namespace helper
{

namespace system
{

    static unsigned int getFileHash(const string& filename)
    {
        typedef struct attrlist attrlist_t;

        struct FInfoAttrBuf {
            u_int32_t       length;
            fsobj_type_t    objType;    //ATTR_CMN_OBJTYPE
            timespec        timeSpec;   //ATTR_CMN_MODTIME
            //char            finderInfo[32]; //ATTR_CMN_FNDRINFO
            off_t           totalSize;  //ATTR_FILE_TOTALSIZE
        }  __attribute__((aligned(4), packed));
        typedef struct FInfoAttrBuf FInfoAttrBuf;

        int             err;
        attrlist_t      attrList;
        FInfoAttrBuf    attrBuf;

        memset(&attrList, 0, sizeof(attrList));
        attrList.bitmapcount = ATTR_BIT_MAP_COUNT;
        attrList.commonattr  = ATTR_CMN_OBJTYPE | ATTR_CMN_MODTIME;
        attrList.fileattr  = ATTR_FILE_TOTALSIZE;

        err = getattrlist(filename.c_str(), &attrList, &attrBuf, sizeof(attrBuf), 0);
        if (err != 0) {
           // printf("FileStatus(%s) ERROR\n");
        }

        unsigned int result(0);

        if (err == 0) {
//            assert(attrBuf.length == sizeof(attrBuf));
//            printf("Information for %s:\n", filename.c_str());
            // attrBuf.timeSpec.tv_sec est la date du fichier en secondes: good enough ?
//            printf("modification time: %d\n",attrBuf.timeSpec.tv_sec);
//            printf("size: %d\n",attrBuf.totalSize);
/*
            switch (attrBuf.objType) {
                case VREG:
                    printf("file type    = '%.4s' %d-%d-%d-%d\n", swap32(&attrBuf.finderInfo[0]), attrBuf.finderInfo[0],attrBuf.finderInfo[1],attrBuf.finderInfo[2],attrBuf.finderInfo[3]);
                    printf("file creator = '%.4s'\n", swap32(&attrBuf.finderInfo[4]));
                    break;
                case VDIR:
                    printf("directory\n");
                    break;
                default:
                    printf("other object type, %d\n", attrBuf.objType);
                    break;
            }
            */

            // this "hash" computation is DIRTY.
            result  = attrBuf.totalSize>0?attrBuf.timeSpec.tv_sec % attrBuf.totalSize : attrBuf.timeSpec.tv_sec;
//            printf("hash: %8X\n",result);
        }

        return result;

    }


class MonitoredFile
{
public:
    MonitoredFile(const string &filename, FileEventListener* listener)
    {
        m_filename = filename;
        m_hash = getFileHash(filename);
        m_listener = listener;
    }

    // update hash; returns FALSE if file changed
    bool update()
    {
        unsigned int oldHash = m_hash;
        m_hash = getFileHash(m_filename);
        if (oldHash!=m_hash)
            printf("MonitoredFile.update : File %s changed\n",m_filename.c_str());
        else
            printf("MonitoredFile.update : File %s not changed\n",m_filename.c_str());
        return (oldHash==m_hash);
    }

private:
    unsigned int        m_hash;
public:
    FileEventListener   *m_listener;
    string              m_filename;
};






/*
typedef vector<string> ListOfFiles ;
typedef vector<FileEventListener*> ListOfListeners ;
map<string, ListOfFiles> dir2files ;
map<int, string> fd2fn ;
map<string, ListOfListeners> file2listener ;
*/
typedef list<MonitoredFile> ListOfMonitors ;
ListOfMonitors monitors;

void FileMonitor::removeFileListener(const string& filename,
                                     FileEventListener *listener)
{
    for(ListOfMonitors::iterator it_monitor=monitors.begin(); it_monitor!=monitors.end(); )
        if (it_monitor->m_listener==listener && it_monitor->m_filename==filename)
            it_monitor = monitors.erase(it_monitor);
        else
            it_monitor++;
}


void FileMonitor::removeListener(FileEventListener *listener)
{
    for(ListOfMonitors::iterator it_monitor=monitors.begin(); it_monitor!=monitors.end(); )
        if (it_monitor->m_listener==listener)
            it_monitor = monitors.erase(it_monitor);
        else
            it_monitor++;
}

int FileMonitor::addFile(const std::string& filepath, FileEventListener* listener)
{
    if(!FileSystem::exists(filepath))
        return -1 ;

    for(ListOfMonitors::iterator it_monitor=monitors.begin(); it_monitor!=monitors.end(); it_monitor++)
        if (it_monitor->m_listener==listener && it_monitor->m_filename==filepath)
            return 1;

    monitors.push_back(MonitoredFile(filepath,listener));

    return 1;
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
    for(ListOfMonitors::iterator it_monitor=monitors.begin(); it_monitor!=monitors.end(); it_monitor++)
    {
        if (!it_monitor->update())
        {
  //          printf("FileListener::fileHasChanged(%s) called...\n",it_monitor->m_filename.c_str());
            it_monitor->m_listener->fileHasChanged(it_monitor->m_filename);
        }
    }

    return 0;
}


} // system
} // helper
} // sofa
