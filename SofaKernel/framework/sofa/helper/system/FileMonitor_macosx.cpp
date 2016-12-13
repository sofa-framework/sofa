// TO BE DONE USING THE SPECIFIC API... OR FIND A CROSS PLATEFORM LIBRARY.

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

    static unsigned int getFileHashTimeSize(const string& filename)
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

        unsigned int result(0);

        if (err == 0) {
            // this "hash" computation is DIRTY.
            result  = attrBuf.totalSize>0?attrBuf.timeSpec.tv_sec % attrBuf.totalSize : attrBuf.timeSpec.tv_sec;
        }

        return result;

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
        return (oldHash==m_hashTimeSize);
    }

private:
    unsigned int        m_hashTimeSize; // first criteria
//    unsigned int        m_hashContent;  // subsidiary (slow) criteria
public:
    FileEventListener   *m_listener;
    string              m_filename;
};

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

    printf("FileMonitor::addFile(%s)\n",filepath.c_str());

    for(ListOfMonitors::iterator it_monitor=monitors.begin(); it_monitor!=monitors.end(); it_monitor++)
        if (it_monitor->m_listener==listener && it_monitor->m_filename==filepath)
            return 1;

    monitors.push_back(MonitoredFile(filepath,listener));

    return 1;
}

int FileMonitor::addFile(const std::string& directoryname, const std::string& filename, FileEventListener* listener)
{
    return addFile(directoryname+filename,listener);
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
