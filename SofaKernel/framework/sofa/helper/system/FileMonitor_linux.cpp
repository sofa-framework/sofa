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
//////////////////// C Header /////////////////////////////////////////////////
#include <sys/inotify.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <libgen.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>

//////////////////// C++ Header ///////////////////////////////////////////////
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
using namespace boost::filesystem;

#include "FileSystem.h"
using sofa::helper::system::FileSystem ;

#include "FileMonitor.h"

#include <boost/filesystem.hpp>

using namespace std ;

#define EVENT_SIZE  ( sizeof (struct inotify_event) + NAME_MAX + 1 )
#define BUF_LEN     ( 1024 * ( EVENT_SIZE + 16 ) )

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

int filemonitor_inotifyfd=-1 ;
const char* eventmaskToString(int evtmask);

//
void addAFileListenerInDict(string pathfilename, FileEventListener* listener)
{
    if(file2listener.find(pathfilename)==file2listener.end())
        file2listener[pathfilename] = ListOfListeners() ;

    ListOfListeners& ll = file2listener[pathfilename] ;
    if(find(ll.begin(), ll.end(), listener)==ll.end()){
        ll.push_back(listener) ;
    }
}

//////////////////// Public Function //////////////////////////////////////////
int FileMonitor_init()
{
    if(filemonitor_inotifyfd>=0)
        return filemonitor_inotifyfd ;

    // Here we should add a file to monitor for change.
    filemonitor_inotifyfd = inotify_init() ;
    if ( filemonitor_inotifyfd < 0 ) {
        return -1 ;
    }
    return 0 ;
}

void FileMonitor::removeFileListener(const string& filename,
                                     FileEventListener *listener)
{
    path prefix(FileSystem::getParentDirectory(filename)) ;
    path name(FileSystem::stripDirectory(filename)) ;

    path fullPath = prefix/name;

    if(! exists(status(fullPath)) )
        return;

    path absolutePath = canonical(fullPath) ;
    map<string, ListOfListeners>::iterator cur = file2listener.begin() ;
    map<string, ListOfListeners>::iterator end = file2listener.end() ;

    for(;cur!=end;++cur){

        if(std::find(cur->second.begin(),cur->second.end(), listener)
                != cur->second.end() && cur->first ==  absolutePath){
            cur->second.erase(std::remove(cur->second.begin(),
                                          cur->second.end(), listener),
                              cur->second.end());
        }
    }
}


void FileMonitor::removeListener(FileEventListener *listener){
    //file2listener[parentname+"/"+filename] = listener ;
    map<string, ListOfListeners>::iterator cur = file2listener.begin() ;
    map<string, ListOfListeners>::iterator end = file2listener.end() ;

    for(;cur!=end;++cur){

        if(std::find(cur->second.begin(),cur->second.end(), listener)
                != cur->second.end()){
            cur->second.erase(std::remove(cur->second.begin(),
                                          cur->second.end(), listener),
                              cur->second.end());
        }
    }
}

int FileMonitor::addFile(const std::string& filepath, FileEventListener* listener)
{
    if(listener == NULL)
        return -1 ;

    if(!FileSystem::exists(filepath))
        return -1 ;

    if(filemonitor_inotifyfd<0)
        if( FileMonitor_init() < 0)
            return -1;

    return addFile(FileSystem::getParentDirectory(filepath),
                   FileSystem::stripDirectory(filepath), listener) ;
}

int FileMonitor::addFile(const std::string& parentname,
                         const std::string& filename,
                         FileEventListener* listener)
{
    if(listener == NULL)
        return -1 ;

    path prefix(parentname) ;
    path name(filename) ;

    path fullPath = prefix/name;

    if(! exists(status(fullPath)) )
        return -1;

    path absolutePath = canonical(fullPath) ;


    if(! exists(status(absolutePath)) )
        return -1;

    if(filemonitor_inotifyfd<0)
        FileMonitor_init() ;

    std::string parentnameL = absolutePath.parent_path().string() ;
    std::string filenameL = absolutePath.filename().string() ;

    // Is the directory in the already monitored files ?
    if( dir2files.find(parentnameL) != dir2files.end() ) {
        // If so, is the file in the monitored files ?
        addAFileListenerInDict(parentnameL+"/"+filenameL,listener);
        ListOfFiles& lf=dir2files[parentnameL];
        if(find(lf.begin(), lf.end(), filenameL)==lf.end()){
            dir2files[parentnameL].push_back(filenameL) ;
        }
    } else {
        // If the directory is not yet monitored we add it to the system.
        dir2files[parentnameL] = ListOfFiles() ;
        int wd=inotify_add_watch( filemonitor_inotifyfd,
                                  parentnameL.c_str(),
                                  IN_CLOSE | IN_MOVED_TO ) ;
        fd2fn[wd]=string(parentnameL) ;
        addAFileListenerInDict(parentnameL+"/"+filenameL,listener);

        dir2files[parentnameL].push_back(filenameL) ;
    }

    return 1 ;
}

int FileMonitor::updates(int timeout)
{
    if(filemonitor_inotifyfd<0)
        FileMonitor_init() ;

    fd_set descriptors ;
    struct timeval time_to_wait ;

    time_to_wait.tv_sec = timeout ;
    time_to_wait.tv_usec = 0 ;

    // Do the descriptor changed ?
    FD_ZERO( &descriptors ) ;
    FD_SET(filemonitor_inotifyfd, &descriptors) ;
    int return_value = select (FD_SETSIZE, &descriptors, NULL, NULL, &time_to_wait) ;

    if ( return_value < 0 ) {
        return -1 ;
        /* Error on select */
    } else if ( !return_value ) {
        /* Timeout */
    } else if(FD_ISSET(filemonitor_inotifyfd, &descriptors)) {
        // TODO(damien): I hate this timer but otherwise some events are
        // duplicated so I prefer to give some time to the system to finish
        // its tasks. The 10000 is a value that work on my system :(
        // Arg non pitié je peux pas laisser ça en l'état c'est trop moche.
        // Promis demain je trouve un meilleur solution (ou pas).
        usleep(30000);
        int length = 0 ;
        char buffer[BUF_LEN];
        int buffer_i = 0 ;
        memset(buffer, 'a', BUF_LEN) ;
        length = read( filemonitor_inotifyfd, buffer, BUF_LEN ) ;

        vector<string> changedfiles ;
        while (buffer_i < length) {
            struct inotify_event* pevent = (struct inotify_event *)&buffer[buffer_i] ;
            //cout << "Event received ...from " << string(fd2fn[pevent->wd])
            //      << ":" << pevent->name
            //      << "->" << eventmaskToString(pevent->mask) << endl;
            if(pevent->mask & ( IN_CLOSE_WRITE  | IN_MOVED_TO )) {
                if(dir2files.find(fd2fn[pevent->wd])!=dir2files.end()) {
                    ListOfFiles& dl=dir2files[fd2fn[pevent->wd]] ;
                    string fullname = string(fd2fn[pevent->wd])+"/"+string(pevent->name) ;

                    if(find(dl.begin(), dl.end(), pevent->name)!=dl.end()) {
                        if(find(changedfiles.begin(), changedfiles.end(), fullname)==changedfiles.end()) {
                            changedfiles.push_back(fullname) ;
                        }
                    }
                }
            }
            buffer_i += sizeof(struct inotify_event)+pevent->len ;
        }


        for(vector<string>::iterator f=changedfiles.begin(); f!=changedfiles.end(); f++) {
            ListOfListeners::iterator it = file2listener[*f].begin() ;
            ListOfListeners::iterator end = file2listener[*f].end() ;
            for(;it!=end;++it){
                (*it)->fileHasChanged(*f) ;
            }
        }

        if(changedfiles.size()!=0) {
            return 1 ;
        }
    } else {
        cout << "filemonitorUpdates:: \n" ;
        cout << "\t message to developer(to implement one day) \n" ;
        cout << "\t message to user(please tells the feature is import to you)." <<endl ;
    }
    return 0 ;
}

//////////////////// Utilitary functions///////////////////////////////////////
const char* eventmaskToString(int evtmask)
{
    //Source: http://man7.org/linux/man-pages/man7/inotify.7.html
    if(evtmask & IN_ACCESS) {
        return "IN_ACCESS" ;
    }
    if(evtmask & IN_ATTRIB) {
        return "IN_ATTRIB" ;
    }
    if(evtmask & IN_CLOSE_WRITE) {
        return "IN_CLOSE_WRITE" ;
    }
    if(evtmask & IN_CLOSE_NOWRITE) {
        return "IN_CLOSE_NOWRITE" ;
    }
    if(evtmask & IN_CREATE) {
        return "IN_CREATE" ;
    }
    if(evtmask & IN_DELETE) {
        return "IN_DELETE" ;
    }
    if(evtmask & IN_DELETE_SELF) {
        return "IN_DELETE_SELF" ;
    }
    if(evtmask & IN_MODIFY) {
        return "IN_MODIFY" ;
    }
    if(evtmask & IN_MOVE_SELF) {
        return "IN_MOVE_SELF" ;
    }
    if(evtmask & IN_MOVED_FROM) {
        return "IN_MOVED_SELF" ;
    }
    if(evtmask & IN_MOVED_TO) {
        return "IN_MOVED_TO" ;
    }
    if(evtmask & IN_OPEN) {
        return "IN_OPEN" ;
    }
    if(evtmask & IN_MOVE) {
        return "IN_MOVE" ;
    }
    if(evtmask & IN_CLOSE) {
        return "IN_CLOSE" ;
    }
    if(evtmask & IN_IGNORED) {
        return "IN_IGNORED" ;
    }
    if(evtmask & IN_ISDIR) {
        return "IN_ISDIR" ;
    }
    if(evtmask & IN_Q_OVERFLOW) {
        return "IN_Q_OVERFLOW" ;
    }
    if(evtmask & IN_UNMOUNT) {
        return "IN_UNMOUNT" ;
    }
    if(evtmask == 0) {
        return "NoEvent" ;
    }
    // FIXME
    return "UnknowInotifyEvent" ;
}


}
}
}
