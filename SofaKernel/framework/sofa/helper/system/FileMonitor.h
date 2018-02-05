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
#ifndef SOFA_HELPER_SYSTEM_FILEMONITOR_H
#define SOFA_HELPER_SYSTEM_FILEMONITOR_H

#include<string>

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace system
{
using std::string ;

/// @brief Contains an event-based API to monitor file changes.
///
/// Those functions are here only to avoid depending on an external library, and
/// they provide only basic functionality.
///
/// This set of functions is not meant to be complete, but it can be completed
/// if need be.
///
/// Example of use:
///     1) implement the FileEventListener interface
///     2) register the files you want to monitor using
///         FileMonitor::addFile("mypath/to/file", &mylistener);
///     3) call the update function to process & trigger the events.
///
/// The system does not contains any hidden thread. If you request
/// the monitoring to happens in an hidden thread you can implement it
/// externally.
///
class SOFA_HELPER_API FileEventListener
{
public:
    virtual ~FileEventListener() {}
    virtual void fileHasChanged(const std::string& filename) = 0;
};

class SOFA_HELPER_API FileMonitor
{
public:
    /// @brief add a new filepath to monitor and a listener to be triggered
    /// in case of change.
    /// returns >= 0 if the file was successfully added
    /// returns < 0  in case of error.
    static int addFile(const std::string& filename, FileEventListener* listener) ;

    /// @brief add a new path and file to monitor and a listener to be triggered
    /// in case of change.
    /// returns >= 0 if the file was successfully added
    /// returns < 0  in case of error.
    static int addFile(const std::string& directoryname,
                       const std::string& filename, FileEventListener* listener) ;

    /// @brief check if the file have changed, colalesc the similar events
    /// and notify the listener.
    /// timeout is the number of seconds to block the calling process. Can
    /// be 0 (in this case it return immediately if there is no event.
    /// return -1 in case of error
    /// return >= 0 otherwise.
    static int updates(int timeout=1) ;

    /// @brief remove the provided listener.
    /// If the listener is not existing, do nothing
    /// If the listener is NULL, do nothing
    /// If the listener is associated with one or more file...remove all
    /// the associations.
    /// Keep in mind that the file are still monitored.
    static void removeListener(FileEventListener* listener) ;

    /// @brief remove the provided listener for a given file
    /// If the listener is not existing, do nothing
    /// If the listener is NULL, do nothing
    /// If the listener is associated with one or more file...remove all
    /// the associations.
    /// Keep in mind that the file are still monitored.
    static void removeFileListener(const std::string& filename,
                                   FileEventListener* listener) ;
};

}
}
}

#endif // FILEMONITOR_H_
