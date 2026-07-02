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
#pragma once
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/Snapshot.h>

namespace sofa::core::objectmodel
{

/**
*  \brief Snapshot Class for a container of snapshot
*
*  This class contains methods and elements that will help to manage snapshot in memory.
*/
class SnapshotManager
{
public:
    SnapshotManager();
    ~SnapshotManager();

    std::vector<std::string> recentSnapshotFiles;
    std::map<std::string, std::shared_ptr<sofa::core::objectmodel::Snapshot>> recentSnapshots;


    static void AddRecentFile(const std::string& path, std::vector<std::string>& recentFiles, int maxFiles = 10);
    static void AddRecentSnapshot(std::map<std::string, std::shared_ptr<sofa::core::objectmodel::Snapshot>>& recentSnapshots, std::shared_ptr<sofa::core::objectmodel::Snapshot> snapshot, double snapshotTime );

};
} // namespace sofa::core::objectmodel