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
#include <sofa/core/objectmodel/SnapshotManager.h>
#include <fstream>
#include <string>

namespace sofa::core::objectmodel
{

SnapshotManager::SnapshotManager() = default;

SnapshotManager::~SnapshotManager() = default;

void SnapshotManager::AddRecentFile(const std::string& path, std::vector<std::string>& recentFiles, int maxFiles)
{
    recentFiles.erase(
        std::remove(recentFiles.begin(), recentFiles.end(), path),
        recentFiles.end()
    );
    //recentFiles.insert(recentFiles.begin(), path);
    recentFiles.push_back(path);
    if (recentFiles.size() > maxFiles)
        recentFiles.resize(maxFiles);
}

void SnapshotManager::AddRecentSnapshot(std::map<std::string, std::shared_ptr<sofa::core::objectmodel::Snapshot>>& recentSnapshots, std::shared_ptr<sofa::core::objectmodel::Snapshot> snapshot, double snapshotTime, int maxSnapshots)
{
    static int index = 0;
    recentSnapshots["Memory_Snapshot " + std::to_string(index++) + " at " + std::to_string(snapshotTime)] = snapshot;
    if (recentSnapshots.size() > maxSnapshots)
        recentSnapshots.erase(recentSnapshots.begin());
}

} // namespace sofa::core::objectmodel