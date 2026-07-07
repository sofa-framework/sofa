/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it    *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at    *
* your option) any later version.                                            *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT*
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License*
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License   *
* along with this program. If not, see <http://www.gnu.org/licenses/>.       *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <sofa/simulation/config.h>
#include <sofa/core/objectmodel/Snapshot.h>
#include <sofa/core/sptr.h>
#include <sofa/simulation/Node.h>


namespace sofa::simulation
{

/**
 * \brief Snapshot container and memory save/load helper.
 *
 * Actually, this class:
 * - Store snapshots (from files or memory)
 * - Do save/load in memory
 * - Do save/load in a file
 *
 */
class SOFA_SIMULATION_CORE_API SnapshotManager
{
public:
    SnapshotManager();
    ~SnapshotManager();

    /// Container of snapshot from Files
    std::vector<std::string> m_recentSnapshotsFromFiles;

    /// Container of snapshot from Memory
    std::map<std::string, std::shared_ptr<sofa::core::objectmodel::Snapshot>> m_recentSnapshotsFromMemory;

    /// Store every snapshot from files in m_recentSnapshotsFromFiles and sort them by filename
    void addRecentFile(const std::string& path);

    /// Store every snapshot in memory in m_recentSnapshotsFromMemory and sort them by simulation time
    void addRecentSnapshot(std::shared_ptr<sofa::core::objectmodel::Snapshot> snapshot,
                           double snapshotTime);

    /// Save and store a Snapshot in memory
    void doMemorySave(sofa::core::sptr<sofa::simulation::Node>& groot);

    /// Load a snapshot from memory
    void doMemoryLoad(sofa::core::sptr<sofa::simulation::Node>& groot);

    /// Save and store a Snapshot to a file (or Save and store a group of Snapshot to a file when isGroup is true)
    void doSaveTo(sofa::core::sptr<sofa::simulation::Node>& groot,std::string savePath, bool isGroup);

    /// Load a snapshot from a file
    void doLoadTo(sofa::core::sptr<sofa::simulation::Node>& groot, std::string outPath);

    /// Load a group of snapshots from a file
    void doLoadToGroup(const std::string& filename);


};

} // namespace sofa::simulation
