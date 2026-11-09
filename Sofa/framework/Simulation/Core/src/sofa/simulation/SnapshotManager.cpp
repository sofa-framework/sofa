/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it    *
* under the terms of the GNU Lesser General Public License as published by   *
* the Free Software Foundation; either version 2.1 of the License, or (at   *
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
#include <sofa/simulation/SnapshotManager.h>

#include <sofa/simulation/LoadSnapshotVisitor.h>
#include <sofa/simulation/SaveSnapshotVisitor.h>
#include <fstream>
#include <sofa/helper/logging/Messaging.h>
#include <algorithm>
#include <sofa/core/objectmodel/JSONSnapshot.h>
#include <sofa/core/ObjectFactory.h>
#include <utility>
#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;



namespace sofa::simulation
{

void registerSnapshotManager(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Manages the snapshots in the simulation")
        .add< SnapshotManager >());
}

SnapshotManager::SnapshotManager()
    :d_numberOfSnapshots( initData(&d_numberOfSnapshots, "numberOfSnapshots", "number of Snapshots inside the SnapshotManager"))
{

}

SnapshotManager::~SnapshotManager()
{

}
void SnapshotManager::addSnapshotToFile(const std::string& path)
{
    m_snapshotsFromFiles.erase(std::remove(m_snapshotsFromFiles.begin(), m_snapshotsFromFiles.end(), path), m_snapshotsFromFiles.end());
    m_snapshotsFromFiles.push_back(path);
    d_numberOfSnapshots.setValue(d_numberOfSnapshots.getValue() + 1);
}

void SnapshotManager::addSnapshotToMemory(const std::shared_ptr<sofa::core::objectmodel::Snapshot>& snapshot,
                                        double snapshotTime)
{
    static int index = 0;
    m_snapshotsFromMemory["Memory_Snapshot " + std::to_string(index++) + " at " + std::to_string(snapshotTime)] = snapshot;
    d_numberOfSnapshots.setValue(d_numberOfSnapshots.getValue() + 1);
}

void SnapshotManager::memorySave(const sofa::core::sptr<Node>& groot)
{
    auto snapshot = std::make_shared<sofa::core::objectmodel::Snapshot>();
    auto visitor = SaveSnapshotVisitor(nullptr, *snapshot);
    groot->execute(visitor);
    addSnapshotToMemory(snapshot, groot->getTime());
}

void SnapshotManager::memoryLoad(sofa::core::sptr<Node>& groot)
{
    if (m_snapshotsFromMemory.empty())
    {
        msg_warning("MemoryLoad") << "No Snapshot in memory";
        return;
    }

    const auto& snapshot = m_snapshotsFromMemory.rbegin()->second;
    auto visitor = LoadSnapshotVisitor(nullptr, *snapshot);
    groot->execute(visitor);
}

void SnapshotManager::saveTo(const sofa::core::sptr<sofa::simulation::Node>& groot, const std::string& savePath, bool isSet)
{
    auto m_snapshot = std::make_shared<sofa::core::objectmodel::Snapshot>();
    auto visitor = SaveSnapshotVisitor(nullptr,*m_snapshot);
    groot->execute(visitor);


    std::string FileExtension = FileSystem::getExtension(savePath);
    if (FileExtension == "json" && !isSet)
        core::objectmodel::jsonsnapshot::exportToJSON(*m_snapshot,savePath);
    else if (FileExtension == "json" && isSet)
        core::objectmodel::jsonsnapshot::exportToJSON(m_snapshotsFromMemory,savePath);
    else
        msg_error("SaveSnapshot") << "Snapshot " << savePath << " not supported";

    addSnapshotToFile(savePath);
    msg_info("SaveSnapshot") << "Snapshot " << savePath << " saved";
}

void SnapshotManager::loadTo(sofa::core::sptr<sofa::simulation::Node>& groot, const std::string& outPath)
{
    auto m_snapshot = std::make_shared<sofa::core::objectmodel::Snapshot>();

    std::string FileExtension = FileSystem::getExtension(outPath);

    if (FileSystem::exists(outPath) && FileExtension == "json")
    {
        core::objectmodel::jsonsnapshot::importFromJSON(*m_snapshot,outPath);
    }
    else
    {
        msg_error("LoadSnapshot") << "Snapshot " << outPath << " not found";
        return;
    }
    auto visitor = LoadSnapshotVisitor(nullptr,*m_snapshot);
    groot->execute(visitor);
    addSnapshotToFile(outPath);
    msg_info("LoadSnapshot") << "Snapshot " << outPath << " loaded";
}

void SnapshotManager::loadToSet(const std::string& filename)
{
    if (!FileSystem::exists(filename))
        return;

    std::map<std::shared_ptr<sofa::core::objectmodel::Snapshot>,double> snapshots;

    core::objectmodel::jsonsnapshot::doLoadSet(filename,snapshots);

    for (const auto&[snapshot, snapshotTime] : snapshots)
    {
        addSnapshotToMemory(snapshot, snapshotTime);
    }

    msg_info("LoadSnapshot") << "Snapshot " << filename << " loaded";
}

} // namespace sofa::simulation
