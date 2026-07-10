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
#include <sofa/core/objectmodel/SnapshotJSONExporter.cpp>
#include <utility>
#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;



namespace sofa::simulation
{

SnapshotManager::SnapshotManager() = default;

SnapshotManager::~SnapshotManager() = default;

void SnapshotManager::addSnapshotFromFile(const std::string& path)
{
    m_snapshotsFromFiles.erase(std::remove(m_snapshotsFromFiles.begin(), m_snapshotsFromFiles.end(), path), m_snapshotsFromFiles.end());
    m_snapshotsFromFiles.push_back(path);
}

void SnapshotManager::addSnapshotFromMemory(std::shared_ptr<sofa::core::objectmodel::Snapshot> snapshot,
                                        double snapshotTime)
{
    static int index = 0;
    m_snapshotsFromMemory["Memory_Snapshot " + std::to_string(index++) + " at " + std::to_string(snapshotTime)] = std::move(snapshot);
}

void SnapshotManager::doMemorySave(sofa::core::sptr<Node>& groot)
{
    auto snapshot = std::make_shared<sofa::core::objectmodel::Snapshot>();
    auto visitor = SaveSnapshotVisitor(nullptr, *snapshot);
    groot->execute(visitor);
    addSnapshotFromMemory(snapshot, groot->getTime());
}

void SnapshotManager::doMemoryLoad(sofa::core::sptr<Node>& groot)
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

void SnapshotManager::doSaveTo(sofa::core::sptr<sofa::simulation::Node>& groot,std::string savePath, bool isSet)
{
    auto m_snapshot = std::make_shared<sofa::core::objectmodel::Snapshot>();
    auto visitor = SaveSnapshotVisitor(nullptr,*m_snapshot);
    groot->execute(visitor);


    std::string FileExtension = FileSystem::getExtension(savePath);
    if (FileExtension == "json" && !isSet)
        exportToJSON(*m_snapshot,savePath);
    else if (FileExtension == "json" && isSet)
        exportToJSON(m_snapshotsFromMemory,savePath);
    else
        msg_error("SaveSnapshot") << "Snapshot " << savePath << " not supported";

    addSnapshotFromFile(savePath);
    msg_info("SaveSnapshot") << "Snapshot " << savePath << " saved";
}

void SnapshotManager::doLoadTo(sofa::core::sptr<sofa::simulation::Node>& groot, std::string outPath)
{
    auto m_snapshot = std::make_shared<sofa::core::objectmodel::Snapshot>();
    if (FileSystem::exists(outPath))
    {
        importFrom(*m_snapshot,outPath);
        auto visitor = LoadSnapshotVisitor(nullptr,*m_snapshot);
        groot->execute(visitor);
    }
    addSnapshotFromFile(outPath);
    msg_info("LoadSnapshot") << "Snapshot " << outPath << " loaded";
}

void SnapshotManager::doLoadToSet(const std::string& filename)
{
    if (!FileSystem::exists(filename))
        return;

    std::ifstream file(filename);

    nlohmann::json jSnapshot;

    if (!file.is_open())
    {
        msg_error("SnapshotJSONExporter") << "Cannot open file " << filename << " for reading";
        return;
    }

    file >> jSnapshot;
    file.close();

    for (const auto& snapshotJson : jSnapshot)
    {
        auto snapshot = std::make_shared<sofa::core::objectmodel::Snapshot>();
        snapshot->m_graphRoot = std::make_shared<sofa::core::objectmodel::Snapshot::SnapshotNode>();
        sofa::core::objectmodel::from_json(snapshotJson, *snapshot->m_graphRoot);
        std::string snapshotTime = "0";
        for (const auto& data : snapshot->m_graphRoot->m_dataContainer)
        {
            if (data.name == "time")
                snapshotTime = data.value;
        }

        addSnapshotFromMemory(snapshot, std::stod(snapshotTime));
    }

    msg_info("LoadSnapshot") << "Snapshot " << filename << " loaded";
}

} // namespace sofa::simulation
