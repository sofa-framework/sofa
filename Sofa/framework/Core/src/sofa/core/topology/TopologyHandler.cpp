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
#define SOFA_CORE_TOPOLOGY_TOPOLOGYHANDLER_DEFINITION true
#include <sofa/core/topology/TopologyHandler.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::core::topology
{

size_t TopologyHandler::getNumberOfTopologicalChanges()
{
    return (m_topology->m_changeList.getValue()).size();
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Generic Handling of Topology Event    /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

TopologyHandler::TopologyHandler()
    : m_topology(nullptr)
{
    
}


void TopologyHandler::ApplyTopologyChanges(const std::list<const core::topology::TopologyChange*>& _topologyChangeEvents, const Size _dataSize)
{
    SOFA_UNUSED(_dataSize);

    if (!this->isTopologyHandlerRegistered())
        return;

    std::list<const core::topology::TopologyChange*>::iterator changeIt;
    std::list<const core::topology::TopologyChange*> _changeList = _topologyChangeEvents;

    for (changeIt = _changeList.begin(); changeIt != _changeList.end(); ++changeIt)
    {
        core::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();
        std::string topoChangeType = m_prefix + ": " + m_data_name + " - " + parseTopologyChangeTypeToString(changeType);
        helper::ScopedAdvancedTimer timer(topoChangeType);

        // New version using map of callback
        std::map < core::topology::TopologyChangeType, TopologyChangeCallback>::iterator itM;
        itM = m_callbackMap.find(changeType);

        if (itM != m_callbackMap.end())
        {
            (*itM).second(*changeIt);
        }


        // old version to be removed.
        switch (changeType)
        {
#define SOFA_CASE_EVENT(name,type) \
        case core::topology::name: \
            this->ApplyTopologyChange(static_cast< const type* >( *changeIt ) ); \
            break

            SOFA_CASE_EVENT(ENDING_EVENT, EndingEvent);

            SOFA_CASE_EVENT(POINTSINDICESSWAP, PointsIndicesSwap);
            SOFA_CASE_EVENT(POINTSADDED, PointsAdded);
            SOFA_CASE_EVENT(POINTSREMOVED, PointsRemoved);
            SOFA_CASE_EVENT(POINTSMOVED, PointsMoved);
            SOFA_CASE_EVENT(POINTSRENUMBERING, PointsRenumbering);

            SOFA_CASE_EVENT(EDGESINDICESSWAP, EdgesIndicesSwap);
            SOFA_CASE_EVENT(EDGESADDED, EdgesAdded);
            SOFA_CASE_EVENT(EDGESREMOVED, EdgesRemoved);
            SOFA_CASE_EVENT(EDGESMOVED_REMOVING, EdgesMoved_Removing);
            SOFA_CASE_EVENT(EDGESMOVED_ADDING, EdgesMoved_Adding);
            SOFA_CASE_EVENT(EDGESRENUMBERING, EdgesRenumbering);

            SOFA_CASE_EVENT(TRIANGLESINDICESSWAP, TrianglesIndicesSwap);
            SOFA_CASE_EVENT(TRIANGLESADDED, TrianglesAdded);
            SOFA_CASE_EVENT(TRIANGLESREMOVED, TrianglesRemoved);
            SOFA_CASE_EVENT(TRIANGLESMOVED_REMOVING, TrianglesMoved_Removing);
            SOFA_CASE_EVENT(TRIANGLESMOVED_ADDING, TrianglesMoved_Adding);
            SOFA_CASE_EVENT(TRIANGLESRENUMBERING, TrianglesRenumbering);

            SOFA_CASE_EVENT(TETRAHEDRAINDICESSWAP, TetrahedraIndicesSwap);
            SOFA_CASE_EVENT(TETRAHEDRAADDED, TetrahedraAdded);
            SOFA_CASE_EVENT(TETRAHEDRAREMOVED, TetrahedraRemoved);
            SOFA_CASE_EVENT(TETRAHEDRAMOVED_REMOVING, TetrahedraMoved_Removing);
            SOFA_CASE_EVENT(TETRAHEDRAMOVED_ADDING, TetrahedraMoved_Adding);
            SOFA_CASE_EVENT(TETRAHEDRARENUMBERING, TetrahedraRenumbering);

            SOFA_CASE_EVENT(QUADSINDICESSWAP, QuadsIndicesSwap);
            SOFA_CASE_EVENT(QUADSADDED, QuadsAdded);
            SOFA_CASE_EVENT(QUADSREMOVED, QuadsRemoved);
            SOFA_CASE_EVENT(QUADSMOVED_REMOVING, QuadsMoved_Removing);
            SOFA_CASE_EVENT(QUADSMOVED_ADDING, QuadsMoved_Adding);
            SOFA_CASE_EVENT(QUADSRENUMBERING, QuadsRenumbering);

            SOFA_CASE_EVENT(HEXAHEDRAINDICESSWAP, HexahedraIndicesSwap);
            SOFA_CASE_EVENT(HEXAHEDRAADDED, HexahedraAdded);
            SOFA_CASE_EVENT(HEXAHEDRAREMOVED, HexahedraRemoved);
            SOFA_CASE_EVENT(HEXAHEDRAMOVED_REMOVING, HexahedraMoved_Removing);
            SOFA_CASE_EVENT(HEXAHEDRAMOVED_ADDING, HexahedraMoved_Adding);
            SOFA_CASE_EVENT(HEXAHEDRARENUMBERING, HexahedraRenumbering);
#undef SOFA_CASE_EVENT
        default:
            break;
        } // switch( changeType )

        //++changeIt;
    }
}

void TopologyHandler::update()
{
    DDGNode::cleanDirty();
    if (!this->isTopologyHandlerRegistered())
        return;

    const std::string msg = this->getName() + " - doUpdate: Nbr changes: " + std::to_string(m_topology->m_changeList.getValue().size());
    helper::ScopedAdvancedTimer timer(msg);
    this->handleTopologyChange();
}

void TopologyHandler::addCallBack(core::topology::TopologyChangeType type, TopologyChangeCallback callback)
{
    // need to warn duplicate callback
    m_callbackMap[type] = callback;
}

bool TopologyHandler::registerTopology(sofa::core::topology::BaseMeshTopology* _topology, bool printLog)
{
    m_topology = dynamic_cast<sofa::core::topology::TopologyContainer*>(_topology);

    if (m_topology == nullptr)
    {
        msg_info_when(printLog, "TopologyHandler") << "The " << m_prefix << " managing the TopologyData '" << m_data_name
            << "' won't be registered because linked topology '" << _topology->getName() << "' is not dynamic. Topological changes won't be supported by this Data.";
        return false;
    }

    return true;
}

} // namespace sofa

