/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTopologyMapping/Triangle2EdgeTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>
#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology;
using namespace sofa::core::topology;

// Register in the Factory
int Triangle2EdgeTopologicalMappingClass = core::RegisterObject("Special case of mapping where TriangleSetTopology is converted to EdgeSetTopology")
        .add< Triangle2EdgeTopologicalMapping >();

Triangle2EdgeTopologicalMapping::Triangle2EdgeTopologicalMapping()
    : sofa::core::topology::TopologicalMapping()
    , m_outTopoModifier(NULL)
{
}


Triangle2EdgeTopologicalMapping::~Triangle2EdgeTopologicalMapping()
{
    sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());
    Loc2GlobVec.clear();
    Glob2LocMap.clear();
    Loc2GlobDataVec.endEdit();
}


void Triangle2EdgeTopologicalMapping::init()
{
    // recheck models
    bool modelsOk = true;
    if (!fromModel)
    {
        msg_error() << "Pointer to input topology is invalid.";
        modelsOk = false;
    }

    if (!toModel)
    {
        msg_error() << "Pointer to output topology is invalid.";
        modelsOk = false;
    }

    toModel->getContext()->get(m_outTopoModifier);
    if (!m_outTopoModifier)
    {
        msg_error() << "No EdgeSetTopologyModifier found in the Edge topology Node.";
        modelsOk = false;
    }

    if (!modelsOk)
    {
        this->m_componentstate = sofa::core::objectmodel::ComponentState::Invalid;
        return;
    }


    // INITIALISATION of EDGE mesh from TRIANGULAR mesh :
    // Clear output topology
    toModel->clear();

    // Set the same number of points
    toModel->setNbPoints(fromModel->getNbPoints());

    // create topology maps and add edge into output topology
    const sofa::helper::vector<core::topology::BaseMeshTopology::Edge> &edgeArray = fromModel->getEdges();
    sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());
    Loc2GlobVec.clear();
    Glob2LocMap.clear();

    for (Topology::EdgeID eId=0; eId<edgeArray.size(); ++eId)
    {
        if (fromModel->getTrianglesAroundEdge(eId).size() == 1)
        {
            const Topology::Edge& e = edgeArray[eId];
            toModel->addEdge(e[0], e[1]);

            Loc2GlobVec.push_back(eId);
            Glob2LocMap[eId] = Loc2GlobVec.size() - 1;
        }
    }

    Loc2GlobDataVec.endEdit();

    // Need to fully init the target topology
    toModel->init();

    this->m_componentstate = sofa::core::objectmodel::ComponentState::Valid;
}


unsigned int Triangle2EdgeTopologicalMapping::getFromIndex(unsigned int ind)
{
    if (fromModel->getTrianglesAroundEdge(ind).size()==1)
    {
        return fromModel->getTrianglesAroundEdge(ind)[0];
    }
    else
    {
        return 0;
    }
}

void Triangle2EdgeTopologicalMapping::updateTopologicalMappingTopDown()
{
    if (this->m_componentstate != sofa::core::objectmodel::ComponentState::Valid)
        return;

    sofa::helper::AdvancedTimer::stepBegin("Update Triangle2EdgeTopologicalMapping");

    std::list<const TopologyChange *>::const_iterator itBegin=fromModel->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

    sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

    while( itBegin != itEnd )
    {
        TopologyChangeType changeType = (*itBegin)->getChangeType();
        std::string topoChangeType = "Triangle2EdgeTopologicalMapping - " + parseTopologyChangeTypeToString(changeType);
        sofa::helper::AdvancedTimer::stepBegin(topoChangeType);

        switch( changeType )
        {
        case core::topology::ENDING_EVENT:
        {
            m_outTopoModifier->notifyEndingEvent();
            m_outTopoModifier->propagateTopologicalChanges();
            break;
        }
        case core::topology::EDGESREMOVED:
        {
            unsigned int last = (unsigned int)fromModel->getNbEdges() - 1;
            unsigned int ind_last = (unsigned int)toModel->getNbEdges();

            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const EdgesRemoved *>( *itBegin ) )->getArray();

            unsigned int ind_tmp;
            unsigned int ind_real_last;

            for (unsigned int i = 0; i <tab.size(); ++i)
            {
                unsigned int k = tab[i];

                std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(k);
                if(iter_1 != Glob2LocMap.end())
                {
                    ind_last = ind_last - 1;

                    unsigned int ind_k = Glob2LocMap[k];

                    std::map<unsigned int, unsigned int>::iterator iter_2 = Glob2LocMap.find(last);
                    if(iter_2 != Glob2LocMap.end())
                    {
                        ind_real_last = Glob2LocMap[last];

                        if (k != last)
                        {

                            Glob2LocMap.erase(Glob2LocMap.find(k));
                            Glob2LocMap[k] = ind_real_last;

                            Glob2LocMap.erase(Glob2LocMap.find(last));
                            Glob2LocMap[last] = ind_k;

                            ind_tmp = Loc2GlobVec[ind_real_last];
                            Loc2GlobVec[ind_real_last] = Loc2GlobVec[ind_k];
                            Loc2GlobVec[ind_k] = ind_tmp;
                        }
                    }

                    if(ind_k != ind_last)
                    {

                        Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_last]));
                        Glob2LocMap[Loc2GlobVec[ind_last]] = ind_k;

                        Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[ind_k]));
                        Glob2LocMap[Loc2GlobVec[ind_k]] = ind_last;

                        ind_tmp = Loc2GlobVec[ind_k];
                        Loc2GlobVec[ind_k] = Loc2GlobVec[ind_last];
                        Loc2GlobVec[ind_last] = ind_tmp;

                    }

                    Glob2LocMap.erase(Glob2LocMap.find(Loc2GlobVec[Loc2GlobVec.size() - 1]));
                    Loc2GlobVec.resize( Loc2GlobVec.size() - 1 );

                    sofa::helper::vector< unsigned int > edges_to_remove;
                    edges_to_remove.push_back(ind_k);
                    m_outTopoModifier->removeEdges(edges_to_remove, false);

                }
                else
                {
                    msg_warning() << "Glob2LocMap should have the visible edge " << tab[i];
                    msg_warning() << "INFO_print : Triangle2EdgeTopologicalMapping - nb edges = " << ind_last;
                }

                --last;
            }

            //m_outTopoModifier->propagateTopologicalChanges();
            break;
        }
        case core::topology::TRIANGLESREMOVED:
        {
            const sofa::helper::vector<Topology::TriangleID> &tri2Remove = ( static_cast< const TrianglesRemoved *>( *itBegin ) )->getArray();

            sofa::helper::vector< core::topology::BaseMeshTopology::Edge > edges_to_create;
            sofa::helper::vector< unsigned int > edgesIndexList;
            size_t nb_elems = toModel->getNbEdges();

            // For each triangle removed inside the tri2Remove array. Will look if it has only one neighbour.
            // If yes means it will be added to the edge border topoloy.
            // NB: doesn't check if edge is inside 2 triangles removed. This will be handle in EdgeRemoved event.
            for (unsigned int i = 0; i < tri2Remove.size(); ++i)
            {
                Topology::TriangleID triId = tri2Remove[i];
                const BaseMeshTopology::EdgesInTriangle& edgesInTri = fromModel->getEdgesInTriangle(triId);
                // get each edge of the triangle involved
                for (auto edgeId : edgesInTri)
                {
                    const BaseMeshTopology::TrianglesAroundEdge& triAEdge = fromModel->getTrianglesAroundEdge(edgeId);

                    if (triAEdge.size() != 2) // means only one edge, will be removed later by EdgeRemoved event. Continue.
                        continue;

                    // Id of the opposite Triangle
                    Topology::TriangleID idTriNeigh;
                    if (triId == triAEdge[0])
                        idTriNeigh = triAEdge[1];
                    else
                        idTriNeigh = triAEdge[0];

                    // check if Triangle already processed in a previous iteration
                    bool is_present = false;
                    for (unsigned int k=0; k<i; ++k)
                        if (idTriNeigh == tri2Remove[k])
                        {
                            is_present = true;
                            break;
                        }
                    if (is_present) // already done, continue.
                        continue;

                    // Add this current edge to the output topology
                    Topology::Edge newEdge = fromModel->getEdge(edgeId);

                    // sort newEdge such that newEdge[0] is the smallest one
                    if (newEdge[0]>newEdge[1])
                    {
                        Topology::Point tmp = newEdge[0];
                        newEdge[0] = newEdge[1];
                        newEdge[1] = tmp;
                    }

                    // Add edge to creation buffers
                    edges_to_create.push_back(newEdge);
                    edgesIndexList.push_back(nb_elems);
                    nb_elems+=1;

                    // update topology maps
                    Loc2GlobVec.push_back(edgeId);
                    // check if edge already exist
                    std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(edgeId);
                    if(iter_1 != Glob2LocMap.end() )
                    {
                        dmsg_error() << "Fail to add edge " << edgeId << "which already exists with value: " << (*iter_1).second;
                        Glob2LocMap.erase(iter_1);
                    }
                    Glob2LocMap[edgeId] = (unsigned int)Loc2GlobVec.size()-1;
                }
            }

            m_outTopoModifier->addEdgesProcess(edges_to_create) ;
            m_outTopoModifier->addEdgesWarning(edges_to_create.size(), edges_to_create, edgesIndexList) ;
            m_outTopoModifier->propagateTopologicalChanges();
            break;
        }
        case core::topology::POINTSREMOVED:
        {
            const sofa::helper::vector<unsigned int> tab = ( static_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

            sofa::helper::vector<unsigned int> indices;

            for(unsigned int i = 0; i < tab.size(); ++i)
            {

                indices.push_back(tab[i]);
            }

            sofa::helper::vector<unsigned int>& tab_indices = indices;

            m_outTopoModifier->removePointsWarning(tab_indices, false);
            m_outTopoModifier->propagateTopologicalChanges();
            m_outTopoModifier->removePointsProcess(tab_indices, false);

            break;
        }
        case core::topology::POINTSRENUMBERING:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getIndexArray();
            const sofa::helper::vector<unsigned int> &inv_tab = ( static_cast< const PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

            sofa::helper::vector<unsigned int> indices;
            sofa::helper::vector<unsigned int> inv_indices;

            for(unsigned int i = 0; i < tab.size(); ++i)
            {
                indices.push_back(tab[i]);
                inv_indices.push_back(inv_tab[i]);
            }

            sofa::helper::vector<unsigned int>& tab_indices = indices;
            sofa::helper::vector<unsigned int>& inv_tab_indices = inv_indices;

            m_outTopoModifier->renumberPointsWarning(tab_indices, inv_tab_indices, false);
            m_outTopoModifier->propagateTopologicalChanges();
            m_outTopoModifier->renumberPointsProcess(tab_indices, inv_tab_indices, false);
            break;
        }
        /**
        case core::topology::EDGESADDED:
        {
            This case doesn't need to be handle here as TriangleAdded case is emit first and handle new edge to be added to output topology.
            break;
        }
        */
        case core::topology::TRIANGLESADDED:
        {
            const sofa::component::topology::TrianglesAdded *trianglesAdded = static_cast< const sofa::component::topology::TrianglesAdded * >(*itBegin);

            sofa::helper::vector< BaseMeshTopology::Edge > edges_to_create;
            sofa::helper::vector< BaseMeshTopology::EdgeID > edgeId_to_create;
            sofa::helper::vector< BaseMeshTopology::EdgeID > edgeId_to_remove;

            // Need to first add all the new edges before removing the old one.
            for (auto triId : trianglesAdded->triangleIndexArray)
            {
                const BaseMeshTopology::EdgesInTriangle& edgeInTri = fromModel->getEdgesInTriangle(triId);
                for (auto edgeGlobId : edgeInTri)
                {
                    std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(edgeGlobId);
                    const BaseMeshTopology::TrianglesAroundEdge& triAEdge = fromModel->getTrianglesAroundEdge(edgeGlobId);
                    if (iter_1 != Glob2LocMap.end()) // in the map
                    {
                        if (triAEdge.size() != 1) // already in the map but not anymore on border, add it for later removal.
                            edgeId_to_remove.push_back(edgeGlobId);
                    }
                    else
                    {
                        if (triAEdge.size() > 1) // not in the map and not on border, nothing to do.
                            continue;

                        // not in the map but on border. Need to add this edge.
                        core::topology::BaseMeshTopology::Edge edge = fromModel->getEdge(edgeGlobId);
                        edges_to_create.push_back(edge);
                        edgeId_to_create.push_back((unsigned int)Loc2GlobVec.size());

                        Loc2GlobVec.push_back(edgeGlobId);
                        Glob2LocMap[edgeGlobId] = (unsigned int)Loc2GlobVec.size() - 1;
                    }
                }
            }

            // add new edges to output topology
            m_outTopoModifier->addEdgesProcess(edges_to_create);
            m_outTopoModifier->addEdgesWarning(edges_to_create.size(), edges_to_create, edgeId_to_create);
            m_outTopoModifier->propagateTopologicalChanges();


            // remove edges not anymore on part of the border
            sofa::helper::vector< BaseMeshTopology::EdgeID > local_edgeId_to_remove;
            std::sort(edgeId_to_remove.begin(), edgeId_to_remove.end(), std::greater<BaseMeshTopology::EdgeID>());
            for (auto edgeGlobId : edgeId_to_remove)
            {

                std::map<unsigned int, unsigned int>::iterator iter_1 = Glob2LocMap.find(edgeGlobId);
                if (iter_1 == Glob2LocMap.end())
                {
                    msg_error() << " in TRIANGLESADDED process, edge id " << edgeGlobId << " not found in Glob2LocMap";
                    continue;
                }

                BaseMeshTopology::EdgeID edgeLocId = iter_1->second;
                BaseMeshTopology::EdgeID lastGlobId = Loc2GlobVec.back();

                // swap and pop loc2Glob vec
                Loc2GlobVec[edgeLocId] = lastGlobId;
                Loc2GlobVec.pop_back();

                // redirect glob2loc map
                Glob2LocMap.erase(iter_1);
                Glob2LocMap[lastGlobId] = edgeLocId;

                // add edge for output topology update
                local_edgeId_to_remove.push_back(edgeLocId);
            }

            // remove old edges
            m_outTopoModifier->removeEdges(local_edgeId_to_remove);

            break;
        }
        case core::topology::POINTSADDED:
        {
            const sofa::component::topology::PointsAdded *ta=static_cast< const sofa::component::topology::PointsAdded * >( *itBegin );
            m_outTopoModifier->addPointsProcess(ta->getNbAddedVertices());
            m_outTopoModifier->addPointsWarning(ta->getNbAddedVertices(), ta->ancestorsList, ta->coefs, false);
            m_outTopoModifier->propagateTopologicalChanges();
            break;
        }
        default:
            // Ignore events that are not Edge  related.
            break;
        };

        sofa::helper::AdvancedTimer::stepEnd(topoChangeType);
        ++itBegin;
    }
    Loc2GlobDataVec.endEdit();

    sofa::helper::AdvancedTimer::stepEnd("Update Triangle2EdgeTopologicalMapping");
}

} // namespace topology

} // namespace component

} // namespace sofa

