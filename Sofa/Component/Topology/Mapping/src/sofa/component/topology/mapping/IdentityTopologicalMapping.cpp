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
#include <sofa/component/topology/mapping/IdentityTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/BaseState.h>

#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>
#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::topology::mapping
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology::mapping;
using namespace sofa::core::topology;

// Register in the Factory
int IdentityTopologicalMappingClass = core::RegisterObject("This class is a specific implementation of TopologicalMapping where the destination topology should be kept identical to the source topology. The implementation currently assumes that both topology have been initialized identically.")
        .add< IdentityTopologicalMapping >()

        ;

IdentityTopologicalMapping::IdentityTopologicalMapping()
{
}


IdentityTopologicalMapping::~IdentityTopologicalMapping()
{
}

void IdentityTopologicalMapping::init()
{
    sofa::core::topology::TopologicalMapping::init();
    this->updateLinks();
    if (fromModel && toModel)
    {

    }
}

Index IdentityTopologicalMapping::getFromIndex(Index ind)
{
    return ind;
}

void IdentityTopologicalMapping::updateTopologicalMappingTopDown()
{
    using namespace container::dynamic;

    if (!fromModel || !toModel) return;

    std::list<const TopologyChange *>::const_iterator itBegin=fromModel->beginChange();
    const std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

    if (itBegin == itEnd) return;

    PointSetTopologyModifier *toPointMod = nullptr;
    EdgeSetTopologyModifier *toEdgeMod = nullptr;
    TriangleSetTopologyModifier *toTriangleMod = nullptr;

    TriangleSetTopologyContainer *fromTriangleCon = nullptr;
    fromModel->getContext()->get(fromTriangleCon);


    TriangleSetTopologyContainer *toTriangleCon = nullptr;
    toModel->getContext()->get(toTriangleCon);

    msg_info() << "Begin: " << msgendl
               << "    Nb of points of fromModel : " << fromTriangleCon->getNbPoints() << msgendl
               << "    Nb of edges of fromModel : " << fromTriangleCon->getNbEdges() << msgendl
               << "    Nb of triangles of fromModel : " << fromTriangleCon->getNbTriangles() << msgendl
               << "    Nb of points of toModel : " << toTriangleCon->getNbPoints() << msgendl
               << "    Nb of edges of toModel : " << toTriangleCon->getNbEdges() << msgendl
               << "    Nb of triangles of toModel : " << toTriangleCon->getNbTriangles();

    toModel->getContext()->get(toPointMod);
    if (!toPointMod)
    {
         msg_error()<<"No PointSetTopologyModifier found for target topology." ;
        return;
    }

    while( itBegin != itEnd )
    {
        const TopologyChange* topoChange = *itBegin;
        const TopologyChangeType changeType = topoChange->getChangeType();

        switch( changeType )
        {

        case core::topology::ENDING_EVENT:
        {
            dmsg_info() << "ENDING_EVENT" ;
            toPointMod->notifyEndingEvent();
            break;
        }

        case core::topology::POINTSADDED:
        {
            const PointsAdded * pAdd = static_cast< const PointsAdded * >( topoChange );
            dmsg_info() << "POINTSADDED : " << pAdd->getNbAddedVertices() ;
            toPointMod->addPoints(pAdd->getNbAddedVertices(), pAdd->ancestorsList, pAdd->coefs, true);
            break;
        }
        case core::topology::POINTSREMOVED:
        {
            const PointsRemoved *pRem = static_cast< const PointsRemoved * >( topoChange );
            auto tab = pRem->getArray();
            dmsg_info() << "POINTSREMOVED : " << tab.size() ;
            toPointMod->removePoints(tab, true);
            break;
        }
        case core::topology::POINTSRENUMBERING:
        {
            const PointsRenumbering *pRenumber = static_cast< const PointsRenumbering * >( topoChange );
            const auto &tab = pRenumber->getIndexArray();
            const auto &inv_tab = pRenumber->getinv_IndexArray();
            dmsg_info() << "POINTSRENUMBERING : " << tab.size() ;
            toPointMod->renumberPoints(tab, inv_tab, true);
            break;
        }

        case core::topology::EDGESADDED:
        {
            if (!toEdgeMod) toModel->getContext()->get(toEdgeMod);
            if (!toEdgeMod) break;
            const EdgesAdded *eAdd = static_cast< const EdgesAdded * >( topoChange );
            dmsg_info() << "EDGESADDED : " << eAdd->getNbAddedEdges() ;
            toEdgeMod->addEdges(eAdd->edgeArray, eAdd->ancestorsList, eAdd->coefs);
            break;
        }

        case core::topology::EDGESREMOVED:
        {
            if (!toEdgeMod) toModel->getContext()->get(toEdgeMod);
            if (!toEdgeMod) break;
            const EdgesRemoved *eRem = static_cast< const EdgesRemoved * >( topoChange );
            auto tab = eRem->getArray();
            dmsg_info() << "EDGESREMOVED : " ;
            toEdgeMod->removeEdges(tab, false);
            break;
        }

        case core::topology::TRIANGLESADDED:
        {
            if (!toTriangleMod) toModel->getContext()->get(toTriangleMod);
            if (!toTriangleMod) break;
            const TrianglesAdded *tAdd = static_cast< const TrianglesAdded * >( topoChange );
            dmsg_info() << "TRIANGLESADDED : " << tAdd->getNbAddedTriangles() ;
            toTriangleMod->addTriangles(tAdd->triangleArray, tAdd->ancestorsList, tAdd->coefs);
            break;
        }

        case core::topology::TRIANGLESREMOVED:
        {
            if (!toTriangleMod) toModel->getContext()->get(toTriangleMod);
            if (!toTriangleMod) break;
            const TrianglesRemoved *tRem = static_cast< const TrianglesRemoved * >( topoChange );
            auto tab = tRem->getArray();
            dmsg_info() << "TRIANGLESREMOVED : " << tab.size() ;
            toTriangleMod->removeTriangles(tab, false, false);
            break;
        }

        default:
            break;
        };

        ++itBegin;
    }

    msg_info() << "End: "
               << "    Nb of points of fromModel : " << fromTriangleCon->getNbPoints() << msgendl
               << "    Nb of points of fromState : " << fromTriangleCon->getContext()->getState()->getSize() << msgendl
               << "    Nb of edges of fromModel : " << fromTriangleCon->getNbEdges() << msgendl
               << "    Nb of triangles of fromModel : " << fromTriangleCon->getNbTriangles() << msgendl
               << "    Nb of points of toModel : " << toTriangleCon->getNbPoints() << msgendl
               << "    Nb of points of toState : " << toTriangleCon->getContext()->getState()->getSize() << msgendl
               << "    Nb of edges of toModel : " << toTriangleCon->getNbEdges() << msgendl
               << "    Nb of triangles of toModel : " << toTriangleCon->getNbTriangles() ;

}

} //namespace sofa::component::topology::mapping
