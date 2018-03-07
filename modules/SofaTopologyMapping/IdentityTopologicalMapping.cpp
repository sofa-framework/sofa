/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTopologyMapping/IdentityTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>

#include <SofaBaseTopology/QuadSetTopologyContainer.h>
#include <SofaBaseTopology/QuadSetTopologyModifier.h>

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>

#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/HexahedronSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>
#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology;
using namespace sofa::core::topology;

SOFA_DECL_CLASS(IdentityTopologicalMapping)

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

unsigned int IdentityTopologicalMapping::getFromIndex(unsigned int ind)
{
    return ind;
}

void IdentityTopologicalMapping::updateTopologicalMappingTopDown()
{
    if (!fromModel || !toModel) return;

    std::list<const TopologyChange *>::const_iterator itBegin=fromModel->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

    if (itBegin == itEnd) return;

    PointSetTopologyModifier *toPointMod = NULL;
    EdgeSetTopologyModifier *toEdgeMod = NULL;
    TriangleSetTopologyModifier *toTriangleMod = NULL;

    TriangleSetTopologyContainer *fromTriangleCon = NULL;
    fromModel->getContext()->get(fromTriangleCon);


    TriangleSetTopologyContainer *toTriangleCon = NULL;
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
        serr << "No PointSetTopologyModifier found for target topology." << sendl;
        return;
    }

    while( itBegin != itEnd )
    {
        const TopologyChange* topoChange = *itBegin;
        TopologyChangeType changeType = topoChange->getChangeType();

        switch( changeType )
        {

        case core::topology::ENDING_EVENT:
        {
            dmsg_info() << "ENDING_EVENT" ;
            toPointMod->propagateTopologicalChanges();
            toPointMod->notifyEndingEvent();
            toPointMod->propagateTopologicalChanges();
            break;
        }

        case core::topology::POINTSADDED:
        {
            const PointsAdded * pAdd = static_cast< const PointsAdded * >( topoChange );
            dmsg_info() << "POINTSADDED : " << pAdd->getNbAddedVertices() ;
            toPointMod->addPointsProcess(pAdd->getNbAddedVertices());
            toPointMod->addPointsWarning(pAdd->getNbAddedVertices(), pAdd->ancestorsList, pAdd->coefs, true);
            toPointMod->propagateTopologicalChanges();
            break;
        }
        case core::topology::POINTSREMOVED:
        {
            const PointsRemoved *pRem = static_cast< const PointsRemoved * >( topoChange );
            sofa::helper::vector<unsigned int> tab = pRem->getArray();
            dmsg_info() << "POINTSREMOVED : " << tab.size() ;
            toPointMod->removePointsWarning(tab, true);
            toPointMod->propagateTopologicalChanges();
            toPointMod->removePointsProcess(tab, true);
            break;
        }
        case core::topology::POINTSRENUMBERING:
        {
            const PointsRenumbering *pRenumber = static_cast< const PointsRenumbering * >( topoChange );
            const sofa::helper::vector<unsigned int> &tab = pRenumber->getIndexArray();
            const sofa::helper::vector<unsigned int> &inv_tab = pRenumber->getinv_IndexArray();
            dmsg_info() << "POINTSRENUMBERING : " << tab.size() ;
            toPointMod->renumberPointsWarning(tab, inv_tab, true);
            toPointMod->propagateTopologicalChanges();
            toPointMod->renumberPointsProcess(tab, inv_tab, true);
            break;
        }

        case core::topology::EDGESADDED:
        {
            if (!toEdgeMod) toModel->getContext()->get(toEdgeMod);
            if (!toEdgeMod) break;
            const EdgesAdded *eAdd = static_cast< const EdgesAdded * >( topoChange );
            dmsg_info() << "EDGESADDED : " << eAdd->getNbAddedEdges() ;
            toEdgeMod->addEdgesProcess(eAdd->edgeArray);
            toEdgeMod->addEdgesWarning(eAdd->getNbAddedEdges(), eAdd->edgeArray, eAdd->edgeIndexArray, eAdd->ancestorsList, eAdd->coefs);
            toEdgeMod->propagateTopologicalChanges();
            break;
        }

        case core::topology::EDGESREMOVED:
        {
            if (!toEdgeMod) toModel->getContext()->get(toEdgeMod);
            if (!toEdgeMod) break;
            const EdgesRemoved *eRem = static_cast< const EdgesRemoved * >( topoChange );
            sofa::helper::vector<unsigned int> tab = eRem->getArray();
            dmsg_info() << "EDGESREMOVED : " ;
            toEdgeMod->removeEdgesWarning(tab);
            toEdgeMod->propagateTopologicalChanges();
            toEdgeMod->removeEdgesProcess(tab, false);
            break;
        }

        case core::topology::TRIANGLESADDED:
        {
            if (!toTriangleMod) toModel->getContext()->get(toTriangleMod);
            if (!toTriangleMod) break;
            const TrianglesAdded *tAdd = static_cast< const TrianglesAdded * >( topoChange );
            dmsg_info() << "TRIANGLESADDED : " << tAdd->getNbAddedTriangles() ;
            toTriangleMod->addTrianglesProcess(tAdd->triangleArray);
            toTriangleMod->addTrianglesWarning(tAdd->getNbAddedTriangles(), tAdd->triangleArray, tAdd->triangleIndexArray, tAdd->ancestorsList, tAdd->coefs);
            toTriangleMod->propagateTopologicalChanges();
            break;
        }

        case core::topology::TRIANGLESREMOVED:
        {
            if (!toTriangleMod) toModel->getContext()->get(toTriangleMod);
            if (!toTriangleMod) break;
            const TrianglesRemoved *tRem = static_cast< const TrianglesRemoved * >( topoChange );
            sofa::helper::vector<unsigned int> tab = tRem->getArray();
            dmsg_info() << "TRIANGLESREMOVED : " << tab.size() ;
            toTriangleMod->removeTrianglesWarning(tab);
            toTriangleMod->propagateTopologicalChanges();
            toTriangleMod->removeTrianglesProcess(tab, false);
            break;
        }

        default:
            break;
        };

        ++itBegin;
    }
    toPointMod->propagateTopologicalChanges();

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

} // namespace topology

} // namespace component

} // namespace sofa

