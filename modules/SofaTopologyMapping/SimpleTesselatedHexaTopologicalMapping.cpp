/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <SofaTopologyMapping/SimpleTesselatedHexaTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

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
using sofa::helper::fixed_array;

SOFA_DECL_CLASS ( SimpleTesselatedHexaTopologicalMapping )

// Register in the Factory
int SimpleTesselatedHexaTopologicalMappingClass = core::RegisterObject ( "Special case of mapping where HexahedronSetTopology is converted into a finer HexahedronSetTopology" )
        .add< SimpleTesselatedHexaTopologicalMapping >()
        ;

// Implementation
SimpleTesselatedHexaTopologicalMapping::SimpleTesselatedHexaTopologicalMapping()
{
}

void SimpleTesselatedHexaTopologicalMapping::init()
{
    if(fromModel)
    {
        if(toModel)
        {
            toModel->clear();

            for (int i=0; i<fromModel->getNbPoints(); ++i)
            {
                // points mapped from points
                pointMappedFromPoint.push_back(i);
                toModel->addPoint(fromModel->getPX(i), fromModel->getPY(i), fromModel->getPZ(i));
            }

            int pointIndex = pointMappedFromPoint.size();
            Vector3 pA, pB, p;

            for (int i=0; i<fromModel->getNbHexahedra(); ++i)
            {
                core::topology::BaseMeshTopology::Hexa h = fromModel->getHexahedron(i);

                Vector3 p0(fromModel->getPX(h[0]), fromModel->getPY(h[0]), fromModel->getPZ(h[0]));
                Vector3 p1(fromModel->getPX(h[1]), fromModel->getPY(h[1]), fromModel->getPZ(h[1]));
                Vector3 p2(fromModel->getPX(h[2]), fromModel->getPY(h[2]), fromModel->getPZ(h[2]));
                Vector3 p3(fromModel->getPX(h[3]), fromModel->getPY(h[3]), fromModel->getPZ(h[3]));
                Vector3 p4(fromModel->getPX(h[4]), fromModel->getPY(h[4]), fromModel->getPZ(h[4]));
                Vector3 p5(fromModel->getPX(h[5]), fromModel->getPY(h[5]), fromModel->getPZ(h[5]));
                Vector3 p6(fromModel->getPX(h[6]), fromModel->getPY(h[6]), fromModel->getPZ(h[6]));
                Vector3 p7(fromModel->getPX(h[7]), fromModel->getPY(h[7]), fromModel->getPZ(h[7]));

                // points mapped from edges
                std::pair<std::map<fixed_array<int,2>, int>::iterator, bool> insert_result;
                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[0],h[1]),pointIndex));
                if(insert_result.second)
                {
                    p = (p0+p1)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[1],h[(2)]),pointIndex));
                if(insert_result.second)
                {
                    p = (p1+p2)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[3],h[2]),pointIndex));
                if(insert_result.second)
                {
                    p = (p3+p2)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[0],h[3]),pointIndex));
                if(insert_result.second)
                {
                    p = (p0+p3)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[0],h[4]),pointIndex));
                if(insert_result.second)
                {
                    p = (p0+p4)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[1],h[5]),pointIndex));
                if(insert_result.second)
                {
                    p = (p1+p5)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[2],h[6]),pointIndex));
                if(insert_result.second)
                {
                    p = (p2+p6)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[3],h[7]),pointIndex));
                if(insert_result.second)
                {
                    p = (p3+p7)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[4],h[5]),pointIndex));
                if(insert_result.second)
                {
                    p = (p4+p5)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[5],h[6]),pointIndex));
                if(insert_result.second)
                {
                    p = (p5+p6)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[7],h[6]),pointIndex));
                if(insert_result.second)
                {
                    p = (p7+p6)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_result = pointMappedFromEdge.insert(std::make_pair(fixed_array<int,2>(h[4],h[7]),pointIndex));
                if(insert_result.second)
                {
                    p = (p4+p7)/2;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                // points mapped from facets
                std::pair<std::map<fixed_array<int,4>, int>::iterator, bool> insert_facets_result;
                insert_facets_result = pointMappedFromFacet.insert(std::make_pair(fixed_array<int,4>(h[0], h[1], h[2], h[3]), pointIndex));
                if (insert_facets_result.second)
                {
                    p = (p0+p1+p2+p3)/4;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_facets_result = pointMappedFromFacet.insert(std::make_pair(fixed_array<int,4>(h[0], h[1], h[5], h[4]), pointIndex));
                if (insert_facets_result.second)
                {
                    p = (p0+p1+p5+p4)/4;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_facets_result = pointMappedFromFacet.insert(std::make_pair(fixed_array<int,4>(h[1], h[2], h[6], h[5]), pointIndex));
                if (insert_facets_result.second)
                {
                    p = (p1+p2+p6+p5)/4;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_facets_result = pointMappedFromFacet.insert(std::make_pair(fixed_array<int,4>(h[3], h[2], h[6], h[7]), pointIndex));
                if (insert_facets_result.second)
                {
                    p = (p3+p2+p6+p7)/4;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_facets_result = pointMappedFromFacet.insert(std::make_pair(fixed_array<int,4>(h[0], h[3], h[7], h[4]), pointIndex));
                if (insert_facets_result.second)
                {
                    p = (p0+p4+p7+p3)/4;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                insert_facets_result = pointMappedFromFacet.insert(std::make_pair(fixed_array<int,4>(h[4], h[5], h[6], h[7]), pointIndex));
                if (insert_facets_result.second)
                {
                    p = (p4+p5+p6+p7)/4;
                    toModel->addPoint(p[0], p[1], p[2]);
                    pointIndex++;
                }

                // points mapped from hexahedra
                pointMappedFromHexa.push_back(pointIndex);
                p = (p0+p1+p2+p3+p4+p5+p6+p7)/8;
                toModel->addPoint(p[0], p[1], p[2]);
                pointIndex++;
            }

            for (int i=0; i<fromModel->getNbHexahedra(); ++i)
            {
                core::topology::BaseMeshTopology::Hexa h = fromModel->getHexahedron(i);

                Vec3d p0(fromModel->getPX(h[0]), fromModel->getPY(h[0]), fromModel->getPZ(h[0]));
                Vec3d p1(fromModel->getPX(h[1]), fromModel->getPY(h[1]), fromModel->getPZ(h[1]));
                Vec3d p2(fromModel->getPX(h[2]), fromModel->getPY(h[2]), fromModel->getPZ(h[2]));
                Vec3d p3(fromModel->getPX(h[3]), fromModel->getPY(h[3]), fromModel->getPZ(h[3]));
                Vec3d p4(fromModel->getPX(h[4]), fromModel->getPY(h[4]), fromModel->getPZ(h[4]));
                Vec3d p5(fromModel->getPX(h[5]), fromModel->getPY(h[5]), fromModel->getPZ(h[5]));
                Vec3d p6(fromModel->getPX(h[6]), fromModel->getPY(h[6]), fromModel->getPZ(h[6]));
                Vec3d p7(fromModel->getPX(h[7]), fromModel->getPY(h[7]), fromModel->getPZ(h[7]));

                toModel->addHexa(h[0],
                        pointMappedFromEdge[fixed_array<int,2>(h[0],h[1])],
                        pointMappedFromFacet[fixed_array<int,4>(h[0],h[1],h[2],h[3])],
                        pointMappedFromEdge[fixed_array<int,2>(h[0],h[3])],
                        pointMappedFromEdge[fixed_array<int,2>(h[0],h[4])],
                        pointMappedFromFacet[fixed_array<int,4>(h[0],h[1],h[5],h[4])],
                        pointMappedFromHexa[i],
                        pointMappedFromFacet[fixed_array<int,4>(h[0],h[3],h[7],h[4])]);

                toModel->addHexa(pointMappedFromEdge[fixed_array<int,2>(h[0],h[1])],
                        h[1],
                        pointMappedFromEdge[fixed_array<int,2>(h[1],h[2])],
                        pointMappedFromFacet[fixed_array<int,4>(h[0],h[1],h[2],h[3])],
                        pointMappedFromFacet[fixed_array<int,4>(h[0],h[1],h[5],h[4])],
                        pointMappedFromEdge[fixed_array<int,2>(h[1],h[5])],
                        pointMappedFromFacet[fixed_array<int,4>(h[1],h[2],h[6],h[5])],
                        pointMappedFromHexa[i]);

                toModel->addHexa(pointMappedFromFacet[fixed_array<int,4>(h[0],h[1],h[2],h[3])],
                        pointMappedFromEdge[fixed_array<int,2>(h[1],h[2])],
                        h[2],
                        pointMappedFromEdge[fixed_array<int,2>(h[3],h[2])],
                        pointMappedFromHexa[i],
                        pointMappedFromFacet[fixed_array<int,4>(h[1],h[2],h[6],h[5])],
                        pointMappedFromEdge[fixed_array<int,2>(h[2],h[6])],
                        pointMappedFromFacet[fixed_array<int,4>(h[3],h[2],h[6],h[7])]);

                toModel->addHexa(pointMappedFromEdge[fixed_array<int,2>(h[0],h[3])],
                        pointMappedFromFacet[fixed_array<int,4>(h[0],h[1],h[2],h[3])],
                        pointMappedFromEdge[fixed_array<int,2>(h[3],h[2])],
                        h[3],
                        pointMappedFromFacet[fixed_array<int,4>(h[0],h[3],h[7],h[4])],
                        pointMappedFromHexa[i],
                        pointMappedFromFacet[fixed_array<int,4>(h[3],h[2],h[6],h[7])],
                        pointMappedFromEdge[fixed_array<int,2>(h[3],h[7])]);

                toModel->addHexa(pointMappedFromEdge[fixed_array<int,2>(h[0],h[4])],
                        pointMappedFromFacet[fixed_array<int,4>(h[0],h[1],h[5],h[4])],
                        pointMappedFromHexa[i],
                        pointMappedFromFacet[fixed_array<int,4>(h[0],h[3],h[7],h[4])],
                        h[4],
                        pointMappedFromEdge[fixed_array<int,2>(h[4],h[5])],
                        pointMappedFromFacet[fixed_array<int,4>(h[4],h[5],h[6],h[7])],
                        pointMappedFromEdge[fixed_array<int,2>(h[4],h[7])]);

                toModel->addHexa(pointMappedFromFacet[fixed_array<int,4>(h[0],h[1],h[5],h[4])],
                        pointMappedFromEdge[fixed_array<int,2>(h[1],h[5])],
                        pointMappedFromFacet[fixed_array<int,4>(h[1],h[2],h[6],h[5])],
                        pointMappedFromHexa[i],
                        pointMappedFromEdge[fixed_array<int,2>(h[4],h[5])],
                        h[5],
                        pointMappedFromEdge[fixed_array<int,2>(h[5],h[6])],
                        pointMappedFromFacet[fixed_array<int,4>(h[4],h[5],h[6],h[7])]);

                toModel->addHexa(pointMappedFromHexa[i],
                        pointMappedFromFacet[fixed_array<int,4>(h[1],h[2],h[6],h[5])],
                        pointMappedFromEdge[fixed_array<int,2>(h[2],h[6])],
                        pointMappedFromFacet[fixed_array<int,4>(h[3],h[2],h[6],h[7])],
                        pointMappedFromFacet[fixed_array<int,4>(h[4],h[5],h[6],h[7])],
                        pointMappedFromEdge[fixed_array<int,2>(h[5],h[6])],
                        h[6],
                        pointMappedFromEdge[fixed_array<int,2>(h[7],h[6])]);

                toModel->addHexa(pointMappedFromFacet[fixed_array<int,4>(h[0],h[3],h[7],h[4])],
                        pointMappedFromHexa[i],
                        pointMappedFromFacet[fixed_array<int,4>(h[3],h[2],h[6],h[7])],
                        pointMappedFromEdge[fixed_array<int,2>(h[3],h[7])],
                        pointMappedFromEdge[fixed_array<int,2>(h[4],h[7])],
                        pointMappedFromFacet[fixed_array<int,4>(h[4],h[5],h[6],h[7])],
                        pointMappedFromEdge[fixed_array<int,2>(h[7],h[6])],
                        h[7]);
            }
        }
    }
}

} // namespace topology
} // namespace component
} // namespace sofa

