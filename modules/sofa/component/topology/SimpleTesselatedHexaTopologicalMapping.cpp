/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/topology/SimpleTesselatedHexaTopologicalMapping.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/HexahedronSetTopologyModifier.h>
#include <sofa/component/topology/HexahedronSetTopologyChange.h>


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
using namespace sofa::core::componentmodel::topology;

SOFA_DECL_CLASS ( SimpleTesselatedHexaTopologicalMapping )

// Register in the Factory
int SimpleTesselatedHexaTopologicalMappingClass = core::RegisterObject ( "Special case of mapping where HexahedronSetTopology is converted into a finer HexahedronSetTopology" )
        .add< SimpleTesselatedHexaTopologicalMapping >()
        ;

// Implementation
SimpleTesselatedHexaTopologicalMapping::SimpleTesselatedHexaTopologicalMapping ( In* from, Out* to )
    : TopologicalMapping ( from, to ),
      object1 ( initData ( &object1, std::string ( "../.." ), "object1", "First object to map" ) ),
      object2 ( initData ( &object2, std::string ( ".." ), "object2", "Second object to map" ) )
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
                toModel->addPoint(fromModel->getPX(i), fromModel->getPY(i), fromModel->getPZ(i));
                pointMappedFromPoint.push_back(i);
            }

            int n;
            int pointIndex = pointMappedFromPoint.size();
            pointMappedFromEdge.resize(fromModel->getNbHexahedra());
            pointMappedFromFacet.resize(fromModel->getNbHexahedra());
            hexasMappedFromHexa.resize(fromModel->getNbHexahedra());

            for (int i=0; i<fromModel->getNbHexahedra(); ++i)
            {
                Hexa h = fromModel->getHexahedron(i);

                Vec3d p0(fromModel->getPX(h[0]), fromModel->getPY(h[0]), fromModel->getPZ(h[0]));
                Vec3d p1(fromModel->getPX(h[1]), fromModel->getPY(h[1]), fromModel->getPZ(h[1]));
                Vec3d p2(fromModel->getPX(h[2]), fromModel->getPY(h[2]), fromModel->getPZ(h[2]));
                Vec3d p3(fromModel->getPX(h[3]), fromModel->getPY(h[3]), fromModel->getPZ(h[3]));
                Vec3d p4(fromModel->getPX(h[4]), fromModel->getPY(h[4]), fromModel->getPZ(h[4]));
                Vec3d p5(fromModel->getPX(h[5]), fromModel->getPY(h[5]), fromModel->getPZ(h[5]));
                Vec3d p6(fromModel->getPX(h[6]), fromModel->getPY(h[6]), fromModel->getPZ(h[6]));
                Vec3d p7(fromModel->getPX(h[7]), fromModel->getPY(h[7]), fromModel->getPZ(h[7]));

                // points mapped from edges
                Vec3d result = (p0+p1)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p1+p2)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p2+p3)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p3+p0)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p0+p4)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p1+p5)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p2+p6)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p7+p3)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p4+p5)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p5+p6)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p6+p7)/2;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p7+p4)/2;
                toModel->addPoint(result[0], result[1], result[2]);

                n=0;
                while(n<12)
                {
                    pointMappedFromEdge[i][n] = pointIndex;
                    pointIndex++;
                    n++;
                }

                // points mapped from facets
                result = (p0+p1+p2+p3)/4;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p0+p1+p4+p5)/4;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p1+p2+p5+p6)/4;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p2+p3+p6+p7)/4;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p0+p3+p4+p7)/4;
                toModel->addPoint(result[0], result[1], result[2]);
                result = (p4+p5+p6+p7)/4;
                toModel->addPoint(result[0], result[1], result[2]);

                n=0;
                while(n<6)
                {
                    pointMappedFromFacet[i][n] = pointIndex;
                    pointIndex++;
                    n++;
                }

                // points mapped from hexas
                result = (p0+p1+p2+p3+p4+p5+p6+p7)/8;
                toModel->addPoint(result[0], result[1], result[2]);

                pointMappedFromHexa.push_back(pointIndex);
                pointIndex++;
            }

            int hexaIndex = 0;

            for (int i=0; i<fromModel->getNbHexahedra(); ++i)
            {
                Hexa h = fromModel->getHexahedron(i);

                Vec3d p0(fromModel->getPX(h[0]), fromModel->getPY(h[0]), fromModel->getPZ(h[0]));
                Vec3d p1(fromModel->getPX(h[1]), fromModel->getPY(h[1]), fromModel->getPZ(h[1]));
                Vec3d p2(fromModel->getPX(h[2]), fromModel->getPY(h[2]), fromModel->getPZ(h[2]));
                Vec3d p3(fromModel->getPX(h[3]), fromModel->getPY(h[3]), fromModel->getPZ(h[3]));
                Vec3d p4(fromModel->getPX(h[4]), fromModel->getPY(h[4]), fromModel->getPZ(h[4]));
                Vec3d p5(fromModel->getPX(h[5]), fromModel->getPY(h[5]), fromModel->getPZ(h[5]));
                Vec3d p6(fromModel->getPX(h[6]), fromModel->getPY(h[6]), fromModel->getPZ(h[6]));
                Vec3d p7(fromModel->getPX(h[7]), fromModel->getPY(h[7]), fromModel->getPZ(h[7]));

                toModel->addHexa(h[0], pointMappedFromEdge[i][0], pointMappedFromFacet[i][0], pointMappedFromEdge[i][3],
                        pointMappedFromEdge[i][4], pointMappedFromFacet[i][1], pointMappedFromHexa[i], pointMappedFromFacet[i][4]);
                toModel->addHexa(pointMappedFromEdge[i][0], h[1], pointMappedFromEdge[i][1], pointMappedFromFacet[i][0],
                        pointMappedFromFacet[i][1], pointMappedFromEdge[i][5], pointMappedFromFacet[i][2], pointMappedFromHexa[i]);
                toModel->addHexa(pointMappedFromFacet[i][0], pointMappedFromEdge[i][1], h[2], pointMappedFromEdge[i][2],
                        pointMappedFromHexa[i], pointMappedFromFacet[i][2], pointMappedFromEdge[i][6], pointMappedFromFacet[i][3]);
                toModel->addHexa(pointMappedFromEdge[i][3], pointMappedFromFacet[i][0], pointMappedFromEdge[i][2], h[3],
                        pointMappedFromFacet[i][4], pointMappedFromHexa[i], pointMappedFromFacet[i][3], pointMappedFromEdge[i][7]);
                toModel->addHexa(pointMappedFromEdge[i][4], pointMappedFromFacet[i][1], pointMappedFromHexa[i], pointMappedFromFacet[i][4],
                        h[4], pointMappedFromEdge[i][8], pointMappedFromFacet[i][5], pointMappedFromEdge[i][11]);
                toModel->addHexa(pointMappedFromFacet[i][1], pointMappedFromEdge[i][5], pointMappedFromFacet[i][2], pointMappedFromHexa[i],
                        pointMappedFromEdge[i][8], h[5], pointMappedFromEdge[i][9], pointMappedFromFacet[i][5]);
                toModel->addHexa(pointMappedFromHexa[i], pointMappedFromFacet[i][2], pointMappedFromEdge[i][6], pointMappedFromFacet[i][3],
                        pointMappedFromFacet[i][5], pointMappedFromEdge[i][9], h[6], pointMappedFromEdge[i][10]);
                toModel->addHexa(pointMappedFromFacet[i][4], pointMappedFromHexa[i], pointMappedFromFacet[i][3], pointMappedFromEdge[i][7],
                        pointMappedFromEdge[i][11], pointMappedFromFacet[i][5], pointMappedFromEdge[i][10], h[7]);

                n=0;
                while(n<8)
                {
                    hexasMappedFromHexa[i][n] = hexaIndex;
                    hexaIndex++;
                    n++;
                }
            }
        }
    }
}

} // namespace topology
} // namespace component
} // namespace sofa

