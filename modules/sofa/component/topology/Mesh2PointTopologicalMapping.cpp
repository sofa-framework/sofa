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
#include <sofa/component/topology/Mesh2PointTopologicalMapping.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/PointSetTopologyModifier.h>
#include <sofa/component/topology/TetrahedronSetTopologyChange.h>
#include <sofa/component/topology/EdgeSetTopologyChange.h>
#include <sofa/component/topology/PointSetTopologyChange.h>
#include <sofa/component/topology/TriangleSetTopologyChange.h>
#include <sofa/component/topology/QuadSetTopologyChange.h>
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
using namespace sofa::core::topology;

SOFA_DECL_CLASS ( Mesh2PointTopologicalMapping )

// Register in the Factory
int Mesh2PointTopologicalMappingClass = core::RegisterObject ( "This class maps any mesh primitive (point, edge, triangle...) into a point using a relative position from the primitive" )
        .add< Mesh2PointTopologicalMapping >()
        ;

// Implementation
Mesh2PointTopologicalMapping::Mesh2PointTopologicalMapping ( In* from, Out* to )
    : TopologicalMapping ( from, to ),
      pointBaryCoords ( initData ( &pointBaryCoords, "pointBaryCoords", "Coordinates for the points of the output topology created from the points of the input topology" ) ),
      edgeBaryCoords ( initData ( &edgeBaryCoords, "edgeBaryCoords", "Coordinates for the points of the output topology created from the edges of the input topology" ) ),
      triangleBaryCoords ( initData ( &triangleBaryCoords, "triangleBaryCoords", "Coordinates for the points of the output topology created from the triangles of the input topology" ) ),
      quadBaryCoords ( initData ( &quadBaryCoords, "quadBaryCoords", "Coordinates for the points of the output topology created from the quads of the input topology" ) ),
      tetraBaryCoords ( initData ( &tetraBaryCoords, "tetraBaryCoords", "Coordinates for the points of the output topology created from the tetra of the input topology" ) ),
      hexaBaryCoords ( initData ( &hexaBaryCoords, "hexaBaryCoords", "Coordinates for the points of the output topology created from the hexa of the input topology" ) )
{
    pointBaryCoords.setGroup("BaryCoords");
    edgeBaryCoords.setGroup("BaryCoords");
    triangleBaryCoords.setGroup("BaryCoords");
    quadBaryCoords.setGroup("BaryCoords");
    tetraBaryCoords.setGroup("BaryCoords");
    hexaBaryCoords.setGroup("BaryCoords");
}

void Mesh2PointTopologicalMapping::init()
{
    if(fromModel)
    {
        if(toModel)
        {
            int toModelLastPointIndex = 0;

            // point to point mapping
            if (!pointBaryCoords.getValue().empty())
            {
                pointsMappedFrom[POINT].resize(fromModel->getNbPoints());
                for (int i=0; i<fromModel->getNbPoints(); i++)
                {
                    for (unsigned int j=0; j<pointBaryCoords.getValue().size(); j++)
                    {
                        toModel->addPoint(fromModel->getPX(i)+pointBaryCoords.getValue()[j][0], fromModel->getPY(i)+pointBaryCoords.getValue()[j][1], fromModel->getPZ(i)+pointBaryCoords.getValue()[j][2]);
                        pointsMappedFrom[POINT][i].push_back(toModelLastPointIndex);
                        pointSource.push_back(std::make_pair(POINT,i));
                        toModelLastPointIndex++;
                    }
                }
            }

            // edge to point mapping
            if (!edgeBaryCoords.getValue().empty())
            {
                pointsMappedFrom[EDGE].resize(fromModel->getNbEdges());
                for (int i=0; i<fromModel->getNbEdges(); i++)
                {
                    for (unsigned int j=0; j<edgeBaryCoords.getValue().size(); j++)
                    {
                        Edge e = fromModel->getEdge(i);

                        Vec3d p0(fromModel->getPX(e[0]), fromModel->getPY(e[0]), fromModel->getPZ(e[0]));
                        Vec3d p1(fromModel->getPX(e[1]), fromModel->getPY(e[1]), fromModel->getPZ(e[1]));

                        double fx = edgeBaryCoords.getValue()[j][0];

                        Vec3d result = p0 * (1-fx)
                                + p1 * fx;

                        toModel->addPoint(result[0], result[1], result[2]);

                        pointsMappedFrom[EDGE][i].push_back(toModelLastPointIndex);
                        pointSource.push_back(std::make_pair(EDGE,i));
                        toModelLastPointIndex++;
                    }
                }
            }

            // triangle to point mapping
            if (!triangleBaryCoords.getValue().empty())
            {
                pointsMappedFrom[TRIANGLE].resize(fromModel->getNbTriangles());
                for (int i=0; i<fromModel->getNbTriangles(); i++)
                {
                    for (unsigned int j=0; j<triangleBaryCoords.getValue().size(); j++)
                    {
                        Triangle t = fromModel->getTriangle(i);

                        Vec3d p0(fromModel->getPX(t[0]), fromModel->getPY(t[0]), fromModel->getPZ(t[0]));
                        Vec3d p1(fromModel->getPX(t[1]), fromModel->getPY(t[1]), fromModel->getPZ(t[1]));
                        Vec3d p2(fromModel->getPX(t[2]), fromModel->getPY(t[2]), fromModel->getPZ(t[2]));

                        double fx = triangleBaryCoords.getValue()[j][0];
                        double fy = triangleBaryCoords.getValue()[j][1];

                        Vec3d result =  p0 * (1-fx-fy)
                                + p1 * fx
                                + p2 * fy;

                        toModel->addPoint(result[0], result[1], result[2]);

                        pointsMappedFrom[TRIANGLE][i].push_back(toModelLastPointIndex);
                        pointSource.push_back(std::make_pair(TRIANGLE,i));
                        toModelLastPointIndex++;
                    }
                }
            }

            // quad to point mapping
            if (!quadBaryCoords.getValue().empty())
            {
                pointsMappedFrom[QUAD].resize(fromModel->getNbQuads());
                for (int i=0; i<fromModel->getNbQuads(); i++)
                {
                    for (unsigned int j=0; j<quadBaryCoords.getValue().size(); j++)
                    {
                        Quad q = fromModel->getQuad(i);

                        Vec3d p0(fromModel->getPX(q[0]), fromModel->getPY(q[0]), fromModel->getPZ(q[0]));
                        Vec3d p1(fromModel->getPX(q[1]), fromModel->getPY(q[1]), fromModel->getPZ(q[1]));
                        Vec3d p2(fromModel->getPX(q[2]), fromModel->getPY(q[2]), fromModel->getPZ(q[2]));
                        Vec3d p3(fromModel->getPX(q[3]), fromModel->getPY(q[3]), fromModel->getPZ(q[3]));

                        double fx = quadBaryCoords.getValue()[j][0];
                        double fy = quadBaryCoords.getValue()[j][1];

                        Vec3d result =  p0 * ((1-fx) * (1-fy))
                                + p1 * ((  fx) * (1-fy))
                                + p2 * ((1-fx) * (  fy))
                                + p3 * ((  fx) * (  fy));

                        toModel->addPoint(result[0], result[1], result[2]);

                        pointsMappedFrom[QUAD][i].push_back(toModelLastPointIndex);
                        pointSource.push_back(std::make_pair(QUAD,i));
                        toModelLastPointIndex++;
                    }
                }
            }

            // tetrahedron to point mapping
            if (!tetraBaryCoords.getValue().empty())
            {
                pointsMappedFrom[TETRA].resize(fromModel->getNbTetrahedra());
                for (int i=0; i<fromModel->getNbTetrahedra(); i++)
                {
                    for (unsigned int j=0; j<tetraBaryCoords.getValue().size(); j++)
                    {
                        Tetra t = fromModel->getTetrahedron(i);

                        Vec3d p0(fromModel->getPX(t[0]), fromModel->getPY(t[0]), fromModel->getPZ(t[0]));
                        Vec3d p1(fromModel->getPX(t[1]), fromModel->getPY(t[1]), fromModel->getPZ(t[1]));
                        Vec3d p2(fromModel->getPX(t[2]), fromModel->getPY(t[2]), fromModel->getPZ(t[2]));
                        Vec3d p3(fromModel->getPX(t[3]), fromModel->getPY(t[3]), fromModel->getPZ(t[3]));

                        double fx = tetraBaryCoords.getValue()[j][0];
                        double fy = tetraBaryCoords.getValue()[j][1];
                        double fz = tetraBaryCoords.getValue()[j][2];

                        Vec3d result =  p0 * (1-fx-fy-fz)
                                + p1 * fx
                                + p2 * fy
                                + p3 * fz;

                        toModel->addPoint(result[0], result[1], result[2]);

                        pointsMappedFrom[TETRA][i].push_back(toModelLastPointIndex);
                        pointSource.push_back(std::make_pair(TETRA,i));
                        toModelLastPointIndex++;
                    }
                }
            }

            // hexahedron to point mapping
            if (!hexaBaryCoords.getValue().empty())
            {
                pointsMappedFrom[HEXA].resize(fromModel->getNbHexahedra());
                for (int i=0; i<fromModel->getNbHexahedra(); i++)
                {
                    for (unsigned int j=0; j<hexaBaryCoords.getValue().size(); j++)
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

                        double fx = hexaBaryCoords.getValue()[j][0];
                        double fy = hexaBaryCoords.getValue()[j][1];
                        double fz = hexaBaryCoords.getValue()[j][2];

                        Vec3d result =  p0 * ((1-fx) * (1-fy) * (1-fz))
                                + p1 * ((  fx) * (1-fy) * (1-fz))
                                + p2 * ((1-fx) * (  fy) * (1-fz))
                                + p3 * ((  fx) * (  fy) * (1-fz))
                                + p4 * ((1-fx) * (1-fy) * (  fz))
                                + p5 * ((  fx) * (1-fy) * (  fz))
                                + p6 * ((  fx) * (  fy) * (  fz))
                                + p7 * ((1-fx) * (  fy) * (  fz));

                        toModel->addPoint(result[0], result[1], result[2]);

                        pointsMappedFrom[HEXA][i].push_back(toModelLastPointIndex);
                        pointSource.push_back(std::make_pair(HEXA,i));
                        toModelLastPointIndex++;
                    }
                }
            }
        }
    }
}

void Mesh2PointTopologicalMapping::updateTopologicalMappingTopDown()
{
    if(fromModel && toModel)
    {
        std::list<const TopologyChange *>::const_iterator changeIt=fromModel->beginChange();
        std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

        PointSetTopologyModifier *to_pstm;
        toModel->getContext()->get(to_pstm);

        while( changeIt != itEnd )
        {
            TopologyChangeType changeType = (*changeIt)->getChangeType();

            switch( changeType )
            {
            case core::topology::POINTSINDICESSWAP:
            {
                unsigned int i1 = ( static_cast< const PointsIndicesSwap * >( *changeIt ) )->index[0];
                unsigned int i2 = ( static_cast< const PointsIndicesSwap* >( *changeIt ) )->index[1];
//				sout << "INPUT SWAP POINTS "<<i1 << " " << i2 << sendl;
                swapInput(POINT,i1,i2);
                break;
            }
            case core::topology::POINTSADDED:
            {
                /// @TODO
                break;
            }
            case core::topology::POINTSREMOVED:
            {
                const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRemoved * >( *changeIt ) )->getArray();
//				 sout << "INPUT REMOVE POINTS "<<tab << sendl;
                removeInput(POINT, tab );
                break;
            }
            case core::topology::POINTSRENUMBERING:
            {
                const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRenumbering * >( *changeIt ) )->getinv_IndexArray();
//				 sout << "INPUT RENUMBER POINTS "<<tab << sendl;
                renumberInput(POINT, tab );
                break;
            }
            case core::topology::EDGESADDED:
            {
                /// @TODO
                break;
            }
            case core::topology::EDGESREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const EdgesRemoved *>( *changeIt ) )->getArray();
//				sout << "INPUT REMOVE EDGES "<<tab << sendl;
                removeInput(EDGE, tab );
                break;
            }
            case core::topology::TRIANGLESADDED:
            {
                /// @TODO
                break;
            }
            case core::topology::TRIANGLESREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TrianglesRemoved *>( *changeIt ) )->getArray();
//				sout << "INPUT REMOVE TRIANGLES "<<tab << sendl;
                removeInput(TRIANGLE, tab );
                break;
            }
            case core::topology::QUADSADDED:
            {
                /// @TODO
                break;
            }
            case core::topology::QUADSREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const QuadsRemoved *>( *changeIt ) )->getArray();
//				sout << "INPUT REMOVE QUADS "<<tab << sendl;
                removeInput(QUAD, tab );
                break;
            }
            case core::topology::TETRAHEDRAADDED:
            {
                /// @TODO
                break;
            }
            case core::topology::TETRAHEDRAREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TetrahedraRemoved *>( *changeIt ) )->getArray();
//				sout << "INPUT REMOVE TETRAHEDRA "<<tab << sendl;
                removeInput(TETRA, tab );
                break;
            }
            case core::topology::HEXAHEDRAADDED:
            {
                /// @TODO
                break;
            }
            case core::topology::HEXAHEDRAREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const HexahedraRemoved *>( *changeIt ) )->getArray();
//				sout << "INPUT REMOVE HEXAHEDRA "<<tab << sendl;
                removeInput(HEXA, tab );
                break;
            }
            case core::topology::ENDING_EVENT:
            {
                pointsToRemove.erase(BaseMeshTopology::InvalidID);
                if (to_pstm != NULL && !pointsToRemove.empty())
                {
                    sofa::helper::vector<unsigned int> vitems;
                    vitems.reserve(pointsToRemove.size());
                    vitems.insert(vitems.end(), pointsToRemove.rbegin(), pointsToRemove.rend());

                    to_pstm->removePointsWarning(vitems);
                    to_pstm->propagateTopologicalChanges();
                    to_pstm->removePointsProcess(vitems);
                    removeOutputPoints(vitems);
                    pointsToRemove.clear();
                }

                break;
            }

            default:
                break;

            }
            ++changeIt;
        }
    }
}

void Mesh2PointTopologicalMapping::swapInput(Element elem, int i1, int i2)
{
    if (pointsMappedFrom[elem].empty()) return;
    vector<int> i1Map = pointsMappedFrom[elem][i1];
    vector<int> i2Map = pointsMappedFrom[elem][i2];

    pointsMappedFrom[elem][i1] = i2Map;
    for(unsigned int i = 0; i < i2Map.size(); ++i)
    {
        if (i2Map[i] != -1) pointSource[i2Map[i]].second = i1;
    }

    pointsMappedFrom[elem][i2] = i1Map;
    for(unsigned int i = 0; i < i1Map.size(); ++i)
    {
        if (i1Map[i] != -1) pointSource[i1Map[i]].second = i2;
    }
}

void Mesh2PointTopologicalMapping::removeInput(Element elem,  const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFrom[elem].empty()) return;
    unsigned int last = pointsMappedFrom[elem].size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapInput(elem, index[i], last );
        for (unsigned int j = 0; j < pointsMappedFrom[elem][last].size(); ++j)
        {
            int map = pointsMappedFrom[elem][last][j];
            if (map != -1)
            {
                pointsToRemove.insert(map);
                pointSource[map].second = -1;
            }
        }
        --last;
    }

    pointsMappedFrom[elem].resize( last + 1 );
}

void Mesh2PointTopologicalMapping::renumberInput(Element elem, const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFrom[elem].empty()) return;
    helper::vector< vector<int> > copy = pointsMappedFrom[elem];
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        const vector<int>& map = copy[index[i]];
        pointsMappedFrom[elem][i] = map;
        for (unsigned int j = 0; j < map.size(); ++j)
        {
            int m = map[j];
            if (m != -1)
                pointSource[m].second = i;
        }
    }
}

void Mesh2PointTopologicalMapping::swapOutputPoints(int i1, int i2, bool removeLast)
{
    std::pair<Element, int> i1Source = pointSource[i1];
    std::pair<Element, int> i2Source = pointSource[i2];
    pointSource[i1] = i2Source;
    pointSource[i2] = i1Source;
    if (i1Source.second != -1)
    {
        // replace i1 by i2 in pointsMappedFrom[i1Source.first][i1Source.second]
        vector<int> & pts = pointsMappedFrom[i1Source.first][i1Source.second];
        for (unsigned int j = 0; j < pts.size(); ++j)
        {
            if (pts[j] == i1)
            {
                if (removeLast)
                    pts[j] = -1;
                else
                    pts[j] = i2;
            }
        }
    }
    if (i2Source.second != -1)
    {
        // replace i2 by i1 in pointsMappedFrom[i2Source.first][i1Source.second]
        vector<int> & pts = pointsMappedFrom[i2Source.first][i2Source.second];
        for (unsigned int j = 0; j < pts.size(); ++j)
        {
            if (pts[j] == i2)
                pts[j] = i1;
        }
    }
}

void Mesh2PointTopologicalMapping::removeOutputPoints( const sofa::helper::vector<unsigned int>& index )
{
    unsigned int last = pointSource.size() - 1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapOutputPoints( index[i], last, true );
        --last;
    }

    pointSource.resize(last + 1);
}

} // namespace topology
} // namespace component
} // namespace sofa

