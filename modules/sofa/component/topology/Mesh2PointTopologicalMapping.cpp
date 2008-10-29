/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
using namespace sofa::core::componentmodel::topology;

SOFA_DECL_CLASS ( Mesh2PointTopologicalMapping )

// Register in the Factory
int Mesh2PointTopologicalMappingClass = core::RegisterObject ( "" )
        .add< Mesh2PointTopologicalMapping >()
        ;

// Implementation
Mesh2PointTopologicalMapping::Mesh2PointTopologicalMapping ( In* from, Out* to )
    : TopologicalMapping ( from, to ),
      object1 ( initData ( &object1, std::string ( "../.." ), "object1", "First object to map" ) ),
      object2 ( initData ( &object2, std::string ( ".." ), "object2", "Second object to map" ) ),
      pointBaryCoords ( initData ( &pointBaryCoords, "pointBaryCoords", "Coordinates for the points of the output topology created from the points of the input topology" ) ),
      edgeBaryCoords ( initData ( &edgeBaryCoords, "edgeBaryCoords", "Coordinates for the points of the output topology created from the edges of the input topology" ) ),
      triangleBaryCoords ( initData ( &triangleBaryCoords, "triangleBaryCoords", "Coordinates for the points of the output topology created from the triangles of the input topology" ) ),
      quadBaryCoords ( initData ( &quadBaryCoords, "quadBaryCoords", "Coordinates for the points of the output topology created from the quads of the input topology" ) ),
      tetraBaryCoords ( initData ( &tetraBaryCoords, "tetraBaryCoords", "Coordinates for the points of the output topology created from the tetra of the input topology" ) ),
      hexaBaryCoords ( initData ( &hexaBaryCoords, "hexaBaryCoords", "Coordinates for the points of the output topology created from the hexa of the input topology" ) )
{
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
                pointsMappedFromPoint.resize(fromModel->getNbPoints());
                for (int i=0; i<fromModel->getNbPoints(); i++)
                {
                    for (unsigned int j=0; j<pointBaryCoords.getValue().size(); j++)
                    {
                        toModel->addPoint(fromModel->getPX(i)+pointBaryCoords.getValue()[j][0], fromModel->getPY(i)+pointBaryCoords.getValue()[j][1], fromModel->getPZ(i)+pointBaryCoords.getValue()[j][2]);
                        pointsMappedFromPoint[i].push_back(toModelLastPointIndex);
                        toModelLastPointIndex++;
                    }
                }
            }

            // edge to point mapping
            if (!edgeBaryCoords.getValue().empty())
            {
                pointsMappedFromEdge.resize(fromModel->getNbEdges());
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

                        pointsMappedFromEdge[i].push_back(toModelLastPointIndex);
                        toModelLastPointIndex++;
                    }
                }
            }

            // triangle to point mapping
            if (!triangleBaryCoords.getValue().empty())
            {
                pointsMappedFromTriangle.resize(fromModel->getNbTriangles());
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

                        pointsMappedFromTriangle[i].push_back(toModelLastPointIndex);
                        toModelLastPointIndex++;
                    }
                }
            }

            // quad to point mapping
            if (!quadBaryCoords.getValue().empty())
            {
                pointsMappedFromQuad.resize(fromModel->getNbQuads());
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

                        pointsMappedFromQuad[i].push_back(toModelLastPointIndex);
                        toModelLastPointIndex++;
                    }
                }
            }

            // tetrahedron to point mapping
            if (!tetraBaryCoords.getValue().empty())
            {
                pointsMappedFromTetra.resize(fromModel->getNbTetras());
                for (int i=0; i<fromModel->getNbTetras(); i++)
                {
                    for (unsigned int j=0; j<tetraBaryCoords.getValue().size(); j++)
                    {
                        Tetra t = fromModel->getTetra(i);

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

                        pointsMappedFromTetra[i].push_back(toModelLastPointIndex);
                        toModelLastPointIndex++;
                    }
                }
            }

            // hexahedron to point mapping
            if (!tetraBaryCoords.getValue().empty())
            {
                pointsMappedFromHexa.resize(fromModel->getNbHexas());
                for (int i=0; i<fromModel->getNbHexas(); i++)
                {
                    for (unsigned int j=0; j<hexaBaryCoords.getValue().size(); j++)
                    {
                        Hexa h = fromModel->getHexa(i);

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

                        pointsMappedFromHexa[i].push_back(toModelLastPointIndex);
                        toModelLastPointIndex++;
                    }
                }
            }
            nbOutputPoints = toModelLastPointIndex;
        }
    }
}

void Mesh2PointTopologicalMapping::updateTopologicalMappingTopDown()
{
    if(fromModel && toModel)
    {
        std::list<const TopologyChange *>::const_iterator changeIt=fromModel->firstChange();
        std::list<const TopologyChange *>::const_iterator itEnd=fromModel->lastChange();

        PointSetTopologyModifier *to_pstm;
        toModel->getContext()->get(to_pstm);

        while( changeIt != itEnd )
        {
            TopologyChangeType changeType = (*changeIt)->getChangeType();

            switch( changeType )
            {
            case core::componentmodel::topology::POINTSINDICESSWAP:
            {
                unsigned int i1 = ( static_cast< const PointsIndicesSwap * >( *changeIt ) )->index[0];
                unsigned int i2 = ( static_cast< const PointsIndicesSwap* >( *changeIt ) )->index[1];
                std::cout << "INPUT SWAP POINTS "<<i1 << " " << i2 << std::endl;
                swapInputPoints(i1,i2);
                break;
            }
            case core::componentmodel::topology::POINTSADDED:
            {
                /// @TODO
                break;
            }
            case core::componentmodel::topology::POINTSREMOVED:
            {
                const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRemoved * >( *changeIt ) )->getArray();
                std::cout << "INPUT REMOVE POINTS "<<tab << std::endl;
                removeInputPoints( tab );
                break;
            }
            case core::componentmodel::topology::POINTSRENUMBERING:
            {
                const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRenumbering * >( *changeIt ) )->getinv_IndexArray();
                std::cout << "INPUT RENUMBER POINTS "<<tab << std::endl;
                renumberInputPoints( tab );
                break;
            }
            case core::componentmodel::topology::EDGESADDED:
            {
                /// @TODO
                break;
            }
            case core::componentmodel::topology::EDGESREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const EdgesRemoved *>( *changeIt ) )->getArray();
                std::cout << "INPUT REMOVE EDGES "<<tab << std::endl;
                removeInputEdges( tab );
                break;
            }
            case core::componentmodel::topology::TRIANGLESADDED:
            {
                /// @TODO
                break;
            }
            case core::componentmodel::topology::TRIANGLESREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TrianglesRemoved *>( *changeIt ) )->getArray();
                std::cout << "INPUT REMOVE TRIANGLES "<<tab << std::endl;
                removeInputTriangles( tab );
                break;
            }
            case core::componentmodel::topology::QUADSADDED:
            {
                /// @TODO
                break;
            }
            case core::componentmodel::topology::QUADSREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const QuadsRemoved *>( *changeIt ) )->getArray();
                std::cout << "INPUT REMOVE QUADS "<<tab << std::endl;
                removeInputQuads( tab );
                break;
            }
            case core::componentmodel::topology::TETRAHEDRAADDED:
            {
                /// @TODO
                break;
            }
            case core::componentmodel::topology::TETRAHEDRAREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TetrahedraRemoved *>( *changeIt ) )->getArray();
                std::cout << "INPUT REMOVE TETRAS "<<tab << std::endl;
                removeInputTetras( tab );
                break;
            }
            case core::componentmodel::topology::HEXAHEDRAADDED:
            {
                /// @TODO
                break;
            }
            case core::componentmodel::topology::HEXAHEDRAREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const HexahedraRemoved *>( *changeIt ) )->getArray();
                std::cout << "INPUT REMOVE HEXAS "<<tab << std::endl;
                removeInputHexas( tab );
                break;
            }
            case core::componentmodel::topology::ENDING_EVENT:
            {
                pointsToRemove.erase(-1);
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

void Mesh2PointTopologicalMapping::swapInputPoints(int i1, int i2)
{
    if (pointsMappedFromPoint.empty()) return;
    vector<int> i1Map = pointsMappedFromPoint[i1];
    vector<int> i2Map = pointsMappedFromPoint[i2];

    pointsMappedFromPoint[i1] = i2Map;
    pointsMappedFromPoint[i2] = i1Map;
}

void Mesh2PointTopologicalMapping::removeInputPoints( const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFromPoint.empty()) return;
    unsigned int last = pointsMappedFromPoint.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapInputPoints( index[i], last );
        pointsToRemove.insert(pointsMappedFromPoint[last].begin(), pointsMappedFromPoint[last].end());
        --last;
    }

    pointsMappedFromPoint.resize( last + 1 );
}

void Mesh2PointTopologicalMapping::renumberInputPoints( const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFromPoint.empty()) return;
    helper::vector< vector<int> > copy = pointsMappedFromPoint;
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        vector<int> map = copy[index[i]];
        pointsMappedFromPoint[i] = map;
    }
}

void Mesh2PointTopologicalMapping::swapInputEdges(int i1, int i2)
{
    if (pointsMappedFromEdge.empty()) return;
    vector<int> i1Map = pointsMappedFromEdge[i1];
    vector<int> i2Map = pointsMappedFromEdge[i2];
    pointsMappedFromEdge[i1] = i2Map;
    pointsMappedFromEdge[i2] = i1Map;
}

void Mesh2PointTopologicalMapping::removeInputEdges( const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFromEdge.empty()) return;

    unsigned int last = pointsMappedFromEdge.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapInputEdges( index[i], last );
        pointsToRemove.insert(pointsMappedFromEdge[last].begin(), pointsMappedFromEdge[last].end());
        --last;
    }
    pointsMappedFromEdge.resize( last + 1 );
}

void Mesh2PointTopologicalMapping::swapInputTriangles(int i1, int i2)
{
    if (pointsMappedFromTriangle.empty()) return;
    vector<int> i1Map = pointsMappedFromTriangle[i1];
    vector<int> i2Map = pointsMappedFromTriangle[i2];
    pointsMappedFromTriangle[i1] = i2Map;
    pointsMappedFromTriangle[i2] = i1Map;
}

void Mesh2PointTopologicalMapping::removeInputTriangles( const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFromTriangle.empty()) return;

    unsigned int last = pointsMappedFromTriangle.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapInputTriangles( index[i], last );
        pointsToRemove.insert(pointsMappedFromTriangle[last].begin(), pointsMappedFromTriangle[last].end());
        --last;
    }
    pointsMappedFromTriangle.resize( last + 1 );
}

void Mesh2PointTopologicalMapping::swapInputQuads(int i1, int i2)
{
    if (pointsMappedFromQuad.empty()) return;
    vector<int> i1Map = pointsMappedFromQuad[i1];
    vector<int> i2Map = pointsMappedFromQuad[i2];
    pointsMappedFromQuad[i1] = i2Map;
    pointsMappedFromQuad[i2] = i1Map;
}

void Mesh2PointTopologicalMapping::removeInputQuads( const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFromQuad.empty()) return;

    unsigned int last = pointsMappedFromQuad.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapInputQuads( index[i], last );
        pointsToRemove.insert(pointsMappedFromQuad[last].begin(), pointsMappedFromQuad[last].end());
        --last;
    }
    pointsMappedFromQuad.resize( last + 1 );
}

void Mesh2PointTopologicalMapping::swapInputTetras(int i1, int i2)
{
    if (pointsMappedFromTetra.empty()) return;
    vector<int> i1Map = pointsMappedFromTetra[i1];
    vector<int> i2Map = pointsMappedFromTetra[i2];
    pointsMappedFromTetra[i1] = i2Map;
    pointsMappedFromTetra[i2] = i1Map;
}

void Mesh2PointTopologicalMapping::removeInputTetras( const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFromTetra.empty()) return;
    unsigned int last = pointsMappedFromTetra.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapInputTetras( index[i], last );
        pointsToRemove.insert(pointsMappedFromTetra[last].begin(), pointsMappedFromTetra[last].end());
        --last;
    }
    pointsMappedFromTetra.resize( last + 1 );
}

void Mesh2PointTopologicalMapping::swapInputHexas(int i1, int i2)
{
    if (pointsMappedFromHexa.empty()) return;
    vector<int> i1Map = pointsMappedFromHexa[i1];
    vector<int> i2Map = pointsMappedFromHexa[i2];
    pointsMappedFromHexa[i1] = i2Map;
    pointsMappedFromHexa[i2] = i1Map;
}

void Mesh2PointTopologicalMapping::removeInputHexas( const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFromHexa.empty()) return;

    unsigned int last = pointsMappedFromHexa.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapInputHexas( index[i], last );
        pointsToRemove.insert(pointsMappedFromHexa[last].begin(), pointsMappedFromHexa[last].end());
        --last;
    }
    pointsMappedFromHexa.resize( last + 1 );
}



void Mesh2PointTopologicalMapping::swapOutputPoints(int i1, int i2, bool removeLast)
{
    for (unsigned int i=0; i<pointsMappedFromPoint.size(); ++i)
    {
        vector<int> & pts = pointsMappedFromPoint[i];
        for (unsigned int j = 0; j < pts.size(); ++j)
        {
            if (pts[j] == i2)
                pts[j] = i1;
            else if (pts[j] == i1)
            {
                if (removeLast)
                    pts[j] = -1;
                else
                    pts[j] = i2;
            }
        }
    }
}

void Mesh2PointTopologicalMapping::removeOutputPoints( const sofa::helper::vector<unsigned int>& index )
{
    if (pointsMappedFromPoint.empty()) return;
    unsigned int last = nbOutputPoints - 1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapOutputPoints( index[i], last, true );
        --last;
    }

    nbOutputPoints = last + 1;
}

} // namespace topology
} // namespace component
} // namespace sofa

