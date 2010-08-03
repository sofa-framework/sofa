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
#ifndef SOFA_COMPONENT_MAPPING_MESH2POINTMAPPING_INL
#define SOFA_COMPONENT_MAPPING_MESH2POINTMAPPING_INL

#include "Mesh2PointMechanicalMapping.h"

namespace sofa
{

namespace component
{

namespace mapping
{


template <class BaseMapping>
Mesh2PointMechanicalMapping<BaseMapping>::Mesh2PointMechanicalMapping(In* from, Out* to)
    : Inherit(from, to)
    , topoMap(NULL)
    , inputTopo(NULL)
    , outputTopo(NULL)
{
}

template <class BaseMapping>
Mesh2PointMechanicalMapping<BaseMapping>::~Mesh2PointMechanicalMapping()
{
}

template <class BaseMapping>
void Mesh2PointMechanicalMapping<BaseMapping>::init()
{
    this->Inherit::init();
    this->getContext()->get(topoMap);
    inputTopo = this->fromModel->getContext()->getMeshTopology();
    outputTopo = this->toModel->getContext()->getMeshTopology();
}

template <class BaseMapping>
void Mesh2PointMechanicalMapping<BaseMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    if (!topoMap) return;

    const sofa::helper::vector< sofa::helper::vector<int> >& pointMap = topoMap->getPointsMappedFromPoint();
    const sofa::helper::vector< sofa::helper::vector<int> >& edgeMap = topoMap->getPointsMappedFromEdge();
    const sofa::helper::vector< sofa::helper::vector<int> >& triangleMap = topoMap->getPointsMappedFromTriangle();
    const sofa::helper::vector< sofa::helper::vector<int> >& quadMap = topoMap->getPointsMappedFromQuad();
    const sofa::helper::vector< sofa::helper::vector<int> >& tetraMap = topoMap->getPointsMappedFromTetra();
    const sofa::helper::vector< sofa::helper::vector<int> >& hexaMap = topoMap->getPointsMappedFromHexa();

    if (pointMap.empty() && edgeMap.empty() && triangleMap.empty() && quadMap.empty() && tetraMap.empty() && hexaMap.empty()) return;

    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();
    const core::topology::BaseMeshTopology::SeqTriangles& triangles = inputTopo->getTriangles();
    const core::topology::BaseMeshTopology::SeqQuads& quads = inputTopo->getQuads();
    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = inputTopo->getTetrahedra();
    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = inputTopo->getHexahedra();

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        for(unsigned int j = 0; j < pointMap[i].size(); ++j)
        {
            if (pointMap[i][j] == -1) continue;
            out[pointMap[i][j]] = in[i]+topoMap->getPointBaryCoords()[j];
        }
    }

    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        for(unsigned int j = 0; j < edgeMap[i].size(); ++j)
        {
            if (edgeMap[i][j] == -1) continue;
            double fx = topoMap->getEdgeBaryCoords()[j][0];
            out[edgeMap[i][j]] = in[ edges[i][0] ] * (1-fx)
                    +in[ edges[i][1] ] * fx;
        }
    }

    for(unsigned int i = 0; i < triangleMap.size(); ++i)
    {
        for(unsigned int j = 0; j < triangleMap[i].size(); ++j)
        {
            if (triangleMap[i][j] == -1) continue;
            double fx = topoMap->getTriangleBaryCoords()[j][0];
            double fy = topoMap->getTriangleBaryCoords()[j][1];
            out[triangleMap[i][j]] = in[ triangles[i][0] ] * (1-fx-fy)
                    + in[ triangles[i][1] ] * fx
                    + in[ triangles[i][2] ] * fy;
        }
    }

    for(unsigned int i = 0; i < quadMap.size(); ++i)
    {
        for(unsigned int j = 0; j < quadMap[i].size(); ++j)
        {
            if (quadMap[i][j] == -1) continue;
            double fx = topoMap->getQuadBaryCoords()[j][0];
            double fy = topoMap->getQuadBaryCoords()[j][1];
            out[quadMap[i][j]] = in[ quads[i][0] ] * ((1-fx) * (1-fy))
                    + in[ quads[i][1] ] * ((  fx) * (1-fy))
                    + in[ quads[i][2] ] * ((1-fx) * (  fy))
                    + in[ quads[i][3] ] * ((  fx) * (  fy));
        }
    }

    for(unsigned int i = 0; i < tetraMap.size(); ++i)
    {
        for(unsigned int j = 0; j < tetraMap[i].size(); ++j)
        {
            if (tetraMap[i][j] == -1) continue;
            double fx = topoMap->getTetraBaryCoords()[j][0];
            double fy = topoMap->getTetraBaryCoords()[j][1];
            double fz = topoMap->getTetraBaryCoords()[j][2];
            out[tetraMap[i][j]] = in[ tetrahedra[i][0] ] * (1-fx-fy-fz)
                    + in[ tetrahedra[i][1] ] * fx
                    + in[ tetrahedra[i][2] ] * fy
                    + in[ tetrahedra[i][3] ] * fz;
        }
    }

    for(unsigned int i = 0; i <hexaMap.size(); ++i)
    {
        for(unsigned int j = 0; j < hexaMap[i].size(); ++j)
        {
            if (hexaMap[i][j] == -1) continue;
            double fx = topoMap->getHexaBaryCoords()[j][0];
            double fy = topoMap->getHexaBaryCoords()[j][1];
            double fz = topoMap->getHexaBaryCoords()[j][2];
            out[hexaMap[i][j]] = in[ hexahedra[i][0] ] * ((1-fx) * (1-fy) * (1-fz))
                    + in[ hexahedra[i][1] ] * ((  fx) * (1-fy) * (1-fz))
                    + in[ hexahedra[i][2] ] * ((1-fx) * (  fy) * (1-fz))
                    + in[ hexahedra[i][3] ] * ((  fx) * (  fy) * (1-fz))
                    + in[ hexahedra[i][4] ] * ((1-fx) * (1-fy) * (  fz))
                    + in[ hexahedra[i][5] ] * ((  fx) * (1-fy) * (  fz))
                    + in[ hexahedra[i][6] ] * ((  fx) * (  fy) * (  fz))
                    + in[ hexahedra[i][7] ] * ((1-fx) * (  fy) * (  fz));
        }
    }
}

template <class BaseMapping>
void Mesh2PointMechanicalMapping<BaseMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    if (!topoMap) return;

    const sofa::helper::vector< sofa::helper::vector<int> >& pointMap = topoMap->getPointsMappedFromPoint();
    const sofa::helper::vector< sofa::helper::vector<int> >& edgeMap = topoMap->getPointsMappedFromEdge();
    const sofa::helper::vector< sofa::helper::vector<int> >& triangleMap = topoMap->getPointsMappedFromTriangle();
    const sofa::helper::vector< sofa::helper::vector<int> >& quadMap = topoMap->getPointsMappedFromQuad();
    const sofa::helper::vector< sofa::helper::vector<int> >& tetraMap = topoMap->getPointsMappedFromTetra();
    const sofa::helper::vector< sofa::helper::vector<int> >& hexaMap = topoMap->getPointsMappedFromHexa();

    if (pointMap.empty() && edgeMap.empty() && triangleMap.empty() && quadMap.empty() && tetraMap.empty() && hexaMap.empty()) return;

    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();
    const core::topology::BaseMeshTopology::SeqTriangles& triangles = inputTopo->getTriangles();
    const core::topology::BaseMeshTopology::SeqQuads& quads = inputTopo->getQuads();
    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = inputTopo->getTetrahedra();
    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = inputTopo->getHexahedra();

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        for(unsigned int j = 0; j < pointMap[i].size(); ++j)
        {
            if (pointMap[i][j] == -1) continue;
            out[pointMap[i][j]] = in[i]+topoMap->getPointBaryCoords()[j];
        }
    }

    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        for(unsigned int j = 0; j < edgeMap[i].size(); ++j)
        {
            if (edgeMap[i][j] == -1) continue;
            double fx = topoMap->getEdgeBaryCoords()[j][0];
            out[edgeMap[i][j]] = in[ edges[i][0] ] * (1-fx)
                    +in[ edges[i][1] ] * fx;
        }
    }

    for(unsigned int i = 0; i < triangleMap.size(); ++i)
    {
        for(unsigned int j = 0; j < triangleMap[i].size(); ++j)
        {
            if (triangleMap[i][j] == -1) continue;
            double fx = topoMap->getTriangleBaryCoords()[j][0];
            double fy = topoMap->getTriangleBaryCoords()[j][1];
            out[triangleMap[i][j]] = in[ triangles[i][0] ] * (1-fx-fy)
                    + in[ triangles[i][1] ] * fx
                    + in[ triangles[i][2] ] * fy;
        }
    }

    for(unsigned int i = 0; i < quadMap.size(); ++i)
    {
        for(unsigned int j = 0; j < quadMap[i].size(); ++j)
        {
            if (quadMap[i][j] == -1) continue;
            double fx = topoMap->getQuadBaryCoords()[j][0];
            double fy = topoMap->getQuadBaryCoords()[j][1];
            out[quadMap[i][j]] = in[ quads[i][0] ] * ((1-fx) * (1-fy))
                    + in[ quads[i][1] ] * ((  fx) * (1-fy))
                    + in[ quads[i][2] ] * ((1-fx) * (  fy))
                    + in[ quads[i][3] ] * ((  fx) * (  fy));
        }
    }

    for(unsigned int i = 0; i < tetraMap.size(); ++i)
    {
        for(unsigned int j = 0; j < tetraMap[i].size(); ++j)
        {
            if (tetraMap[i][j] == -1) continue;
            double fx = topoMap->getTetraBaryCoords()[j][0];
            double fy = topoMap->getTetraBaryCoords()[j][1];
            double fz = topoMap->getTetraBaryCoords()[j][2];
            out[tetraMap[i][j]] = in[ tetrahedra[i][0] ] * (1-fx-fy-fz)
                    + in[ tetrahedra[i][1] ] * fx
                    + in[ tetrahedra[i][2] ] * fy
                    + in[ tetrahedra[i][3] ] * fz;
        }
    }

    for(unsigned int i = 0; i <hexaMap.size(); ++i)
    {
        for(unsigned int j = 0; j < hexaMap[i].size(); ++j)
        {
            if (hexaMap[i][j] == -1) continue;
            double fx = topoMap->getHexaBaryCoords()[j][0];
            double fy = topoMap->getHexaBaryCoords()[j][1];
            double fz = topoMap->getHexaBaryCoords()[j][2];
            out[hexaMap[i][j]] = in[ hexahedra[i][0] ] * ((1-fx) * (1-fy) * (1-fz))
                    + in[ hexahedra[i][1] ] * ((  fx) * (1-fy) * (1-fz))
                    + in[ hexahedra[i][2] ] * ((1-fx) * (  fy) * (1-fz))
                    + in[ hexahedra[i][3] ] * ((  fx) * (  fy) * (1-fz))
                    + in[ hexahedra[i][4] ] * ((1-fx) * (1-fy) * (  fz))
                    + in[ hexahedra[i][5] ] * ((  fx) * (1-fy) * (  fz))
                    + in[ hexahedra[i][6] ] * ((  fx) * (  fy) * (  fz))
                    + in[ hexahedra[i][7] ] * ((1-fx) * (  fy) * (  fz));
        }
    }
}

template <class BaseMapping>
void Mesh2PointMechanicalMapping<BaseMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    if (!topoMap) return;

    const sofa::helper::vector< sofa::helper::vector<int> >& pointMap = topoMap->getPointsMappedFromPoint();
    const sofa::helper::vector< sofa::helper::vector<int> >& edgeMap = topoMap->getPointsMappedFromEdge();
    const sofa::helper::vector< sofa::helper::vector<int> >& triangleMap = topoMap->getPointsMappedFromTriangle();
    const sofa::helper::vector< sofa::helper::vector<int> >& quadMap = topoMap->getPointsMappedFromQuad();
    const sofa::helper::vector< sofa::helper::vector<int> >& tetraMap = topoMap->getPointsMappedFromTetra();
    const sofa::helper::vector< sofa::helper::vector<int> >& hexaMap = topoMap->getPointsMappedFromHexa();

    if (pointMap.empty() && edgeMap.empty() && triangleMap.empty() && quadMap.empty() && tetraMap.empty() && hexaMap.empty()) return;

    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();
    const core::topology::BaseMeshTopology::SeqTriangles& triangles = inputTopo->getTriangles();
    const core::topology::BaseMeshTopology::SeqQuads& quads = inputTopo->getQuads();
    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = inputTopo->getTetrahedra();
    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = inputTopo->getHexahedra();

    out.resize(inputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        for(unsigned int j = 0; j < pointMap[i].size(); ++j)
        {
            if (pointMap[i][j] == -1) continue;
            out[i] += in[pointMap[i][j]];
        }
    }

    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        for(unsigned int j = 0; j < edgeMap[i].size(); ++j)
        {
            if (edgeMap[i][j] == -1) continue;
            double fx = topoMap->getEdgeBaryCoords()[j][0];
            out[edges[i][0]] += in[ edgeMap[i][j] ] * (1-fx);
            out[edges[i][1]] += in[ edgeMap[i][j] ] * fx;
        }
    }

    for(unsigned int i = 0; i < triangleMap.size(); ++i)
    {
        for(unsigned int j = 0; j < triangleMap[i].size(); ++j)
        {
            if (triangleMap[i][j] == -1) continue;
            double fx = topoMap->getTriangleBaryCoords()[j][0];
            double fy = topoMap->getTriangleBaryCoords()[j][1];
            out[ triangles[i][0] ] += in[triangleMap[i][j]] * (1-fx-fy);
            out[ triangles[i][1] ] += in[triangleMap[i][j]] * fx;
            out[ triangles[i][2] ] += in[triangleMap[i][j]] * fy;
        }
    }

    for(unsigned int i = 0; i < quadMap.size(); ++i)
    {
        for(unsigned int j = 0; j < quadMap[i].size(); ++j)
        {
            if (quadMap[i][j] == -1) continue;
            double fx = topoMap->getQuadBaryCoords()[j][0];
            double fy = topoMap->getQuadBaryCoords()[j][1];
            out[ quads[i][0] ] += in[quadMap[i][j]] * ((1-fx) * (1-fy));
            out[ quads[i][1] ] += in[quadMap[i][j]] * ((  fx) * (1-fy));
            out[ quads[i][2] ] += in[quadMap[i][j]] * ((1-fx) * (  fy));
            out[ quads[i][3] ] += in[quadMap[i][j]] * ((  fx) * (  fy));
        }
    }

    for(unsigned int i = 0; i < tetraMap.size(); ++i)
    {
        for(unsigned int j = 0; j < tetraMap[i].size(); ++j)
        {
            if (tetraMap[i][j] == -1) continue;
            double fx = topoMap->getTetraBaryCoords()[j][0];
            double fy = topoMap->getTetraBaryCoords()[j][1];
            double fz = topoMap->getTetraBaryCoords()[j][2];
            out[ tetrahedra[i][0] ] += in[tetraMap[i][j]] * (1-fx-fy-fz);
            out[ tetrahedra[i][1] ] += in[tetraMap[i][j]] * fx;
            out[ tetrahedra[i][2] ] += in[tetraMap[i][j]] * fy;
            out[ tetrahedra[i][3] ] += in[tetraMap[i][j]] * fz;
        }
    }

    for(unsigned int i = 0; i <hexaMap.size(); ++i)
    {
        for(unsigned int j = 0; j < hexaMap[i].size(); ++j)
        {
            if (hexaMap[i][j] == -1) continue;
            double fx = topoMap->getHexaBaryCoords()[j][0];
            double fy = topoMap->getHexaBaryCoords()[j][1];
            double fz = topoMap->getHexaBaryCoords()[j][2];
            out[ hexahedra[i][0] ] += in[hexaMap[i][j]] * ((1-fx) * (1-fy) * (1-fz));
            out[ hexahedra[i][1] ] += in[hexaMap[i][j]] * ((  fx) * (1-fy) * (1-fz));
            out[ hexahedra[i][2] ] += in[hexaMap[i][j]] * ((1-fx) * (  fy) * (1-fz));
            out[ hexahedra[i][3] ] += in[hexaMap[i][j]] * ((  fx) * (  fy) * (1-fz));
            out[ hexahedra[i][4] ] += in[hexaMap[i][j]] * ((1-fx) * (1-fy) * (  fz));
            out[ hexahedra[i][5] ] += in[hexaMap[i][j]] * ((  fx) * (1-fy) * (  fz));
            out[ hexahedra[i][6] ] += in[hexaMap[i][j]] * ((  fx) * (  fy) * (  fz));
            out[ hexahedra[i][7] ] += in[hexaMap[i][j]] * ((1-fx) * (  fy) * (  fz));
        }
    }
}


template <class BaseMapping>
void Mesh2PointMechanicalMapping<BaseMapping>::applyJT(typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in)
{
    using topology::Mesh2PointTopologicalMapping;

    if (!topoMap)
        return;

    const sofa::helper::vector< std::pair< Mesh2PointTopologicalMapping::Element, int> >& pointSource = topoMap->getPointSource();

    if (pointSource.empty())
        return;

    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();
    const core::topology::BaseMeshTopology::SeqTriangles& triangles = inputTopo->getTriangles();
    const core::topology::BaseMeshTopology::SeqQuads& quads = inputTopo->getQuads();
    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = inputTopo->getTetrahedra();
    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = inputTopo->getHexahedra();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            while (colIt != colItEnd)
            {
                const unsigned int indexIn = colIt.index();
                const OutDeriv data = colIt.val();
                std::pair< Mesh2PointTopologicalMapping::Element, int> source = pointSource[indexIn];

                switch (source.first)
                {
                case topology::Mesh2PointTopologicalMapping::POINT:
                {
                    o.addCol(source.second, data);

                    break;
                }
                case topology::Mesh2PointTopologicalMapping::EDGE:
                {
                    core::topology::BaseMeshTopology::Edge e = edges[source.second];
                    typename In::Deriv f = data;
                    double fx = topoMap->getEdgeBaryCoords()[indexIn][0];

                    o.addCol(e[0], f * (1 - fx));
                    o.addCol(e[1], f * fx);

                    break;
                }
                case topology::Mesh2PointTopologicalMapping::TRIANGLE:
                {
                    core::topology::BaseMeshTopology::Triangle t = triangles[source.second];
                    typename In::Deriv f = data;
                    double fx = topoMap->getTriangleBaryCoords()[indexIn][0];
                    double fy = topoMap->getTriangleBaryCoords()[indexIn][1];

                    o.addCol(t[0], f * (1 - fx - fy));
                    o.addCol(t[1], f * fx);
                    o.addCol(t[2], f * fy);

                    break;
                }
                case topology::Mesh2PointTopologicalMapping::QUAD:
                {
                    core::topology::BaseMeshTopology::Quad q = quads[source.second];
                    typename In::Deriv f = data;
                    double fx = topoMap->getQuadBaryCoords()[indexIn][0];
                    double fy = topoMap->getQuadBaryCoords()[indexIn][1];

                    o.addCol(q[0], f * ((1-fx) * (1-fy)));
                    o.addCol(q[1], f * (fx * (1-fy)));
                    o.addCol(q[0], f * ((1-fx) * fy));
                    o.addCol(q[1], f * (fx * fy));

                    break;
                }
                case topology::Mesh2PointTopologicalMapping::TETRA:
                {
                    core::topology::BaseMeshTopology::Tetra t = tetrahedra[source.second];
                    typename In::Deriv f = data;
                    double fx = topoMap->getTetraBaryCoords()[indexIn][0];
                    double fy = topoMap->getTetraBaryCoords()[indexIn][1];
                    double fz = topoMap->getTetraBaryCoords()[indexIn][2];

                    o.addCol(t[0], f * (1-fx-fy-fz));
                    o.addCol(t[1], f * fx);
                    o.addCol(t[0], f * fy);
                    o.addCol(t[1], f * fz);

                    break;
                }
                case topology::Mesh2PointTopologicalMapping::HEXA:
                {
                    core::topology::BaseMeshTopology::Hexa h = hexahedra[source.second];
                    typename In::Deriv f = data;
                    const double fx = topoMap->getHexaBaryCoords()[indexIn][0];
                    const double fy = topoMap->getHexaBaryCoords()[indexIn][1];
                    const double fz = topoMap->getHexaBaryCoords()[indexIn][2];
                    const double oneMinFx = 1 - fx;
                    const double oneMinFy = 1 - fy;
                    const double oneMinFz = 1 - fz;

                    o.addCol(h[0] , f * oneMinFx * oneMinFy * oneMinFz);
                    o.addCol(h[1] , f * fx * oneMinFy * oneMinFz);
                    o.addCol(h[0] , f * oneMinFx * fy * oneMinFz);
                    o.addCol(h[1] , f * fx * fy * oneMinFz);
                    o.addCol(h[0] , f * oneMinFx * oneMinFy * fz);
                    o.addCol(h[1] , f * fx * oneMinFy * fz);
                    o.addCol(h[0] , f * fx * fy * fz);
                    o.addCol(h[1] , f * oneMinFx * fy * fz);

                    break;
                }
                default:

                    break;
                }

                ++colIt;
            }
        }
    }
}


//template <class BaseMapping>
//void Mesh2PointMechanicalMapping<BaseMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
//{
//
//    if (!topoMap) return;
//    const sofa::helper::vector< std::pair<topology::Mesh2PointTopologicalMapping::Element,int> >& pointSource = topoMap->getPointSource();
//    if (pointSource.empty()) return;
//    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();
//    const core::topology::BaseMeshTopology::SeqTriangles& triangles = inputTopo->getTriangles();
//    const core::topology::BaseMeshTopology::SeqQuads& quads = inputTopo->getQuads();
//    const core::topology::BaseMeshTopology::SeqTetrahedra& tetrahedra = inputTopo->getTetrahedra();
//    const core::topology::BaseMeshTopology::SeqHexahedra& hexahedra = inputTopo->getHexahedra();
//
//    int offset = out.size();
//    out.resize(offset+in.size());
//
//    for(unsigned int i = 0; i < in.size(); ++i)
//    {
//      OutConstraintIterator itOut;
//      std::pair< OutConstraintIterator, OutConstraintIterator > iter=in[i].data();
//
//      for (itOut=iter.first;itOut!=iter.second;itOut++)
//        {
//            unsigned int indexIn = itOut->first;
//            OutDeriv data = (OutDeriv) itOut->second;
//	    std::pair<topology::Mesh2PointTopologicalMapping::Element,int> source = pointSource[indexIn];
//	    switch (source.first) {
//	    case topology::Mesh2PointTopologicalMapping::POINT:
//	    {
//		out[i+offset].add( source.second , data );
//		break;
//	    }
//	    case topology::Mesh2PointTopologicalMapping::EDGE:
//	    {
//		core::topology::BaseMeshTopology::Edge e = edges[source.second];
//		typename In::Deriv f = data;
//		double fx = topoMap->getEdgeBaryCoords()[indexIn][0];
//                out[i+offset].add( e[0] , f*(1-fx) );
//                out[i+offset].add( e[1] , f*fx );
//		break;
//	    }
//	    case topology::Mesh2PointTopologicalMapping::TRIANGLE:
//	    {
//		core::topology::BaseMeshTopology::Triangle t = triangles[source.second];
//		typename In::Deriv f = data;
//		double fx = topoMap->getTriangleBaryCoords()[indexIn][0];
//		double fy = topoMap->getTriangleBaryCoords()[indexIn][1];
//                out[i+offset].add( t[0] , f*(1-fx-fy) );
//                out[i+offset].add( t[1] , f*fx );
//                out[i+offset].add( t[2] , f*fy );
//		break;
//	    }
//	    case topology::Mesh2PointTopologicalMapping::QUAD:
//	    {
//		core::topology::BaseMeshTopology::Quad q = quads[source.second];
//		typename In::Deriv f = data;
//		double fx = topoMap->getQuadBaryCoords()[indexIn][0];
//		double fy = topoMap->getQuadBaryCoords()[indexIn][1];
//                out[i+offset].add( q[0] , f*((1-fx) * (1-fy)) );
//                out[i+offset].add( q[1] , f*((  fx) * (1-fy)) );
//                out[i+offset].add( q[0] , f*((1-fx) * (  fy)) );
//                out[i+offset].add( q[1] , f*((  fx) * (  fy)) );
//		break;
//	    }
//	    case topology::Mesh2PointTopologicalMapping::TETRA:
//	    {
//		core::topology::BaseMeshTopology::Tetra t = tetrahedra[source.second];
//		typename In::Deriv f = data;
//		double fx = topoMap->getTetraBaryCoords()[indexIn][0];
//		double fy = topoMap->getTetraBaryCoords()[indexIn][1];
//		double fz = topoMap->getTetraBaryCoords()[indexIn][2];
//                out[i+offset].add( t[0] , f*(1-fx-fy-fz) );
//                out[i+offset].add( t[1] , f*fx );
//                out[i+offset].add( t[0] , f*fy );
//                out[i+offset].add( t[1] , f*fz );
//		break;
//	    }
//	    case topology::Mesh2PointTopologicalMapping::HEXA:
//	    {
//		core::topology::BaseMeshTopology::Hexa h = hexahedra[source.second];
//		typename In::Deriv f = data;
//		double fx = topoMap->getHexaBaryCoords()[indexIn][0];
//		double fy = topoMap->getHexaBaryCoords()[indexIn][1];
//		double fz = topoMap->getHexaBaryCoords()[indexIn][2];
//                out[i+offset].add( h[0] , f*((1-fx) * (1-fy) * (1-fz)) );
//                out[i+offset].add( h[1] , f*((  fx) * (1-fy) * (1-fz)) );
//                out[i+offset].add( h[0] , f*((1-fx) * (  fy) * (1-fz)) );
//                out[i+offset].add( h[1] , f*((  fx) * (  fy) * (1-fz)) );
//                out[i+offset].add( h[0] , f*((1-fx) * (1-fy) * (  fz)) );
//                out[i+offset].add( h[1] , f*((  fx) * (1-fy) * (  fz)) );
//                out[i+offset].add( h[0] , f*((  fx) * (  fy) * (  fz)) );
//                out[i+offset].add( h[1] , f*((1-fx) * (  fy) * (  fz)) );
//		break;
//	    }
//	    default:
//		break;
//	    }
//	}
//    }
//}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
