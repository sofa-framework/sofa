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

    const topology::PointData< sofa::helper::vector<int> >& pointMap = topoMap->getPointsMappedFromPoint();
    const topology::EdgeData< sofa::helper::vector<int> >& edgeMap = topoMap->getPointsMappedFromEdge();
    const topology::TriangleData< sofa::helper::vector<int> >& triangleMap = topoMap->getPointsMappedFromTriangle();
    const topology::QuadData< sofa::helper::vector<int> >& quadMap = topoMap->getPointsMappedFromQuad();
    const topology::TetrahedronData< sofa::helper::vector<int> >& tetraMap = topoMap->getPointsMappedFromTetra();
    const topology::HexahedronData< sofa::helper::vector<int> >& hexaMap = topoMap->getPointsMappedFromHexa();

    if (pointMap.empty() && edgeMap.empty() && triangleMap.empty() && quadMap.empty() && tetraMap.empty() && hexaMap.empty()) return;

    const core::componentmodel::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();
    const core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = inputTopo->getTriangles();
    const core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = inputTopo->getQuads();
    const core::componentmodel::topology::BaseMeshTopology::SeqTetras& tetras = inputTopo->getTetras();
    const core::componentmodel::topology::BaseMeshTopology::SeqHexas& hexas = inputTopo->getHexas();

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        for(unsigned int j = 0; j < pointMap[i].size(); ++j)
        {
            out[pointMap[i][j]] = in[i]+topoMap->getPointBaryCoords()[j];
        }
    }

    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        for(unsigned int j = 0; j < edgeMap[i].size(); ++j)
        {
            double fx = topoMap->getEdgeBaryCoords()[j][0];
            out[edgeMap[i][j]] = in[ edges[i][0] ] * (1-fx)
                    +in[ edges[i][1] ] * fx;
        }
    }

    for(unsigned int i = 0; i < triangleMap.size(); ++i)
    {
        for(unsigned int j = 0; j < triangleMap[i].size(); ++j)
        {
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
            double fx = topoMap->getTetraBaryCoords()[j][0];
            double fy = topoMap->getTetraBaryCoords()[j][1];
            double fz = topoMap->getTetraBaryCoords()[j][2];
            out[tetraMap[i][j]] = in[ tetras[i][0] ] * (1-fx-fy-fz)
                    + in[ tetras[i][1] ] * fx
                    + in[ tetras[i][2] ] * fy
                    + in[ tetras[i][3] ] * fz;
        }
    }

    for(unsigned int i = 0; i <hexaMap.size(); ++i)
    {
        for(unsigned int j = 0; j < hexaMap[i].size(); ++j)
        {
            double fx = topoMap->getHexaBaryCoords()[j][0];
            double fy = topoMap->getHexaBaryCoords()[j][1];
            double fz = topoMap->getHexaBaryCoords()[j][2];
            out[hexaMap[i][j]] = in[ hexas[i][0] ] * ((1-fx) * (1-fy) * (1-fz))
                    + in[ hexas[i][1] ] * ((  fx) * (1-fy) * (1-fz))
                    + in[ hexas[i][2] ] * ((1-fx) * (  fy) * (1-fz))
                    + in[ hexas[i][3] ] * ((  fx) * (  fy) * (1-fz))
                    + in[ hexas[i][4] ] * ((1-fx) * (1-fy) * (  fz))
                    + in[ hexas[i][5] ] * ((  fx) * (1-fy) * (  fz))
                    + in[ hexas[i][6] ] * ((  fx) * (  fy) * (  fz))
                    + in[ hexas[i][7] ] * ((1-fx) * (  fy) * (  fz));
        }
    }
}

template <class BaseMapping>
void Mesh2PointMechanicalMapping<BaseMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    if (!topoMap) return;

    const topology::PointData< sofa::helper::vector<int> >& pointMap = topoMap->getPointsMappedFromPoint();
    const topology::EdgeData< sofa::helper::vector<int> >& edgeMap = topoMap->getPointsMappedFromEdge();
    const topology::TriangleData< sofa::helper::vector<int> >& triangleMap = topoMap->getPointsMappedFromTriangle();
    const topology::QuadData< sofa::helper::vector<int> >& quadMap = topoMap->getPointsMappedFromQuad();
    const topology::TetrahedronData< sofa::helper::vector<int> >& tetraMap = topoMap->getPointsMappedFromTetra();
    const topology::HexahedronData< sofa::helper::vector<int> >& hexaMap = topoMap->getPointsMappedFromHexa();

    if (pointMap.empty() && edgeMap.empty() && triangleMap.empty() && quadMap.empty() && tetraMap.empty() && hexaMap.empty()) return;

    const core::componentmodel::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();
    const core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = inputTopo->getTriangles();
    const core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = inputTopo->getQuads();
    const core::componentmodel::topology::BaseMeshTopology::SeqTetras& tetras = inputTopo->getTetras();
    const core::componentmodel::topology::BaseMeshTopology::SeqHexas& hexas = inputTopo->getHexas();

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        for(unsigned int j = 0; j < pointMap[i].size(); ++j)
        {
            out[pointMap[i][j]] = in[i]+topoMap->getPointBaryCoords()[j];
        }
    }

    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        for(unsigned int j = 0; j < edgeMap[i].size(); ++j)
        {
            double fx = topoMap->getEdgeBaryCoords()[j][0];
            out[edgeMap[i][j]] = in[ edges[i][0] ] * (1-fx)
                    +in[ edges[i][1] ] * fx;
        }
    }

    for(unsigned int i = 0; i < triangleMap.size(); ++i)
    {
        for(unsigned int j = 0; j < triangleMap[i].size(); ++j)
        {
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
            double fx = topoMap->getTetraBaryCoords()[j][0];
            double fy = topoMap->getTetraBaryCoords()[j][1];
            double fz = topoMap->getTetraBaryCoords()[j][2];
            out[tetraMap[i][j]] = in[ tetras[i][0] ] * (1-fx-fy-fz)
                    + in[ tetras[i][1] ] * fx
                    + in[ tetras[i][2] ] * fy
                    + in[ tetras[i][3] ] * fz;
        }
    }

    for(unsigned int i = 0; i <hexaMap.size(); ++i)
    {
        for(unsigned int j = 0; j < hexaMap[i].size(); ++j)
        {
            double fx = topoMap->getHexaBaryCoords()[j][0];
            double fy = topoMap->getHexaBaryCoords()[j][1];
            double fz = topoMap->getHexaBaryCoords()[j][2];
            out[hexaMap[i][j]] = in[ hexas[i][0] ] * ((1-fx) * (1-fy) * (1-fz))
                    + in[ hexas[i][1] ] * ((  fx) * (1-fy) * (1-fz))
                    + in[ hexas[i][2] ] * ((1-fx) * (  fy) * (1-fz))
                    + in[ hexas[i][3] ] * ((  fx) * (  fy) * (1-fz))
                    + in[ hexas[i][4] ] * ((1-fx) * (1-fy) * (  fz))
                    + in[ hexas[i][5] ] * ((  fx) * (1-fy) * (  fz))
                    + in[ hexas[i][6] ] * ((  fx) * (  fy) * (  fz))
                    + in[ hexas[i][7] ] * ((1-fx) * (  fy) * (  fz));
        }
    }
}

template <class BaseMapping>
void Mesh2PointMechanicalMapping<BaseMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    if (!topoMap) return;

    const topology::PointData< sofa::helper::vector<int> >& pointMap = topoMap->getPointsMappedFromPoint();
    const topology::EdgeData< sofa::helper::vector<int> >& edgeMap = topoMap->getPointsMappedFromEdge();
    const topology::TriangleData< sofa::helper::vector<int> >& triangleMap = topoMap->getPointsMappedFromTriangle();
    const topology::QuadData< sofa::helper::vector<int> >& quadMap = topoMap->getPointsMappedFromQuad();
    const topology::TetrahedronData< sofa::helper::vector<int> >& tetraMap = topoMap->getPointsMappedFromTetra();
    const topology::HexahedronData< sofa::helper::vector<int> >& hexaMap = topoMap->getPointsMappedFromHexa();

    if (pointMap.empty() && edgeMap.empty() && triangleMap.empty() && quadMap.empty() && tetraMap.empty() && hexaMap.empty()) return;

    const core::componentmodel::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();
    const core::componentmodel::topology::BaseMeshTopology::SeqTriangles& triangles = inputTopo->getTriangles();
    const core::componentmodel::topology::BaseMeshTopology::SeqQuads& quads = inputTopo->getQuads();
    const core::componentmodel::topology::BaseMeshTopology::SeqTetras& tetras = inputTopo->getTetras();
    const core::componentmodel::topology::BaseMeshTopology::SeqHexas& hexas = inputTopo->getHexas();

    out.resize(inputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        for(unsigned int j = 0; j < pointMap[i].size(); ++j)
        {
            out[i] += in[pointMap[i][j]];
        }
    }

    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        for(unsigned int j = 0; j < edgeMap[i].size(); ++j)
        {
            double fx = topoMap->getEdgeBaryCoords()[j][0];
            out[edges[i][0]] += in[ edgeMap[i][j] ] * (1-fx);
            out[edges[i][1]] += in[ edgeMap[i][j] ] * fx;
        }
    }

    for(unsigned int i = 0; i < triangleMap.size(); ++i)
    {
        for(unsigned int j = 0; j < triangleMap[i].size(); ++j)
        {
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
            double fx = topoMap->getTetraBaryCoords()[j][0];
            double fy = topoMap->getTetraBaryCoords()[j][1];
            double fz = topoMap->getTetraBaryCoords()[j][2];
            out[ tetras[i][0] ] += in[tetraMap[i][j]] * (1-fx-fy-fz);
            out[ tetras[i][1] ] += in[tetraMap[i][j]] * fx;
            out[ tetras[i][2] ] += in[tetraMap[i][j]] * fy;
            out[ tetras[i][3] ] += in[tetraMap[i][j]] * fz;
        }
    }

    for(unsigned int i = 0; i <hexaMap.size(); ++i)
    {
        for(unsigned int j = 0; j < hexaMap[i].size(); ++j)
        {
            double fx = topoMap->getHexaBaryCoords()[j][0];
            double fy = topoMap->getHexaBaryCoords()[j][1];
            double fz = topoMap->getHexaBaryCoords()[j][2];
            out[ hexas[i][0] ] += in[hexaMap[i][j]] * ((1-fx) * (1-fy) * (1-fz));
            out[ hexas[i][1] ] += in[hexaMap[i][j]] * ((  fx) * (1-fy) * (1-fz));
            out[ hexas[i][2] ] += in[hexaMap[i][j]] * ((1-fx) * (  fy) * (1-fz));
            out[ hexas[i][3] ] += in[hexaMap[i][j]] * ((  fx) * (  fy) * (1-fz));
            out[ hexas[i][4] ] += in[hexaMap[i][j]] * ((1-fx) * (1-fy) * (  fz));
            out[ hexas[i][5] ] += in[hexaMap[i][j]] * ((  fx) * (1-fy) * (  fz));
            out[ hexas[i][6] ] += in[hexaMap[i][j]] * ((  fx) * (  fy) * (  fz));
            out[ hexas[i][7] ] += in[hexaMap[i][j]] * ((1-fx) * (  fy) * (  fz));
        }
    }
}

template <class BaseMapping>
void Mesh2PointMechanicalMapping<BaseMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{
    /*
        if (!topoMap) return;
        const topology::PointData<int>& pointSource = topoMap->getPointSource();
        if (pointSource.empty()) return;
        const core::componentmodel::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

        int offset = out.size();
        out.resize(offset+in.size());

        for(unsigned int c = 0; c < in.size(); ++c)
        {
    	for(unsigned int j=0;j<in[c].size();j++)
            {
                const typename Out::SparseDeriv cIn = in[c][j];
    	    int source = pointSource[cIn.index];
    	    if (source > 0)
    	    {
    		out[c+offset].push_back(typename In::SparseDeriv( source-1 , (typename In::Deriv) cIn.data ));
    	    }
    	    else if (source < 0)
    	    {
    		core::componentmodel::topology::BaseMeshTopology::Edge e = edges[-source-1];
    		typename In::Deriv f = (typename In::Deriv) cIn.data;
    		f*=0.5f;
    		out[c+offset].push_back(typename In::SparseDeriv( e[0] , f ));
    		out[c+offset].push_back(typename In::SparseDeriv( e[1] , f ));
    	    }
    	}
        }*/
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
