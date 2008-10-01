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
#ifndef SOFA_COMPONENT_MAPPING_SIMPLETESSELATEDTETRAMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SIMPLETESSELATEDTETRAMAPPING_INL

#include "SimpleTesselatedTetraMechanicalMapping.h"

namespace sofa
{

namespace component
{

namespace mapping
{


template <class BaseMapping>
SimpleTesselatedTetraMechanicalMapping<BaseMapping>::SimpleTesselatedTetraMechanicalMapping(In* from, Out* to)
    : Inherit(from, to)
    , topoMap(NULL)
    , inputTopo(NULL)
    , outputTopo(NULL)
{
}

template <class BaseMapping>
SimpleTesselatedTetraMechanicalMapping<BaseMapping>::~SimpleTesselatedTetraMechanicalMapping()
{
}

template <class BaseMapping>
void SimpleTesselatedTetraMechanicalMapping<BaseMapping>::init()
{
    this->Inherit::init();
    this->getContext()->get(topoMap);
    inputTopo = this->fromModel->getContext()->getMeshTopology();
    outputTopo = this->toModel->getContext()->getMeshTopology();
}

template <class BaseMapping>
void SimpleTesselatedTetraMechanicalMapping<BaseMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    if (!topoMap) return;
    const topology::PointData<int>& pointMap = topoMap->getPointMappedFromPoint();
    const topology::EdgeData<int>& edgeMap = topoMap->getPointMappedFromEdge();
    if (pointMap.empty() && edgeMap.empty()) return;
    const core::componentmodel::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        if (pointMap[i] != -1)
            out[pointMap[i]] = in[i];
    }
    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        if (edgeMap[i] != -1)
            out[edgeMap[i]] = (in[ edges[i][0] ]+in[ edges[i][1] ])/2;
    }
}

template <class BaseMapping>
void SimpleTesselatedTetraMechanicalMapping<BaseMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    if (!topoMap) return;
    const topology::PointData<int>& pointMap = topoMap->getPointMappedFromPoint();
    const topology::EdgeData<int>& edgeMap = topoMap->getPointMappedFromEdge();
    if (pointMap.empty() && edgeMap.empty()) return;
    const core::componentmodel::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        if (pointMap[i] != -1)
            out[pointMap[i]] = in[i];
    }
    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        if (edgeMap[i] != -1)
            out[edgeMap[i]] = (in[ edges[i][0] ]+in[ edges[i][1] ])/2;
    }
}

template <class BaseMapping>
void SimpleTesselatedTetraMechanicalMapping<BaseMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    if (!topoMap) return;
    const topology::PointData<int>& pointMap = topoMap->getPointMappedFromPoint();
    const topology::EdgeData<int>& edgeMap = topoMap->getPointMappedFromEdge();
    if (pointMap.empty() && edgeMap.empty()) return;
    const core::componentmodel::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        if (pointMap[i] != -1)
            out[i] += in[pointMap[i]];
    }
    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        if (edgeMap[i] != -1)
        {
            out[edges[i][0]] += (in[edgeMap[i]])/2;
            out[edges[i][1]] += (in[edgeMap[i]])/2;
        }
    }
}

template <class BaseMapping>
void SimpleTesselatedTetraMechanicalMapping<BaseMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{

    if (!topoMap) return;
    const topology::PointData<int>& pointSource = topoMap->getPointSource();
    if (pointSource.empty()) return;
    const core::componentmodel::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    int offset = out.size();
    out.resize(offset+in.size());

    for(unsigned int c = 0; c < in.size(); ++c)
    {
        for(unsigned int j=0; j<in[c].size(); j++)
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
    }
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
