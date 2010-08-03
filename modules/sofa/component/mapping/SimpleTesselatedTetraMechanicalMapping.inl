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
    const helper::vector<int>& edgeMap = topoMap->getPointMappedFromEdge();
    if (pointMap.getValue().empty() && edgeMap.empty()) return;
    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.getValue().size(); ++i)
    {
        if (pointMap.getValue()[i] != -1)
            out[pointMap.getValue()[i]] = in[i];
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
    const helper::vector<int>& edgeMap = topoMap->getPointMappedFromEdge();
    if (pointMap.getValue().empty() && edgeMap.empty()) return;
    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.getValue().size(); ++i)
    {
        if (pointMap.getValue()[i] != -1)
            out[pointMap.getValue()[i]] = in[i];
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
    const helper::vector<int>& edgeMap = topoMap->getPointMappedFromEdge();
    if (pointMap.getValue().empty() && edgeMap.empty()) return;
    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    out.resize(inputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.getValue().size(); ++i)
    {
        if (pointMap.getValue()[i] != -1)
            out[i] += in[pointMap.getValue()[i]];
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
void SimpleTesselatedTetraMechanicalMapping<BaseMapping>::applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    if (!topoMap) return;

    const topology::PointData<int>& pointSource = topoMap->getPointSource();
    if (pointSource.getValue().empty()) return;

    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                unsigned int indexIn = colIt.index();
                OutDeriv data = (OutDeriv) colIt.val();

                int source = pointSource.getValue()[indexIn];
                if (source > 0)
                {
                    o.addCol(source-1, data);
                }
                else if (source < 0)
                {
                    core::topology::BaseMeshTopology::Edge e = edges[-source-1];
                    InDeriv f =  data;
                    f*=0.5f;
                    o.addCol(e[0] , f);
                    o.addCol(e[1] , f);
                }
            }
        }
    }
}

/*
template <class BaseMapping>
void SimpleTesselatedTetraMechanicalMapping<BaseMapping>::applyJT( typename In::VecConst& out, const typename Out::VecConst& in )
{

    if (!topoMap) return;
    const topology::PointData<int>& pointSource = topoMap->getPointSource();
    if (pointSource.getValue().empty()) return;
    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    int offset = out.size();
    out.resize(offset+in.size());

    for(unsigned int i = 0; i < in.size(); ++i)
      {
        OutConstraintIterator itOut;
        std::pair< OutConstraintIterator, OutConstraintIterator > iter=in[i].data();

        for (itOut=iter.first;itOut!=iter.second;itOut++)
          {
            unsigned int indexIn = itOut->first;
            OutDeriv data = (OutDeriv) itOut->second;
	    int source = pointSource.getValue()[indexIn];
	    if (source > 0)
	    {
                out[i+offset].add(source-1 , data);
	    }
	    else if (source < 0)
	    {
		core::topology::BaseMeshTopology::Edge e = edges[-source-1];
		InDeriv f =  data;
		f*=0.5f;
                out[i+offset].add( e[0] , f );
                out[i+offset].add( e[1] , f );
	    }
	}
    }
}
*/

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
