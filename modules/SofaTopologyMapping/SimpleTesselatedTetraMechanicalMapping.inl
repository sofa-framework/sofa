/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_MAPPING_SIMPLETESSELATEDTETRAMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SIMPLETESSELATEDTETRAMAPPING_INL

#include <SofaTopologyMapping/SimpleTesselatedTetraMechanicalMapping.h>
#include <SofaBaseTopology/TopologyData.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::SimpleTesselatedTetraMechanicalMapping()
    : Inherit()
    , topoMap(NULL)
    , inputTopo(NULL)
    , outputTopo(NULL)
{
}

template <class TIn, class TOut>
SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::~SimpleTesselatedTetraMechanicalMapping()
{
}

template <class TIn, class TOut>
void SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::init()
{
    this->getContext()->get(topoMap);
    if (topoMap)
    {
        inputTopo = topoMap->getFrom();
        outputTopo = topoMap->getTo();
        if (outputTopo)
            this->toModel->resize(outputTopo->getNbPoints());
    }
    this->Inherit::init();
}

template <class TIn, class TOut>
void SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::apply ( const core::MechanicalParams* /* mparams */, OutDataVecCoord& dOut, const InDataVecCoord& dIn)
{
    if (!topoMap) return;
    const helper::vector<int>& pointMap = topoMap->getPointMappedFromPoint();
    const helper::vector<int>& edgeMap = topoMap->getPointMappedFromEdge();
    if (pointMap.empty() && edgeMap.empty()) return;
    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    helper::ReadAccessor<InDataVecCoord> in = dIn;
    helper::WriteAccessor<OutDataVecCoord> out = dOut;

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        if (pointMap[i] != -1)
            out[pointMap[i]] = in[i];
    }
    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        if (edgeMap[i] != -1)
            out[edgeMap[i]] = (in[ edges[i][0] ]+in[ edges[i][1] ])*0.5f;
    }
}

template <class TIn, class TOut>
void SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::applyJ( const core::MechanicalParams* /* mparams */, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{

    if (!topoMap) return;
    const helper::vector<int>& pointMap = topoMap->getPointMappedFromPoint();
    const helper::vector<int>& edgeMap = topoMap->getPointMappedFromEdge();
    if (pointMap.empty() && edgeMap.empty()) return;
    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    helper::ReadAccessor<InDataVecDeriv> in = dIn;
    helper::WriteAccessor<OutDataVecDeriv> out = dOut;

    out.resize(outputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        if (pointMap[i] != -1)
            out[pointMap[i]] = in[i];
    }
    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        if (edgeMap[i] != -1)
            out[edgeMap[i]] = (in[ edges[i][0] ]+in[ edges[i][1] ])*0.5f;
    }
}

template <class TIn, class TOut>
void SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::applyJT( const core::MechanicalParams* /* mparams */, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    if (!topoMap) return;
    const helper::vector<int>& pointMap = topoMap->getPointMappedFromPoint();
    const helper::vector<int>& edgeMap = topoMap->getPointMappedFromEdge();
    if (pointMap.empty() && edgeMap.empty()) return;
    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    helper::ReadAccessor<OutDataVecDeriv> in = dIn;
    helper::WriteAccessor<InDataVecDeriv> out = dOut;

    out.resize(inputTopo->getNbPoints());
    for(unsigned int i = 0; i < pointMap.size(); ++i)
    {
        if (pointMap[i] != -1)
            out[i] += in[pointMap[i]];
    }
    for(unsigned int i = 0; i < edgeMap.size(); ++i)
    {
        if (edgeMap[i] != -1)
        {
            out[edges[i][0]] += (in[edgeMap[i]])*0.5f;
            out[edges[i][1]] += (in[edgeMap[i]])*0.5f;
        }
    }
}


template <class TIn, class TOut>
void SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::applyJT( const core::ConstraintParams * /*cparams*/, InDataMatrixDeriv& dOut, const OutDataMatrixDeriv& dIn)
{

    if (!topoMap) return;

    const helper::vector<int>& pointSource = topoMap->getPointSource();
    if (pointSource.empty()) return;

    const core::topology::BaseMeshTopology::SeqEdges& edges = inputTopo->getEdges();

    const OutMatrixDeriv& in = dIn.getValue();
    InMatrixDeriv& out = *dOut.beginEdit();

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

                int source = pointSource[indexIn];
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
    dOut.endEdit();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
