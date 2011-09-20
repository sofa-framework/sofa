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

#include <sofa/component/mapping/SimpleTesselatedTetraMechanicalMapping.h>
#include <sofa/component/topology/PointData.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::SimpleTesselatedTetraMechanicalMapping(core::State<In>* from, core::State<Out>* to)
    : Inherit(from, to)
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
    this->Inherit::init();
    this->getContext()->get(topoMap);
    inputTopo = this->fromModel->getContext()->getMeshTopology();
    outputTopo = this->toModel->getContext()->getMeshTopology();
}

template <class TIn, class TOut>
void SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::apply ( const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, OutDataVecCoord& dOut, const InDataVecCoord& dIn)
{
    const InVecCoord& in = dIn.getValue();
    OutVecCoord& out = *dOut.beginEdit();

    if (!topoMap) return;
    const topology::PointData<sofa::helper::vector<int> >& pointMap = topoMap->getPointMappedFromPoint();
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

    dOut.endEdit();
}

template <class TIn, class TOut>
void SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::applyJ( const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    const InVecDeriv& in = dIn.getValue();
    OutVecDeriv& out = *dOut.beginEdit();

    if (!topoMap) return;
    const topology::PointData<sofa::helper::vector<int> >& pointMap = topoMap->getPointMappedFromPoint();
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
    dOut.endEdit();
}

template <class TIn, class TOut>
void SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::applyJT( const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    const OutVecDeriv& in = dIn.getValue();
    InVecDeriv& out = *dOut.beginEdit();

    if (!topoMap) return;
    const topology::PointData<sofa::helper::vector<int> >& pointMap = topoMap->getPointMappedFromPoint();
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
    dOut.endEdit();
}


template <class TIn, class TOut>
void SimpleTesselatedTetraMechanicalMapping<TIn, TOut>::applyJT( const core::ConstraintParams * /*cparams*/ /* PARAMS FIRST */, InDataMatrixDeriv& dOut, const OutDataMatrixDeriv& dIn)
{
    const OutMatrixDeriv& in = dIn.getValue();
    InMatrixDeriv& out = *dOut.beginEdit();

    if (!topoMap) return;

    const topology::PointData<sofa::helper::vector<int> >& pointSource = topoMap->getPointSource();
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
    dOut.endEdit();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
