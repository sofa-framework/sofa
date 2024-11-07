/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/mapping/linear/CellAveragingMapping.h>
#include <sofa/core/MappingHelper.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraintEigenUtils.h>


namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
CellAveragingMapping<TIn, TOut>::CellAveragingMapping()
    : l_topology(initLink("topology", "link to the topology container"))
{
    m_Js.resize( 1, &m_J );
}

template <class TIn, class TOut>
void CellAveragingMapping<TIn, TOut>::init()
{
    Inherit1::init();

    if (l_topology.empty())
    {
        msg_warning() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (!l_topology)
    {
        msg_error() << "No topology found";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    buildJMatrix();
    this->getToModel()->resize( l_topology->getNbPoints() );

    Inherit1::init();

    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Invalid)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class TIn, class TOut>
void CellAveragingMapping<TIn, TOut>::apply(
    const core::MechanicalParams* mparams, DataVecCoord_t<Out>& out,
    const DataVecCoord_t<In>& in)
{
    SOFA_UNUSED(mparams);

    if (l_topology)
    {
        helper::WriteOnlyAccessor< Data<VecCoord_t<Out>> > pointValues = out;
        helper::ReadAccessor< Data<VecCoord_t<In> > > cellValues = in;

        const auto nbPoints = l_topology->getNbPoints();
        pointValues.resize(nbPoints);

        const auto& triangles = l_topology->getTriangles();
        if (triangles.size() != cellValues.size())
        {
            msg_error() << "Triangle count does not match input size.";
        }
        else
        {
            for (std::size_t v_i = 0; v_i < nbPoints; ++v_i)
            {
                const auto trianglesAround_i = l_topology->getTrianglesAroundVertex(v_i);
                const auto nbTrianglesAround = trianglesAround_i.size();
                pointValues[v_i] = std::accumulate(trianglesAround_i.begin(), trianglesAround_i.end(), Coord_t<In>{},
                    [&cellValues](const Coord_t<In>& a, const sofa::Index triangleId)
                    {
                        return a + cellValues[triangleId];
                    });
                if (nbTrianglesAround > 0)
                {
                    pointValues[v_i] /= nbTrianglesAround;
                }
            }
        }
    }
}

template <class TIn, class TOut>
void CellAveragingMapping<TIn, TOut>::applyJ(
    const core::MechanicalParams* mparams,
    DataVecDeriv_t<Out>& out,
    const DataVecDeriv_t<In>& in)
{
    SOFA_UNUSED(mparams);

    if (l_topology)
    {
        helper::WriteOnlyAccessor< Data<VecDeriv_t<Out>> > pointValues = out;
        helper::ReadAccessor< Data<VecDeriv_t<In> > > cellValues = in;

        const auto nbPoints = l_topology->getNbPoints();
        pointValues.resize(nbPoints);

        const auto& triangles = l_topology->getTriangles();
        if (triangles.size() != cellValues.size())
        {
            msg_error() << "Triangle count does not match input size.";
        }
        else
        {
            for (std::size_t v_i = 0; v_i < nbPoints; ++v_i)
            {
                const auto& trianglesAround_i = l_topology->getTrianglesAroundVertex(v_i);
                const auto nbTrianglesAround = trianglesAround_i.size();
                pointValues[v_i] += std::accumulate(trianglesAround_i.begin(), trianglesAround_i.end(), Deriv_t<In>{},
                    [&cellValues](const Deriv_t<In>& a, const sofa::Index triangleId)
                    {
                        return a + cellValues[triangleId];
                    });
            }
        }
    }
}

template <class TIn, class TOut>
void CellAveragingMapping<TIn, TOut>::applyJT(
    const core::MechanicalParams* mparams,
    DataVecDeriv_t<In>& out,
    const DataVecDeriv_t<Out>& in)
{
    SOFA_UNUSED(mparams);

    if (l_topology)
    {
        helper::WriteOnlyAccessor< Data<VecDeriv_t<In>> > cellValues = out;
        helper::ReadAccessor< Data<VecDeriv_t<Out> > > pointValues = in;

        const auto& triangles = l_topology->getTriangles();
        const auto nbTriangles = triangles.size();
        cellValues.resize(nbTriangles);

        for (std::size_t t_i = 0; t_i < nbTriangles; ++t_i)
        {
            const auto& [t0, t1, t2] = triangles[t_i].array();
            cellValues[t_i] += pointValues[t0] + pointValues[t1] + pointValues[t2];
        }
    }
}

template <class TIn, class TOut>
void CellAveragingMapping<TIn, TOut>::applyJT(
    const core::ConstraintParams* mparams, DataMatrixDeriv_t<In>& out,
    const DataMatrixDeriv_t<Out>& in)
{
    SOFA_UNUSED(mparams);

    auto childMatRa  = sofa::helper::getReadAccessor(in);
    auto parentMatWa = sofa::helper::getWriteAccessor(out);
    sofa::linearalgebra::addMultTransposeEigen(parentMatWa.wref(), m_J.compressedMatrix, childMatRa.ref());
}

template <class TIn, class TOut>
const type::vector<sofa::linearalgebra::BaseMatrix*>*
CellAveragingMapping<TIn, TOut>::getJs()
{
    return &m_Js;
}

template <class TIn, class TOut>
void CellAveragingMapping<TIn, TOut>::buildJMatrix()
{
    if (l_topology)
    {
        const auto nbPoints = l_topology->getNbPoints();
        const auto& triangles = l_topology->getTriangles();

        m_J.compressedMatrix.resize( nbPoints, triangles.size());

        for (std::size_t v_i = 0; v_i < nbPoints; ++v_i)
        {
            m_J.compressedMatrix.startVec( v_i );

            const auto& trianglesAround_i = l_topology->getTrianglesAroundVertex(v_i);
            for (const auto& t : trianglesAround_i)
            {
                m_J.compressedMatrix.insertBack( v_i, t ) = 1;
            }
        }
    }
}

}
