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
#include <sofa/component/mapping/nonlinear/AreaMapping.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraintEigenUtils.h>

namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
AreaMapping<TIn, TOut>::AreaMapping()
    : d_computeRestArea(initData(&d_computeRestArea, false, "computeRestArea", "if 'computeRestArea = true', then rest area of each element equal 0, otherwise rest area is the initial area of each of them"))
    , d_restArea(initData(&d_restArea, "restArea", "Rest area of the surface primitives"))
    , l_topology(initLink("topology", "link to the topology container"))
{}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::init()
{
    if (l_topology.empty())
    {
        msg_warning() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    const auto nbSurfacePrimitives = l_topology->getNbQuads() + l_topology->getNbTriangles();

    if (nbSurfacePrimitives == 0)
    {
        msg_error() << "No topology component containing surface primitives found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();

    this->getToModel()->resize( nbSurfacePrimitives );
    jacobian.resizeBlocks(nbSurfacePrimitives, pos.size());

    Inherit1::init();
}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams,
    OutDataVecCoord& out, const InDataVecCoord& in)
{
    helper::WriteOnlyAccessor< Data<OutVecCoord> > _out = out;
    helper::ReadAccessor< Data<InVecCoord> > _in = in;

    const auto& triangles = l_topology->getTriangles();
    const auto& quads = l_topology->getQuads();

    jacobian.clear();

    for (unsigned int triangleId = 0; triangleId < triangles.size(); ++triangleId)
    {
        const auto& triangle = triangles[triangleId];

        const auto& n0 = TIn::getCPos(_in[triangle[0]]);
        const auto& n1 = TIn::getCPos(_in[triangle[1]]);
        const auto& n2 = TIn::getCPos(_in[triangle[2]]);

        const auto n01 = n1 - n0;
        const auto n02 = n2 - n0;
        const auto N = sofa::type::cross(n01, n02);
        const auto norm = N.norm();

        const auto area = static_cast<typename In::Real>(0.5) * norm;

        _out[triangleId] = area;

        const auto k = 1 / (2 * norm);

        sofa::type::fixed_array<JacobianEntry, 3> jacobianEntries {
            JacobianEntry{triangle[0], k * sofa::type::cross(n1-n2, N)},
            JacobianEntry{triangle[1], k * sofa::type::cross(n02, N)},
            JacobianEntry{triangle[2],-k * sofa::type::cross(n01, N)},
        };

        //insertion in increasing column order
        std::sort(jacobianEntries.begin(), jacobianEntries.end());

        jacobian.beginRow(triangleId);
        for (const auto& [vertexId, jacobianValue] : jacobianEntries)
        {
            for (unsigned d = 0; d < In::spatial_dimensions; ++d)
            {
                jacobian.insertBack(triangleId, vertexId * Nin + d, jacobianValue[d]);
            }
        }
    }

    for (unsigned int quadId = 0; quadId < quads.size(); ++quadId)
    {
        const auto& quad = quads[quadId];
        const auto quadArea = sofa::geometry::Quad::area(
            TIn::getCPos(_in[quad[0]]),
            TIn::getCPos(_in[quad[1]]),
            TIn::getCPos(_in[quad[2]]),
            TIn::getCPos(_in[quad[3]]));

        _out[triangles.size() + quadId] = quadArea;
    }

    jacobian.compress();
}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::applyJ(const core::MechanicalParams* mparams,
    OutDataVecDeriv& out, const InDataVecDeriv& in)
{
    if( jacobian.rowSize() )
    {
        auto dOutWa = sofa::helper::getWriteOnlyAccessor(out);
        auto dInRa = sofa::helper::getReadAccessor(in);
        jacobian.mult(dOutWa.wref(),dInRa.ref());
    }
}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::applyJT(const core::MechanicalParams* mparams,
    InDataVecDeriv& out, const OutDataVecDeriv& in)
{
    if( jacobian.rowSize() )
    {
        auto dOutRa = sofa::helper::getReadAccessor(in);
        auto dInWa = sofa::helper::getWriteOnlyAccessor(out);
        jacobian.addMultTranspose(dInWa.wref(),dOutRa.ref());
    }
}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::applyJT(const core::ConstraintParams* cparams,
    InDataMatrixDeriv& out, const OutDataMatrixDeriv& in)
{
    SOFA_UNUSED(cparams);
    auto childMatRa  = sofa::helper::getReadAccessor(in);
    auto parentMatWa = sofa::helper::getWriteAccessor(out);
    addMultTransposeEigen(parentMatWa.wref(), jacobian.compressedMatrix, childMatRa.ref());
}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams,
    core::MultiVecDerivId parentForceId, core::ConstMultiVecDerivId childForceId)
{
    const unsigned geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness )
    {
        return;
    }

    helper::WriteAccessor<Data<InVecDeriv> > parentForceAccessor(*parentForceId[this->fromModel.get()].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacementAccessor(*mparams->readDx(this->fromModel.get()));
    const SReal kfactor = mparams->kFactor();
    helper::ReadAccessor<Data<OutVecDeriv> > childForceAccessor(mparams->readF(this->toModel.get()));
}

}
