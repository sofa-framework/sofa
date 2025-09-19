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
#include <sofa/component/mapping/nonlinear/BaseNonLinearMapping.inl>
#include <sofa/core/BaseLocalMappingMatrix.h>

namespace sofa::component::mapping::nonlinear
{

using sofa::type::dyad;
using sofa::type::dot;

template <class TIn, class TOut>
AreaMapping<TIn, TOut>::AreaMapping()
    : l_topology(initLink("topology", "link to the topology container"))
{}

template <class TIn, class TOut>
auto AreaMapping<TIn, TOut>::computeSecondDerivativeArea(
    const sofa::type::fixed_array<sofa::type::Vec3, 3>& triangleVertices) ->
    sofa::type::Mat<3, 3, sofa::type::Mat<3, 3, Real> >
{
    const auto& v = triangleVertices;

    const auto N = sofa::type::cross(v[1] - v[0], v[2] - v[0]);
    const auto n2 = dot(N, N);

    sofa::type::MatNoInit<3, 3, sofa::type::MatNoInit<3, 3, Real>> d2A;

    const auto ka = 1 / (2 * std::sqrt(std::pow(n2, 3)));

    static constexpr auto skewSign = type::crossProductMatrix(sofa::type::Vec<3, Real>{1,1,1});

    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            auto& entry = d2A(i,j);

            const auto i1 = (i + 1) % 3;
            const auto j1 = (j + 1) % 3;
            const auto i2 = (i + 2) % 3;
            const auto j2 = (j + 2) % 3;

            const auto N_cross_Pi1Pi2 = N.cross(v[i1] - v[i2]);
            const auto N_cross_Pj1Pj2 = N.cross(v[j1] - v[j2]);

            const auto outer_a = dyad(N_cross_Pi1Pi2, N_cross_Pj1Pj2);
            static const auto& id = sofa::type::Mat<3, 3, SReal>::Identity();

            const auto dot_product = dot(v[i1] - v[i2], v[j1] - v[j2]);
            const auto outer_b = dyad(v[j1] - v[j2], v[i1] - v[i2]);

            entry = - outer_a + n2 * (dot_product * id - outer_b);

            if (i != j) // diagonal blocks are skipped because skewSign(i,j) == 0
            {
                const auto sign = skewSign(i,j);
                entry += sign * n2 * type::crossProductMatrix(N);
            }

            entry *= ka;
        }
    }

    return d2A;
}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::init()
{
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

    const auto nbTriangles = l_topology->getNbTriangles();

    if (nbTriangles == 0)
    {
        msg_error() << "No topology component containing triangles found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (l_topology->getNbQuads())
    {
        msg_error() << "Quads are found in the topology, but they are not supported in this component. Consider converting them to triangles.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();

    this->getToModel()->resize( nbTriangles );
    this->m_jacobian.resizeBlocks(nbTriangles, pos.size());

    Inherit1::init();

    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Invalid)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams,
    DataVecCoord_t<Out>& out, const DataVecCoord_t<In>& in)
{
    SOFA_UNUSED( mparams );

    helper::WriteOnlyAccessor< Data<VecCoord_t<Out>> > _out = out;
    helper::ReadAccessor< Data<VecCoord_t<In> > > _in = in;

    m_vertices = _in.operator->();

    const auto& triangles = l_topology->getTriangles();

    this->m_jacobian.clear();

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
            JacobianEntry{triangle[0], k * sofa::type::cross(n1-n2, N)}, // dArea_dn0
            JacobianEntry{triangle[1], k * sofa::type::cross(n02, N)}, // dArea_dn1
            JacobianEntry{triangle[2],-k * sofa::type::cross(n01, N)}, // dArea_dn2
        };

        //insertion in increasing column order
        std::sort(jacobianEntries.begin(), jacobianEntries.end());

        this->m_jacobian.beginRow(triangleId);
        for (const auto& [vertexId, jacobianValue] : jacobianEntries)
        {
            for (unsigned d = 0; d < In::spatial_dimensions; ++d)
            {
                this->m_jacobian.insertBack(triangleId, vertexId * Nin + d, jacobianValue[d]);
            }
        }
    }

    this->m_jacobian.compress();
}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::matrixFreeApplyDJT(
    const core::MechanicalParams* mparams, Real kFactor,
    Data<VecDeriv_t<In>>& parentForce,
    const Data<VecDeriv_t<In>>& parentDisplacement,
    const Data<VecDeriv_t<Out>>& childForce)
{
    SOFA_UNUSED(mparams);
    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();

    helper::WriteAccessor parentForceAccessor(parentForce);
    helper::ReadAccessor parentDisplacementAccessor(parentDisplacement);
    helper::ReadAccessor childForceAccessor(childForce);

    const auto& triangles = l_topology->getTriangles();
    for (unsigned int triangleId = 0; triangleId < triangles.size(); ++triangleId)
    {
        const Deriv_t<Out>& childForceTri = childForceAccessor[triangleId];

        if( childForceTri[0] < 0 || geometricStiffness==1 )
        {
            const auto& triangle = triangles[triangleId];

            const sofa::type::fixed_array<Coord_t<In>, 3> v{
                (*m_vertices)[triangle[0]],
                (*m_vertices)[triangle[1]],
                (*m_vertices)[triangle[2]]
            };

            //it's a 3x3 matrix, where each entry is a 3x3 matrix
            const auto d2Area_d2x = computeSecondDerivativeArea(v);

            for (unsigned int i = 0; i < 3; ++i)
            {
                for (unsigned int j = 0; j < 3; ++j)
                {
                    parentForceAccessor[triangle[i]] +=
                        kFactor
                        * d2Area_d2x(i,j)
                        * parentDisplacementAccessor[triangle[j]]
                        * childForceTri[0];
                }
            }
        }
    }
}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::doUpdateK(const core::MechanicalParams* mparams,
    const Data<VecDeriv_t<Out>>& childForce, SparseKMatrixEigen& matrix)
{
    SOFA_UNUSED(mparams);
    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();

    const helper::ReadAccessor childForceAccessor(childForce);

    const auto& triangles = l_topology->getTriangles();
    for (unsigned int triangleId = 0; triangleId < triangles.size(); ++triangleId)
    {
        const Deriv_t<Out>& childForceTri = childForceAccessor[triangleId];

        if( childForceTri[0] < 0 || geometricStiffness==1 )
        {
            const auto& triangle = triangles[triangleId];

            const sofa::type::fixed_array<Coord_t<In>, 3> v{
                (*m_vertices)[triangle[0]],
                (*m_vertices)[triangle[1]],
                (*m_vertices)[triangle[2]]
            };

            //it's a 3x3 matrix, where each entry is a 3x3 matrix
            const auto d2Area_d2x = computeSecondDerivativeArea(v);

            for (unsigned int i = 0; i < 3; ++i)
            {
                for (unsigned int j = 0; j < 3; ++j)
                {
                    matrix.addBlock(triangle[i], triangle[j], d2Area_d2x(i,j) * childForceTri[0]);
                }
            }
        }
    }
}

template <class TIn, class TOut>
void AreaMapping<TIn, TOut>::buildGeometricStiffnessMatrix(
    sofa::core::GeometricStiffnessMatrix* matrices)
{
    const unsigned& geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness )
    {
        return;
    }

    const auto childForce = this->toModel->readTotalForces();
    const auto dJdx = matrices->getMappingDerivativeIn(this->fromModel).withRespectToPositionsIn(this->fromModel);

    const auto& triangles = l_topology->getTriangles();
    for (unsigned int triangleId = 0; triangleId < triangles.size(); ++triangleId)
    {
        const Deriv_t<Out>& childForceTri = childForce[triangleId];

        if( childForceTri[0] < 0 || geometricStiffness==1 )
        {
            const auto& triangle = triangles[triangleId];

            const sofa::type::fixed_array<Coord_t<In>, 3> v{
                (*m_vertices)[triangle[0]],
                (*m_vertices)[triangle[1]],
                (*m_vertices)[triangle[2]]
            };

            //it's a 3x3 matrix, where each entry is a 3x3 matrix
            const auto d2Area_d2x = computeSecondDerivativeArea(v);

            for (unsigned int i = 0; i < 3; ++i)
            {
                for (unsigned int j = 0; j < 3; ++j)
                {
                    dJdx(triangle[i] * Nin, triangle[j] * Nin) += d2Area_d2x(i,j) * childForceTri[0];
                }
            }
        }
    }

}


}
