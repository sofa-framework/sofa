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
#include <sofa/component/mapping/nonlinear/VolumeMapping.h>
#include <sofa/core/BaseLocalMappingMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraintEigenUtils.h>

namespace sofa::component::mapping::nonlinear
{

using sofa::type::dyad;
using sofa::type::dot;

template <class TIn, class TOut>
VolumeMapping<TIn, TOut>::VolumeMapping()
    : l_topology(initLink("topology", "link to the topology container"))
{}

template <class TIn, class TOut>
auto VolumeMapping<TIn, TOut>::computeSecondDerivativeVolume(
    const sofa::type::fixed_array<sofa::type::Vec3, 4>& tetrahedronVertices) ->
    sofa::type::Mat<4, 4, sofa::type::Mat<3, 3, Real> >
{
    const sofa::type::fixed_array<sofa::type::Vec3, 3> v {
        tetrahedronVertices[1] - tetrahedronVertices[0],
        tetrahedronVertices[2] - tetrahedronVertices[0],
        tetrahedronVertices[3] - tetrahedronVertices[0]
    };

    const auto H12 = sofa::type::crossProductMatrix(v[1] - v[2]) / 6;
    const auto H13 = sofa::type::crossProductMatrix(v[2] - v[0]) / 6;
    const auto H14 = sofa::type::crossProductMatrix(v[0] - v[1]) / 6;

    const auto H23 = sofa::type::crossProductMatrix(-v[2]) / 6;
    const auto H24 = sofa::type::crossProductMatrix(v[1]) / 6;
    const auto H34 = sofa::type::crossProductMatrix(-v[0]) / 6;

    sofa::type::Mat<4, 4, sofa::type::Mat<3, 3, SReal> > hessian;

    hessian(0, 1) = H12;
    hessian(1, 0) = H12.transposed();
    hessian(0, 2) = H13;
    hessian(2, 0) = H13.transposed();
    hessian(0, 3) = H14;
    hessian(3, 0) = H14.transposed();
    hessian(1, 2) = H23;
    hessian(2, 1) = H23.transposed();
    hessian(1, 3) = H24;
    hessian(3, 1) = H24.transposed();
    hessian(2, 3) = H34;
    hessian(3, 2) = H34.transposed();

    return hessian;
}

template <class TIn, class TOut>
void VolumeMapping<TIn, TOut>::init()
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

    const auto nbTetrahedra = l_topology->getNbTetrahedra();

    if (l_topology->getNbTetrahedra() == 0)
    {
        msg_error() << "No topology component containing tetrahedron found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (l_topology->getNbHexahedra())
    {
        msg_error() << "Hexahedra are found in the topology, but they are not supported in this component. Consider converting them to tetrahedra.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    typename core::behavior::MechanicalState<In>::ReadVecCoord pos = this->getFromModel()->readPositions();

    this->getToModel()->resize( nbTetrahedra );
    jacobian.resizeBlocks(nbTetrahedra, pos.size());

    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    Inherit1::init();

    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Invalid)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class TIn, class TOut>
void VolumeMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams,
    DataVecCoord_t<Out>& out, const DataVecCoord_t<In>& in)
{
    helper::WriteOnlyAccessor< Data<VecCoord_t<Out>> > _out = out;
    helper::ReadAccessor< Data<VecCoord_t<In> > > _in = in;

    m_vertices = _in.operator->();

    const auto& tetrahedra = l_topology->getTetrahedra();

    jacobian.clear();

    for (unsigned int tetId = 0; tetId < tetrahedra.size(); ++tetId)
    {
        const auto& tetra = tetrahedra[tetId];

        const sofa::type::fixed_array<sofa::type::Vec3, 4> vertices {
            TIn::getCPos(_in[tetra[0]]),
            TIn::getCPos(_in[tetra[1]]),
            TIn::getCPos(_in[tetra[2]]),
            TIn::getCPos(_in[tetra[3]])
        };

        const sofa::type::fixed_array<sofa::type::Vec3, 3> v {
            vertices[1] - vertices[0],
            vertices[2] - vertices[0],
            vertices[3] - vertices[0]
        };

        const auto volume = std::abs(sofa::type::dot(v[0], sofa::type::cross(v[1], v[2]))) / 6;

        _out[tetId] = volume;

        const sofa::type::fixed_array<sofa::type::Vec3, 3> c {
            sofa::type::cross(v[1], v[2]) / 6,
            sofa::type::cross(v[2], v[0]) / 6,
            sofa::type::cross(v[0], v[1]) / 6
        };

        sofa::type::fixed_array<JacobianEntry, 4> jacobianEntries {
            JacobianEntry{tetra[0], -(c[0] + c[1] + c[2])}, // dVol_dv0
            JacobianEntry{tetra[1], c[0]}, // dVol_dv1
            JacobianEntry{tetra[2], c[1]}, // dVol_dv2
            JacobianEntry{tetra[3], c[2]}, // dVol_dv3
        };

        //insertion in increasing column order
        std::sort(jacobianEntries.begin(), jacobianEntries.end());

        jacobian.beginRow(tetId);
        for (const auto& [vertexId, jacobianValue] : jacobianEntries)
        {
            for (unsigned d = 0; d < In::spatial_dimensions; ++d)
            {
                jacobian.insertBack(tetId, vertexId * Nin + d, jacobianValue[d]);
            }
        }
    }

    jacobian.compress();
}

template <class TIn, class TOut>
void VolumeMapping<TIn, TOut>::applyJ(const core::MechanicalParams* mparams,
    DataVecDeriv_t<Out>& out, const DataVecDeriv_t<In>& in)
{
    if( jacobian.rowSize() )
    {
        auto dOutWa = sofa::helper::getWriteOnlyAccessor(out);
        auto dInRa = sofa::helper::getReadAccessor(in);
        jacobian.mult(dOutWa.wref(),dInRa.ref());
    }
}

template <class TIn, class TOut>
void VolumeMapping<TIn, TOut>::applyJT(const core::MechanicalParams* mparams,
    DataVecDeriv_t<In>& out, const DataVecDeriv_t<Out>& in)
{
    if( jacobian.rowSize() )
    {
        auto dOutRa = sofa::helper::getReadAccessor(in);
        auto dInWa = sofa::helper::getWriteOnlyAccessor(out);
        jacobian.addMultTranspose(dInWa.wref(),dOutRa.ref());
    }
}

template <class TIn, class TOut>
void VolumeMapping<TIn, TOut>::applyJT(const core::ConstraintParams* cparams,
    DataMatrixDeriv_t<In>& out, const DataMatrixDeriv_t<Out>& in)
{
    SOFA_UNUSED(cparams);
    auto childMatRa  = sofa::helper::getReadAccessor(in);
    auto parentMatWa = sofa::helper::getWriteAccessor(out);
    addMultTransposeEigen(parentMatWa.wref(), jacobian.compressedMatrix, childMatRa.ref());
}

template <class TIn, class TOut>
void VolumeMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams,
    core::MultiVecDerivId parentForceId, core::ConstMultiVecDerivId childForceId)
{
    if (!m_vertices)
    {
        return;
    }

    const unsigned geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness )
    {
        return;
    }

    helper::WriteAccessor<Data<VecDeriv_t<In> > > parentForceAccessor(*parentForceId[this->fromModel.get()].write());
    helper::ReadAccessor<Data<VecDeriv_t<In> > > parentDisplacementAccessor(*mparams->readDx(this->fromModel.get()));
    const SReal kFactor = mparams->kFactor();
    helper::ReadAccessor<Data<VecDeriv_t<Out> > > childForceAccessor(mparams->readF(this->toModel.get()));

    if( K.compressedMatrix.nonZeros() )
    {
        K.addMult( parentForceAccessor.wref(), parentDisplacementAccessor.ref(), (typename In::Real)kFactor );
    }
    else
    {
        const auto& tetrahedra = l_topology->getTetrahedra();
        for (unsigned int tetId = 0; tetId < tetrahedra.size(); ++tetId)
        {
            const Deriv_t<Out>& childForceTetra = childForceAccessor[tetId];

            if( childForceTetra[0] < 0 || geometricStiffness==1 )
            {
                const auto& tetra = tetrahedra[tetId];

                const sofa::type::fixed_array<Coord_t<In>, 4> v{
                    (*m_vertices)[tetra[0]],
                    (*m_vertices)[tetra[1]],
                    (*m_vertices)[tetra[2]],
                    (*m_vertices)[tetra[3]],
                };

                //it's a 4x4 matrix, where each entry is a 3x3 matrix
                const auto d2Vol_d2x = computeSecondDerivativeVolume(v);

                for (unsigned int i = 0; i < 4; ++i)
                {
                    for (unsigned int j = 0; j < 4; ++j)
                    {
                        parentForceAccessor[tetra[i]] +=
                            kFactor
                            * d2Vol_d2x[i][j]
                            * parentDisplacementAccessor[tetra[j]]
                            * childForceTetra[0];
                    }
                }
            }
        }
    }
}

template <class TIn, class TOut>
void VolumeMapping<TIn, TOut>::updateK(const core::MechanicalParams* mparams,
    core::ConstMultiVecDerivId childForceId)
{
    SOFA_UNUSED(mparams);
    const unsigned geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness ) { K.resize(0,0); return; }

    helper::ReadAccessor<Data<VecDeriv_t<Out> > > childForce( *childForceId[this->toModel.get()].read() );

    {
        unsigned int kSize = this->fromModel->getSize();
        K.resizeBlocks(kSize, kSize);
    }

    const auto& tetrahedra = l_topology->getTetrahedra();
    for (unsigned int tetId = 0; tetId < tetrahedra.size(); ++tetId)
    {
        const Deriv_t<Out>& childForceTri = childForce[tetId];

        if( childForceTri[0] < 0 || geometricStiffness==1 )
        {
            const auto& tetra = tetrahedra[tetId];

            const sofa::type::fixed_array<Coord_t<In>, 4> v{
                (*m_vertices)[tetra[0]],
                (*m_vertices)[tetra[1]],
                (*m_vertices)[tetra[2]],
                (*m_vertices)[tetra[3]]
            };

            //it's a 4x4 matrix, where each entry is a 3x3 matrix
            const auto d2Volume_d2x = computeSecondDerivativeVolume(v);

            for (unsigned int i = 0; i < 4; ++i)
            {
                for (unsigned int j = 0; j < 4; ++j)
                {
                    K.addBlock(tetra[i], tetra[j], d2Volume_d2x[i][j] * childForceTri[0]);
                }
            }
        }
    }

    K.compress();
}

template <class TIn, class TOut>
const linearalgebra::BaseMatrix* VolumeMapping<TIn, TOut>::getK()
{
    return &K;
}

template <class TIn, class TOut>
void VolumeMapping<TIn, TOut>::buildGeometricStiffnessMatrix(
    sofa::core::GeometricStiffnessMatrix* matrices)
{
    const unsigned& geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness )
    {
        return;
    }

    const auto childForce = this->toModel->readTotalForces();
    const auto dJdx = matrices->getMappingDerivativeIn(this->fromModel).withRespectToPositionsIn(this->fromModel);

    const auto& tetrahedra = l_topology->getTetrahedra();
    for (unsigned int tetId = 0; tetId < tetrahedra.size(); ++tetId)
    {
        const Deriv_t<Out>& childForceTri = childForce[tetId];

        if( childForceTri[0] < 0 || geometricStiffness==1 )
        {
            const auto& tetra = tetrahedra[tetId];

            const sofa::type::fixed_array<Coord_t<In>, 4> v{
                (*m_vertices)[tetra[0]],
                (*m_vertices)[tetra[1]],
                (*m_vertices)[tetra[2]],
                (*m_vertices)[tetra[3]]
            };

            //it's a 4x4 matrix, where each entry is a 3x3 matrix
            const auto d2Vol_d2x = computeSecondDerivativeVolume(v);

            for (unsigned int i = 0; i < 4; ++i)
            {
                for (unsigned int j = 0; j < 4; ++j)
                {
                    dJdx(tetra[i] * Nin, tetra[j] * Nin) += d2Vol_d2x[i][j] * childForceTri[0];
                }
            }
        }
    }

}

template <class TIn, class TOut>
const type::vector<sofa::linearalgebra::BaseMatrix*>* VolumeMapping<TIn, TOut>::
getJs()
{
    return &baseMatrices;
}

}
