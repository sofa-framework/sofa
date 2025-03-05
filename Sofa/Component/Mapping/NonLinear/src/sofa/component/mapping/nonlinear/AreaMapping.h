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

#include <sofa/component/mapping/nonlinear/config.h>
#include <sofa/component/mapping/nonlinear/BaseNonLinearMapping.h>
#include <sofa/component/mapping/nonlinear/NonLinearMappingData.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>


namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
class AreaMapping : public BaseNonLinearMapping<TIn, TOut, true>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(AreaMapping,TIn,TOut), SOFA_TEMPLATE3(BaseNonLinearMapping,TIn,TOut, true));

    using In = TIn;
    using Out = TOut;

    using Real = Real_t<Out>;

    static constexpr auto Nin = In::deriv_total_size;

    SingleLink<AreaMapping<TIn, TOut>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    static sofa::type::Mat<3,3,sofa::type::Mat<3,3,Real>> computeSecondDerivativeArea(const sofa::type::fixed_array<sofa::type::Vec3, 3>& triangleVertices);

    void init() override;

    void apply(const core::MechanicalParams* mparams, DataVecCoord_t<Out>& out, const DataVecCoord_t<In>& in) override;
    void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices) override;

protected:
    AreaMapping();

    void matrixFreeApplyDJT(const core::MechanicalParams* mparams, Real kFactor,
                            Data<VecDeriv_t<In> >& parentForce,
                            const Data<VecDeriv_t<In> >& parentDisplacement,
                            const Data<VecDeriv_t<Out> >& childForce) override;

    using typename Inherit1::SparseKMatrixEigen;

    void doUpdateK(
        const core::MechanicalParams* mparams, const Data<VecDeriv_t<Out> >& childForce,
        SparseKMatrixEigen& matrix) override;

    const VecCoord_t<In>* m_vertices{nullptr};

    using JacobianEntry = typename Inherit1::JacobianEntry;
};

#if !defined(SOFA_COMPONENT_MAPPING_NONLINEAR_AREAMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API AreaMapping< defaulttype::Vec3Types, defaulttype::Vec1Types >;
#endif

}
