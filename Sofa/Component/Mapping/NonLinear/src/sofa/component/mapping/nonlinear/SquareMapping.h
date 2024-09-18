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

#include <sofa/component/mapping/nonlinear/AssembledNonLinearMapping.h>
#include <sofa/component/mapping/nonlinear/NonLinearMappingData.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>

namespace sofa::component::mapping::nonlinear
{

/**
    x -> x²

    @author Matthieu Nesme
    @date 2016

*/
template <class TIn, class TOut>
class SquareMapping : public AssembledNonLinearMapping<TIn, TOut, false>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SquareMapping,TIn,TOut), SOFA_TEMPLATE3(AssembledNonLinearMapping,TIn,TOut,false));

    using In = TIn;
    using Out = TOut;

    using Real = Real_t<Out>;

    static constexpr auto Nin = In::deriv_total_size;

    void init() override;

    void apply(const core::MechanicalParams *mparams, DataVecCoord_t<Out>& out, const DataVecCoord_t<In>& in) override;
    void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForce ) override;
    void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices) override;

protected:

    void matrixFreeApplyDJT(const core::MechanicalParams* mparams, Real kFactor,
                            Data<VecDeriv_t<In> >& parentForce,
                            const Data<VecDeriv_t<In> >& parentDisplacement,
                            const Data<VecDeriv_t<Out> >& childForce) override;
};




#if !defined(SOFA_COMPONENT_MAPPING_SquareMapping_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API SquareMapping< defaulttype::Vec1Types, defaulttype::Vec1Types >;
#endif

} // namespace sofa::component::mapping::nonlinear
