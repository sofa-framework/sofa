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
#include <sofa/component/mapping/linear/config.h>
#include <sofa/component/mapping/linear/LinearMapping.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
class CellAveragingMapping : public LinearMapping<TIn, TOut>
{
public:
    SOFA_CLASS(CellAveragingMapping, SOFA_TEMPLATE2(LinearMapping, TIn, TOut));

    using In = TIn;
    using Out = TOut;

    CellAveragingMapping();

    void init() override;

    void apply(const core::MechanicalParams* mparams,
        DataVecCoord_t<Out>& out,
        const DataVecCoord_t<In>& in) override;
    void applyJ(const core::MechanicalParams* mparams,
        DataVecDeriv_t<Out>& out,
        const DataVecDeriv_t<In>& in) override;
    void applyJT(const core::MechanicalParams* mparams,
        DataVecDeriv_t<Out>& out,
        const DataVecDeriv_t<In>& in) override;

    SingleLink<CellAveragingMapping, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
};




#if !defined(SOFA_COMPONENT_MAPPING_LINEAR_CELL_AVERAGING_MAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API CellAveragingMapping<defaulttype::Vec1Types, defaulttype::Vec1Types >;
#endif

}
