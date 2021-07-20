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

#include <sofa/core/config.h>
#include <sofa/type/fwd.h>
#include <sofa/core/MatrixAccumulator.h>
#include <sofa/core/MechanicalStatesMatrixAccumulators.h>
#include <sofa/core/behavior/BaseMass.h>

namespace sofa::core::behavior
{

class SOFA_CORE_API MassMatrixAccumulator     : public virtual MatrixAccumulatorInterface {};
class SOFA_CORE_API ListMassMatrixAccumulator : public ListMatrixAccumulator<MassMatrixAccumulator>{};

class SOFA_CORE_API MassMatrix : public MassMatrixAccumulator {};

} //namespace sofa::core::behavior

namespace sofa::core::matrixaccumulator
{

template<>
struct get_abstract_strong<Contribution::MASS>
{
    using type = behavior::MassMatrixAccumulator;
    using ComponentType = core::behavior::BaseMass;
    using MatrixBuilderType = behavior::MassMatrixAccumulator;
};

template<>
struct get_list_abstract_strong<Contribution::MASS>
{
    using type = behavior::ListMassMatrixAccumulator;
    using ComponentType = core::behavior::BaseMass;
};

} //namespace sofa::core::matrixaccumulator
