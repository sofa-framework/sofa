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
#include <sofa/core/DerivativeMatrix.h>
#include <sofa/type/fwd.h>
#include <sofa/core/MatrixAccumulator.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/MechanicalStatesMatrixAccumulators.h>

namespace sofa::core::behavior
{
class SOFA_CORE_API StiffnessMatrixAccumulator : public virtual MatrixAccumulatorInterface {};
class SOFA_CORE_API ListStiffnessMatrixAccumulator : public ListMatrixAccumulator<StiffnessMatrixAccumulator>{};

class SOFA_CORE_API DampingMatrixAccumulator : public virtual MatrixAccumulatorInterface {};
class SOFA_CORE_API ListDampingMatrixAccumulator : public ListMatrixAccumulator<DampingMatrixAccumulator>{};

} //namespace sofa::core::behavior

namespace sofa::core::matrixaccumulator
{
template<>
struct get_abstract_strong<Contribution::STIFFNESS>
{
    using type = behavior::StiffnessMatrixAccumulator;
    using ComponentType = behavior::BaseForceField;
    using MatrixBuilderType = sofa::core::behavior::StiffnessMatrix;
};

template<>
struct get_abstract_strong<Contribution::DAMPING>
{
    using type = behavior::DampingMatrixAccumulator;
    using ComponentType = behavior::BaseForceField;
    using MatrixBuilderType = sofa::core::behavior::DampingMatrix;
};

template<>
struct get_list_abstract_strong<Contribution::STIFFNESS>
{
    using type = behavior::ListStiffnessMatrixAccumulator;
    using ComponentType = behavior::BaseForceField;
};

template<>
struct get_list_abstract_strong<Contribution::DAMPING>
{
    using type = behavior::ListDampingMatrixAccumulator;
    using ComponentType = behavior::BaseForceField;
};

} //namespace sofa::core::matrixaccumulator


namespace sofa::core::behavior
{

class SOFA_CORE_API StiffnessMatrix
    : public DerivativeMatrix<matrixaccumulator::Contribution::STIFFNESS>
{
public:

    struct DF
    {
        DF(BaseMechanicalState* _mstate1, StiffnessMatrix* _mat)
            : mstate1(_mstate1), mat(_mat) {}

        Derivative withRespectToPositionsIn(BaseMechanicalState* mstate2) const
        {
            return Derivative{this->mstate1, mstate2, this->mat};
        }

    private:
        BaseMechanicalState* mstate1 { nullptr };
        StiffnessMatrix* mat { nullptr };
    };

    DF getForceDerivativeIn(BaseMechanicalState* mstate)
    {
        return DF{mstate, this};
    }
};

class SOFA_CORE_API DampingMatrix
    : public DerivativeMatrix<matrixaccumulator::Contribution::DAMPING>
{
public:

    struct DF
    {
        DF(BaseMechanicalState* _mstate1, DampingMatrix* _mat)
            : mstate1(_mstate1), mat(_mat) {}

        Derivative withRespectToVelocityIn(BaseMechanicalState* mstate2) const
        {
            return Derivative{this->mstate1, mstate2, this->mat};
        }

    private:
        BaseMechanicalState* mstate1 { nullptr };
        DampingMatrix* mat { nullptr };
    };

    DF getForceDerivativeIn(BaseMechanicalState* mstate)
    {
        return DF{mstate, this};
    }
};

}
