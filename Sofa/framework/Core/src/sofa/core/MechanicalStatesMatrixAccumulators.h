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

#include <sofa/core/MatrixAccumulator.h>

namespace sofa::core
{

template<matrixaccumulator::Contribution c>
class MechanicalStatesMatrixAccumulators
{
public:
    using MatrixAccumulator = matrixaccumulator::get_abstract_strong_type<c>;
    using ComponentType = matrixaccumulator::get_component_type<c>;

    void setMatrixAccumulator(MatrixAccumulator* matrixAccumulator,
                              BaseState* mstate1,
                              BaseState* mstate2);

    void setMatrixAccumulator(MatrixAccumulator* matrixAccumulator,
                              BaseState* mstate1);

    void setMechanicalParams(const core::MechanicalParams* mparams) { m_mparams = mparams; }

    const core::MechanicalParams* getMechanicalParams() const { return m_mparams; }

// protected:

    std::map<std::pair<BaseState*, BaseState*>,
        MatrixAccumulator*> m_submatrix;

public:

    MechanicalStatesMatrixAccumulators() = default;
    virtual ~MechanicalStatesMatrixAccumulators() = default;

private:

    const core::MechanicalParams* m_mparams { nullptr };
};


template <matrixaccumulator::Contribution c>
void MechanicalStatesMatrixAccumulators<c>::setMatrixAccumulator(
    MatrixAccumulator* matrixAccumulator, BaseState* mstate1,
    BaseState* mstate2)
{
    m_submatrix[{mstate1, mstate2}] = matrixAccumulator;
}

template <matrixaccumulator::Contribution c>
void MechanicalStatesMatrixAccumulators<c>::setMatrixAccumulator(
    MatrixAccumulator* matrixAccumulator, BaseState* mstate1)
{
    setMatrixAccumulator(matrixAccumulator, mstate1, mstate1);
}
}
