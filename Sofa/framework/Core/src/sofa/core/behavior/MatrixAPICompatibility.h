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

#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/MatrixAccumulator.h>
#include <sofa/core/behavior/DefaultMultiMatrixAccessor.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>

namespace sofa::core::behavior
{

/// This class exists only for compatibility reasons. To be removed once the deprecated API
/// addKToMatrix and addMToMatrix is removed
class MatrixAccessorCompat : public sofa::core::behavior::DefaultMultiMatrixAccessor
{
public:
    MatrixRef getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const override
    {
        MatrixRef r;
        const auto& matrix = matrixMap.at(sofa::type::fixed_array<const sofa::core::behavior::BaseMechanicalState*, 2>{mstate, mstate});
        r.matrix = matrix.get();
        r.offset = 0;
        return r;
    }


    InteractionMatrixRef getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const override
    {
        InteractionMatrixRef r;
        const auto& matrix = matrixMap.at(sofa::type::fixed_array<const sofa::core::behavior::BaseMechanicalState*, 2>{mstate1, mstate2});
        r.matrix = matrix.get();
        r.offRow = 0;
        r.offCol = 0;
        return r;
    }

    void setMatrix(sofa::core::behavior::BaseMechanicalState* mstate1, sofa::core::behavior::BaseMechanicalState* mstate2, std::shared_ptr<linearalgebra::BaseMatrix> matrix)
    {
        matrixMap[sofa::type::fixed_array<const sofa::core::behavior::BaseMechanicalState*, 2>{mstate1, mstate2}] = matrix;
    }

    std::map<
        sofa::type::fixed_array<const sofa::core::behavior::BaseMechanicalState*, 2>, std::shared_ptr<linearalgebra::BaseMatrix> > matrixMap;
};

/// A fake BaseMatrix redirecting its add methods to the MatrixAccumulator API
/// This class exists only for compatibility reasons. To be removed once the deprecated API
/// addKToMatrix, addBToMatrix and addMToMatrix is removed
template<matrixaccumulator::Contribution c>
class AddToMatrixCompatMatrix : public sofa::linearalgebra::BaseMatrix
{
public:
    static constexpr const char* compatibilityMessage = "This message appears only for compatibility"
        " of the deprecated API addKToMatrix, addBToMatrix and addMToMatrix. Update your code with "
        "the new API buildStiffnessMatrix, buildDampingMatrix or buildMassMatrix to remove this warning. ";

    ~AddToMatrixCompatMatrix() override = default;
    Index rowSize() const override
    {
        msg_error(component) << compatibilityMessage << "rowSize is not a supported operation in the compatibility";
        return {};
    }
    Index colSize() const override
    {
        msg_error(component) << compatibilityMessage << "colSize is not a supported operation in the compatibility";
        return {};
    }
    SReal element(Index i, Index j) const override
    {
        SOFA_UNUSED(i);
        SOFA_UNUSED(j);

        msg_error(component) << compatibilityMessage << "element is not a supported operation in the compatibility";
        return {};
    }
    void resize(Index nbRow, Index nbCol) override
    {
        SOFA_UNUSED(nbRow);
        SOFA_UNUSED(nbCol);

        msg_error(component) << compatibilityMessage << "resize is not a supported operation in the compatibility";
    }
    void clear() override
    {
        msg_error(component) << compatibilityMessage << "clear is not a supported operation in the compatibility";
    }
    void set(Index i, Index j, double v) override
    {
        SOFA_UNUSED(i);
        SOFA_UNUSED(j);
        SOFA_UNUSED(v);

        msg_error(component) << compatibilityMessage << "set is not a supported operation in the compatibility";
    }
    void add(Index row, Index col, double v) override
    {
        if constexpr (c == matrixaccumulator::Contribution::MASS)
        {
            matrices->add(row, col, v);
        }
        else if constexpr (c == matrixaccumulator::Contribution::STIFFNESS)
        {
            const auto dfdx = matrices->getForceDerivativeIn(mstate1).withRespectToPositionsIn(mstate2);
            dfdx(row, col) += v;
        }
        else if constexpr (c == matrixaccumulator::Contribution::DAMPING)
        {
            const auto dfdx = matrices->getForceDerivativeIn(mstate1).withRespectToVelocityIn(mstate2);
            dfdx(row, col) += v;
        }
    }
    void add(Index row, Index col, const type::Mat3x3d& _M) override
    {
        if constexpr (c == matrixaccumulator::Contribution::MASS)
        {
            matrices->add(row, col, _M);
        }
        else if constexpr (c == matrixaccumulator::Contribution::STIFFNESS)
        {
            const auto dfdx = matrices->getForceDerivativeIn(mstate1).withRespectToPositionsIn(mstate2);
            dfdx(row, col) += _M;
        }
        else if constexpr (c == matrixaccumulator::Contribution::DAMPING)
        {
            const auto dfdx = matrices->getForceDerivativeIn(mstate1).withRespectToVelocityIn(mstate2);
            dfdx(row, col) += _M;
        }
    }
    void add(Index row, Index col, const type::Mat3x3f& _M) override
    {
        if constexpr (c == matrixaccumulator::Contribution::MASS)
        {
            matrices->add(row, col, _M);
        }
        else if constexpr (c == matrixaccumulator::Contribution::STIFFNESS)
        {
            const auto dfdx = matrices->getForceDerivativeIn(mstate1).withRespectToPositionsIn(mstate2);
            dfdx(row, col) += _M;
        }
        else if constexpr (c == matrixaccumulator::Contribution::DAMPING)
        {
            const auto dfdx = matrices->getForceDerivativeIn(mstate1).withRespectToVelocityIn(mstate2);
            dfdx(row, col) += _M;
        }
    }

    sofa::core::matrixaccumulator::get_component_type<c>* component { nullptr };
    sofa::core::matrixaccumulator::get_matrix_builder_type<c>* matrices { nullptr };
    sofa::core::behavior::BaseMechanicalState* mstate1 { nullptr };
    sofa::core::behavior::BaseMechanicalState* mstate2 { nullptr };
};

/// A fake BaseMatrix redirecting its clearRowCol method to the ZeroDirichletCondition API
/// This class exists only for compatibility reasons. To be removed once the deprecated API
/// applyConstraint is removed
class ApplyConstraintCompat : public sofa::linearalgebra::BaseMatrix
{
    static constexpr const char* compatibilityMessage = "This message appears only for compatibility"
        " of the deprecated API applyConstraint. Update your code with the new API "
        " to remove this warning. ";

public:
    Index rowSize() const override
    {
        msg_error(component) << compatibilityMessage << "rowSize is not a supported operation in the compatibility";
        return {};
    }
    Index colSize() const override
    {
        msg_error(component) << compatibilityMessage << "colSize is not a supported operation in the compatibility";
        return {};
    }
    SReal element(Index i, Index j) const override
    {
        SOFA_UNUSED(i);
        SOFA_UNUSED(j);

        msg_error(component) << compatibilityMessage << "element is not a supported operation in the compatibility";
        return {};
    }
    void resize(Index nbRow, Index nbCol) override
    {
        SOFA_UNUSED(nbRow);
        SOFA_UNUSED(nbCol);

        msg_error(component) << compatibilityMessage << "resize is not a supported operation in the compatibility";
    }
    void clear() override
    {
        msg_error(component) << compatibilityMessage << "clear is not a supported operation in the compatibility";
    }
    void set(Index i, Index j, double v) override
    {
        SOFA_UNUSED(i);
        SOFA_UNUSED(j);
        SOFA_UNUSED(v);
    }
    void add(Index row, Index col, double v) override
    {
        SOFA_UNUSED(row);
        SOFA_UNUSED(col);
        SOFA_UNUSED(v);

        msg_error(component) << compatibilityMessage << "add is not a supported operation in the compatibility";
    }

    void clearRowCol(Index i) override
    {
        zeroDirichletCondition->discardRowCol(i, i);
    }


    BaseProjectiveConstraintSet* component { nullptr };
    sofa::core::behavior::ZeroDirichletCondition* zeroDirichletCondition { nullptr };
};
}
