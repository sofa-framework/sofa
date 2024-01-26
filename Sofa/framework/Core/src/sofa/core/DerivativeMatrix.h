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
#include <sofa/core/MechanicalStatesMatrixAccumulators.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa::core
{

template<matrixaccumulator::Contribution c>
class DerivativeMatrix : public MechanicalStatesMatrixAccumulators<c>
{
public:
    using MatrixAccumulator = typename MechanicalStatesMatrixAccumulators<c>::MatrixAccumulator;

    struct DerivativeElement
    {
        DerivativeElement(sofa::SignedIndex _row, sofa::SignedIndex _col, MatrixAccumulator* _mat)
            : row(_row), col(_col), mat(_mat)
        {}
        void operator+=(const float value) const { mat->add(row, col, value); }
        void operator+=(const double value) const { mat->add(row, col, value); }

        template<sofa::Size L, sofa::Size C, class real>
        void operator+=(const sofa::type::Mat<L, C, real> & value) const { mat->matAdd(row, col, value); }

        void operator+=(const sofa::type::Mat<1, 1, float> & value) const { mat->add(row, col, value); }
        void operator+=(const sofa::type::Mat<1, 1, double>& value) const { mat->add(row, col, value); }
        void operator+=(const sofa::type::Mat<2, 2, float> & value) const { mat->add(row, col, value); }
        void operator+=(const sofa::type::Mat<2, 2, double>& value) const { mat->add(row, col, value); }
        void operator+=(const sofa::type::Mat<3, 3, float> & value) const { mat->add(row, col, value); }
        void operator+=(const sofa::type::Mat<3, 3, double>& value) const { mat->add(row, col, value); }
        void operator+=(const sofa::type::Mat<6, 6, float> & value) const { mat->add(row, col, value); }
        void operator+=(const sofa::type::Mat<6, 6, double>& value) const { mat->add(row, col, value); }

        [[nodiscard]] bool isValid() const { return mat != nullptr; }
        operator bool() const { return isValid(); }

    private:
        sofa::SignedIndex row;
        sofa::SignedIndex col;
        MatrixAccumulator* mat { nullptr };
    };

    struct Derivative
    {
        DerivativeElement operator()(sofa::SignedIndex row, sofa::SignedIndex col) const
        {
            return DerivativeElement{row, col, mat};
        }

        Derivative(BaseState* _mstate1,
                   BaseState* _mstate2,
                   DerivativeMatrix* _mat)
        : mstate1(_mstate1)
        , mstate2(_mstate2)
        , mat(_mat->m_submatrix[{_mstate1, _mstate2}])
        {}

        [[nodiscard]] bool isValid() const { return mat != nullptr; }
        operator bool() const { return isValid(); }

        void checkValidity(const objectmodel::BaseObject* object) const
        {
            msg_error_when(!isValid() || !mstate1 || !mstate2, object)
                << "The force derivative in mechanical state '"
                << (mstate1 ? mstate1->getPathName() : "null")
                << "' with respect to state variable in mechanical state '"
                << (mstate2 ? mstate2->getPathName() : "null")
                << "' is invalid";
        }

    private:
        BaseState* mstate1 { nullptr };
        BaseState* mstate2 { nullptr };
        MatrixAccumulator* mat { nullptr };
    };

};


}
