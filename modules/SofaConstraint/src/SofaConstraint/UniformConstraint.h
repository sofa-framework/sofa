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
#include <sofa/core/behavior/Constraint.h>

namespace sofa::constraint
{

template < class DataTypes >
class UniformConstraint : public sofa::core::behavior::Constraint< DataTypes >
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UniformConstraint, DataTypes), SOFA_TEMPLATE(sofa::core::behavior::Constraint, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real  Real;

    typedef sofa::Data<VecCoord>    DataVecCoord;
    typedef sofa::Data<VecDeriv>    DataVecDeriv;
    typedef sofa::Data<MatrixDeriv> DataMatrixDeriv;

    void buildConstraintMatrix(const sofa::core::ConstraintParams* cParams, DataMatrixDeriv & c, unsigned int &cIndex, const DataVecCoord &x) override;

    void getConstraintViolation(const sofa::core::ConstraintParams* cParams, sofa::defaulttype::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &v) override;

    void getConstraintResolution(const sofa::core::ConstraintParams* cParams, std::vector<sofa::core::behavior::ConstraintResolution*>& crVector, unsigned int& offset) override;

    sofa::Data<bool> d_iterative;
    sofa::Data<bool> d_constraintRestPos;
protected:

    unsigned int m_constraintIndex;

    UniformConstraint();
};

} // namespace sofa::constraint
