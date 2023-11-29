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
#include <sofa/component/constraint/lagrangian/model/config.h>

#include <sofa/core/behavior/Constraint.h>

namespace sofa::component::constraint::lagrangian::model
{

template < class DataTypes >
class UniformLagrangianConstraint : public sofa::core::behavior::Constraint< DataTypes >
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UniformLagrangianConstraint, DataTypes), SOFA_TEMPLATE(sofa::core::behavior::Constraint, DataTypes));

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

    void getConstraintViolation(const sofa::core::ConstraintParams* cParams, sofa::linearalgebra::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &v) override;

    void getConstraintResolution(const sofa::core::ConstraintParams* cParams, std::vector<sofa::core::behavior::ConstraintResolution*>& crVector, unsigned int& offset) override;

    sofa::Data<bool> d_iterative; ///< Iterate over the bilateral constraints, otherwise a block factorisation is computed.
    sofa::Data<bool> d_constraintRestPos; ///< if false, constrains the pos to be zero / if true constraint the current position to stay at rest position
protected:

    unsigned int m_constraintIndex;

    UniformLagrangianConstraint();

    virtual type::vector<std::string> getConstraintIdentifiers() override final
    {
        type::vector<std::string> ids = getStopperIdentifiers();
        ids.push_back("Uniform");
        ids.push_back("Bilateral");
        return ids;
    }

    virtual type::vector<std::string> getStopperIdentifiers(){ return {}; }

};

#if !defined(SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_UNIFORMLAGRANGIANCONSTRAINT_CPP)
    extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API UniformLagrangianConstraint<sofa::defaulttype::Vec1Types>;
#endif


} // namespace sofa::component::constraint::lagrangian::model
