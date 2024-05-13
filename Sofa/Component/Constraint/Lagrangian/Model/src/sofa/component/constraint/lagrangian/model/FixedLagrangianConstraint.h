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
#include <sofa/core/behavior/ConstraintResolution.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/topology/TopologySubsetIndices.h>

namespace sofa::component::constraint::lagrangian::model
{

template< class DataTypes >
class FixedLagrangianConstraint : public core::behavior::Constraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedLagrangianConstraint,DataTypes), SOFA_TEMPLATE(core::behavior::Constraint,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef typename core::behavior::Constraint<DataTypes> Inherit;

    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef core::objectmodel::Data<VecCoord>		DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>		DataVecDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

    typedef sofa::core::topology::TopologySubsetIndices SetIndex;

protected:

    sofa::type::vector<unsigned int> m_cid;
    sofa::type::vector<Deriv> m_prevForces;

    SetIndex d_indices;
    Data<bool> d_fixAll;

    FixedLagrangianConstraint(MechanicalState* object = nullptr);
    virtual ~FixedLagrangianConstraint() {}

    virtual type::vector<std::string> getConstraintIdentifiers() override final
    {
        type::vector<std::string> ids = getFixedIdentifiers();
        ids.push_back("Bilateral");
        return ids;
    }

    virtual type::vector<std::string> getFixedIdentifiers(){ return {}; }

public:
    void init() override;
    void buildConstraintMatrix(const core::ConstraintParams* cParams, DataMatrixDeriv &c_d, unsigned int &cIndex, const DataVecCoord &x) override;
    void getConstraintViolation(const core::ConstraintParams* cParams, linearalgebra::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &v) override;
    void getConstraintResolution(const core::ConstraintParams *, std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset) override;

private:
    void doBuildConstraintLine(helper::WriteAccessor<DataMatrixDeriv> &c, unsigned int lineNumber);
    void doGetSingleConstraintViolation(linearalgebra::BaseVector *resV, const DataVecCoord * freePos, const DataVecCoord * restPos,unsigned int lineNumber);
    void doGetSingleConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset, unsigned int lineNumber);

};

#if !defined(SOFA_COMPONENT_CONSTRAINTSET_FIXEDLAGRANGIANCONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint< defaulttype::Vec6Types >;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint< defaulttype::Vec3Types >;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint< defaulttype::Vec2Types >;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint< defaulttype::Vec1Types >;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API FixedLagrangianConstraint< defaulttype::Rigid3Types >;
#endif

} //namespace sofa::component::constraint::lagrangian::model
