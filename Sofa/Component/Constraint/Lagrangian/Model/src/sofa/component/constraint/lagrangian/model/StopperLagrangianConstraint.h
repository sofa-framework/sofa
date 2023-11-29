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

namespace sofa::component::constraint::lagrangian::model
{

class StopperLagrangianConstraintResolution1Dof : public core::behavior::ConstraintResolution
{
protected:
    double _invW, _w, _min, _max ;

public:

    StopperLagrangianConstraintResolution1Dof(const double &min, const double &max)
        : core::behavior::ConstraintResolution(1)
        , _min(min)
        , _max(max)
    { 
    }

    void init(int line, SReal** w, SReal*force) override
    {
        _w = w[line][line];
        _invW = 1.0/_w;
        force[line  ] = 0.0;
    }

    void resolution(int line, SReal** /*w*/, SReal* d, SReal* force, SReal*) override
    {
        const double dfree = d[line] - _w * force[line];

        if (dfree > _max)
            force[line] = (_max - dfree) * _invW;
        else if (dfree < _min)
            force[line] = (_min - dfree) * _invW;
        else
            force[line] = 0;
    }
};

template< class DataTypes >
class StopperLagrangianConstraint : public core::behavior::Constraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(StopperLagrangianConstraint,DataTypes), SOFA_TEMPLATE(core::behavior::Constraint,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename Coord::value_type Real;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef typename core::behavior::Constraint<DataTypes> Inherit;

    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef core::objectmodel::Data<VecCoord>		DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>		DataVecDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

protected:

    unsigned int cid;

    Data<int> index; ///< index of the stop constraint
    Data<SReal> min; ///< minimum value accepted
    Data<SReal> max; ///< maximum value accepted



    StopperLagrangianConstraint(MechanicalState* object = nullptr);

    virtual ~StopperLagrangianConstraint() {}


    virtual type::vector<std::string> getConstraintIdentifiers() override final
    {
        type::vector<std::string> ids = getStopperIdentifiers();
        ids.push_back("Stopper");
        ids.push_back("Unilateral");
        return ids;
    }

    virtual type::vector<std::string> getStopperIdentifiers(){ return {}; }



public:
    void init() override;
    void buildConstraintMatrix(const core::ConstraintParams* cParams, DataMatrixDeriv &c_d, unsigned int &cIndex, const DataVecCoord &x) override;
    void getConstraintViolation(const core::ConstraintParams* cParams, linearalgebra::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &v) override;

    void getConstraintResolution(const core::ConstraintParams *, std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset) override;
};

#if !defined(SOFA_COMPONENT_CONSTRAINTSET_STOPPERLAGRANGIANCONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API StopperLagrangianConstraint<defaulttype::Vec1Types>;

#endif

} //namespace sofa::component::constraint::lagrangian::model
