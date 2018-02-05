/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_BEHAVIOR_CONSTRAINT_H
#define SOFA_CORE_BEHAVIOR_CONSTRAINT_H

#include <sofa/core/core.h>
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief Component computing constraints within a simulated body.
 *
 *  This class define the abstract API common to constraints using a given type
 *  of DOFs.
 *  A Constraint computes constraints applied to one simulated body given its
 *  current position and velocity.
 *
 */
template<class DataTypes>
class Constraint : public BaseConstraint
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(Constraint,DataTypes), BaseConstraint);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;

    typedef core::objectmodel::Data<VecCoord>		DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>		DataVecDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;
protected:
    Constraint(MechanicalState<DataTypes> *mm = NULL);

    virtual ~Constraint();
public:
    Data<Real> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    virtual bool isActive() const; ///< if false, the constraint does nothing

    virtual void init() override;

    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes>* getMState() { return mstate; }

    using BaseConstraintSet::getConstraintViolation;

    /// Construct the Constraint violations vector of each constraint
    ///
    /// \param v is the result vector that contains the whole constraints violations
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    virtual void getConstraintViolation(const ConstraintParams* cParams, defaulttype::BaseVector *v) override;

    /// Construct the Constraint violations vector of each constraint
    ///
    /// \param resV is the result vector that contains the whole constraints violations
    /// \param x is the position vector used to compute contraint position violation
    /// \param v is the velocity vector used to compute contraint velocity violation
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    ///
    /// This is the method that should be implemented by the component
    virtual void getConstraintViolation(const ConstraintParams* cParams, defaulttype::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &v) = 0;


    /// Construct the Jacobian Matrix
    ///
    /// \param cId is the result constraint sparse matrix Id
    /// \param cIndex is the index of the next constraint equation: when building the constraint matrix, you have to use this index, and then update it
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    virtual void buildConstraintMatrix(const ConstraintParams* cParams, MultiMatrixDerivId cId, unsigned int &cIndex) override;

    /// Construct the Jacobian Matrix
    ///
    /// \param c is the result constraint sparse matrix
    /// \param cIndex is the index of the next constraint equation: when building the constraint matrix, you have to use this index, and then update it
    /// \param x is the position vector used for contraint equation computation
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    ///
    /// This is the method that should be implemented by the component
    virtual void buildConstraintMatrix(const ConstraintParams* cParams, DataMatrixDeriv & c, unsigned int &cIndex, const DataVecCoord &x) = 0;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const Constraint<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    MechanicalState<DataTypes> *mstate;

    /// Useful when the Constraint is applied only on a subset of dofs.
    /// It is automatically called by buildConstraintMatrix
    ///
    /// That way, we can optimize the time spent to transfer quantities through the mechanical mappings.
    /// Every Dofs are inserted by default. The Constraint using only a subset of dofs should only insert these dofs in the mask.
    virtual void updateForceMask() override;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_CORE_BEHAVIOR_CONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API Constraint<defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Vec2dTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Vec1dTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API Constraint<defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Vec2fTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Vec1fTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Rigid2fTypes>;
#endif
#endif
} // namespace behavior

} // namespace core

} // namespace sofa

#endif
