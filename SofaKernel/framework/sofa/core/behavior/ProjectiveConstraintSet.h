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
#ifndef SOFA_CORE_BEHAVIOR_PROJECTIVECONSTRAINTSET_H
#define SOFA_CORE_BEHAVIOR_PROJECTIVECONSTRAINTSET_H

#include <sofa/core/core.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>

#include <sofa/defaulttype/VecTypes.h>
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
 *  A ProjectiveConstraintSet computes constraints applied to one simulated body given its
 *  current position and velocity.
 *
 */
template<class DataTypes>
class ProjectiveConstraintSet : public BaseProjectiveConstraintSet
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ProjectiveConstraintSet,DataTypes), BaseProjectiveConstraintSet);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
protected:
    ProjectiveConstraintSet(MechanicalState<DataTypes> *mm = NULL);

    virtual ~ProjectiveConstraintSet();
public:

    // to get rid of warnings
    using BaseProjectiveConstraintSet::projectResponse;



    Data<Real> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    virtual bool isActive() const; ///< if false, the constraint does nothing

    virtual void init() override;

    
    virtual helper::vector< core::BaseState* > getModels() override
    {
        helper::vector< core::BaseState* > models;
        models.push_back( getMState() );
        return models;
    }


    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes>* getMState() { return mstate.get(); }

    /// @name Vector operations
    /// @{

    /// Project dx to constrained space (dx models an acceleration).
    ///
    /// This method retrieves the dxId vector from the MechanicalState and call
    /// the internal projectResponse(VecDeriv&) method implemented by
    /// the component.
    virtual void projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId) override;

    /// Project the L matrix of the Lagrange Multiplier equation system.
    ///
    /// This method retrieves the lines of the Jacobian Matrix from the MechanicalState and call
    /// the internal projectResponse(MatrixDeriv&) method implemented by
    /// the component.
    virtual void projectJacobianMatrix(const MechanicalParams* mparams, MultiMatrixDerivId cId) override;

    /// Project v to constrained space (v models a velocity).
    ///
    /// This method retrieves the vId vector from the MechanicalState and call
    /// the internal projectVelocity(VecDeriv&) method implemented by
    /// the component.
    virtual void projectVelocity(const MechanicalParams* mparams, MultiVecDerivId vId) override;

    /// Project x to constrained space (x models a position).
    ///
    /// This method retrieves the xId vector from the MechanicalState and call
    /// the internal projectPosition(VecCoord&) method implemented by
    /// the component.
    virtual void projectPosition(const MechanicalParams* mparams, MultiVecCoordId xId) override;



    /// Project dx to constrained space (dx models an acceleration).
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ProjectiveConstraintSet::projectResponse() method.
    virtual void projectResponse(const MechanicalParams* mparams, DataVecDeriv& dx) = 0;

    /// Project v to constrained space (v models a velocity).
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ProjectiveConstraintSet::projectVelocity() method.
    virtual void projectVelocity(const MechanicalParams* mparams, DataVecDeriv& v) = 0;
    /// Project x to constrained space (x models a position).
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ProjectiveConstraintSet::projectPosition() method.
    virtual void projectPosition(const MechanicalParams* mparams, DataVecCoord& x) = 0;

    /// Project c to constrained space (c models a constraint).
    ///
    /// This method must be implemented by the component to handle Lagrange Multiplier based constraint
    virtual void projectJacobianMatrix(const MechanicalParams* /*mparams*/, DataMatrixDeriv& cData) = 0;

    /// @}


    /// Project the global Mechanical Matrix to constrained space using offset parameter
    virtual void applyConstraint(const MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/) override
    {
        serr << "applyConstraint(mparams, matrix) not implemented" << sendl;
    }


    /// Project the global Mechanical Vector to constrained space using offset parameter
    virtual void applyConstraint(const MechanicalParams* /*mparams*/, defaulttype::BaseVector* /*vector*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/) override
    {
        serr << "applyConstraint(mparams, vector, matrix) not implemented" << sendl;
    }

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

    static std::string templateName(const ProjectiveConstraintSet<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    SingleLink<ProjectiveConstraintSet<DataTypes>,MechanicalState<DataTypes>,BaseLink::FLAG_STRONGLINK> mstate;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_CORE_BEHAVIOR_PROJECTIVECONSTRAINTSET_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec6dTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Rigid2dTypes >;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec6fTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Rigid2fTypes >;
#endif
#endif
} // namespace behavior

} // namespace core

} // namespace sofa

#endif
