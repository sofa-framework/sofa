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
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/SingleStateAccessor.h>

namespace sofa::core::behavior
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
class ProjectiveConstraintSet : public BaseProjectiveConstraintSet, public SingleStateAccessor<DataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(ProjectiveConstraintSet,DataTypes), BaseProjectiveConstraintSet, SOFA_TEMPLATE(SingleStateAccessor, DataTypes));

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
    ProjectiveConstraintSet(MechanicalState<DataTypes> *mm = nullptr);

    ~ProjectiveConstraintSet() override;
public:

    // to get rid of warnings
    using BaseProjectiveConstraintSet::projectResponse;



    Data<Real> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    virtual bool isActive() const; ///< if false, the constraint does nothing

    virtual type::vector< core::BaseState* > getModels() override
    {
        return { this->getMState() };
    }

    /// @name Vector operations
    /// @{

    /// Project dx to constrained space (dx models an acceleration).
    ///
    /// This method retrieves the dxId vector from the MechanicalState and call
    /// the internal projectResponse(VecDeriv&) method implemented by
    /// the component.
    void projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId) override;

    /// Project the L matrix of the Lagrange Multiplier equation system.
    ///
    /// This method retrieves the lines of the Jacobian Matrix from the MechanicalState and call
    /// the internal projectResponse(MatrixDeriv&) method implemented by
    /// the component.
    void projectJacobianMatrix(const MechanicalParams* mparams, MultiMatrixDerivId cId) override;

    /// Project v to constrained space (v models a velocity).
    ///
    /// This method retrieves the vId vector from the MechanicalState and call
    /// the internal projectVelocity(VecDeriv&) method implemented by
    /// the component.
    void projectVelocity(const MechanicalParams* mparams, MultiVecDerivId vId) override;

    /// Project x to constrained space (x models a position).
    ///
    /// This method retrieves the xId vector from the MechanicalState and call
    /// the internal projectPosition(VecCoord&) method implemented by
    /// the component.
    void projectPosition(const MechanicalParams* mparams, MultiVecCoordId xId) override;



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
    void applyConstraint(const MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/) override
    {
        msg_error() << "applyConstraint(mparams, matrix) not implemented.";
    }


    /// Project the global Mechanical Vector to constrained space using offset parameter
    void applyConstraint(const MechanicalParams* /*mparams*/, linearalgebra::BaseVector* /*vector*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/) override
    {
        msg_error() << "applyConstraint(mparams, vector, matrix) not implemented.";
    }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<MechanicalState<DataTypes>*>(context->getMechanicalState()) == nullptr){
            arg->logError("No mechanical state with the datatype '" + std::string(DataTypes::Name()) + "' found in the context node.");
            return false;
        }

        return BaseObject::canCreate(obj, context, arg);
    }
};

#if !defined(SOFA_CORE_BEHAVIOR_PROJECTIVECONSTRAINTSET_CPP)
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec6Types >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec3Types >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec2Types >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec1Types >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Rigid3Types >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Rigid2Types >;


#endif
} // namespace sofa::core::behavior
