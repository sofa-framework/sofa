/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
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
    typedef typename DataTypes::VecConst VecConst;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;

    Constraint(MechanicalState<DataTypes> *mm = NULL);

    virtual ~Constraint();

    Data<Real> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    virtual bool isActive() const; ///< if false, the constraint does nothing

    virtual void init();

    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes>* getMState() { return mstate; }

    /// @name Vector operations
    /// @{

    /// Project dx to constrained space (dx models an acceleration).
    ///
    /// This method retrieves the dx vector from the MechanicalState and call
    /// the internal projectResponse(VecDeriv&) method implemented by
    /// the component.
    virtual void projectResponse();

    /// Project v to constrained space (v models a velocity).
    ///
    /// This method retrieves the v vector from the MechanicalState and call
    /// the internal projectVelocity(VecDeriv&) method implemented by
    /// the component.
    virtual void projectVelocity();

    /// Project x to constrained space (x models a position).
    ///
    /// This method retrieves the x vector from the MechanicalState and call
    /// the internal projectPosition(VecCoord&) method implemented by
    /// the component.
    virtual void projectPosition();

    /// Project vFree to constrained space (vFree models a velocity).
    ///
    /// This method retrieves the vFree vector from the MechanicalState and call
    /// the internal projectVelocity(VecDeriv&) method implemented by
    /// the component.
    virtual void projectFreeVelocity();

    /// Project xFree to constrained space (xFree models a position).
    ///
    /// This method retrieves the xFree vector from the MechanicalState and call
    /// the internal projectPosition(VecCoord&) method implemented by
    /// the component.
    virtual void projectFreePosition();


    /// Project dx to constrained space (dx models an acceleration).
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic Constraint::projectResponse() method.
    virtual void projectResponse(VecDeriv& dx) = 0;
    /// This method must be implemented by the component to handle Lagrange Multiplier based constraint
    virtual void projectResponse(SparseVecDeriv& dx) = 0;

    /// Project v to constrained space (v models a velocity).
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic Constraint::projectVelocity() method.
    virtual void projectVelocity(VecDeriv& v) = 0;

    /// Project x to constrained space (x models a position).
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic Constraint::projectPosition() method.
    virtual void projectPosition(VecCoord& x) = 0;

    /// @}

    /// \todo What is the difference with BaseConstraint::applyConstraint(unsigned int&, double&) ?
    virtual void applyConstraint(unsigned int & contactId); // Pure virtual would be better

    virtual void applyConstraint(VecConst& /*c*/, unsigned int & /*contactId*/) {}

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const Constraint<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    MechanicalState<DataTypes> *mstate;
};

#if defined(WIN32) && !defined(SOFA_BUILD_CORE)
extern template class SOFA_CORE_API Constraint<defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Vec2dTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Vec1dTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Rigid2dTypes>;

extern template class SOFA_CORE_API Constraint<defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Vec2fTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Vec1fTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API Constraint<defaulttype::Rigid2fTypes>;
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
