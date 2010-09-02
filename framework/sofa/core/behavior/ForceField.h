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
#ifndef SOFA_CORE_BEHAVIOR_FORCEFIELD_H
#define SOFA_CORE_BEHAVIOR_FORCEFIELD_H

#include <sofa/core/core.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief Component computing forces within a simulated body.
 *
 *  This class define the abstract API common to force fields using a
 *  given type of DOFs.
 *  A force field computes forces applied to one simulated body
 *  given its current position and velocity.
 *
 *  For implicit integration schemes, it must also compute the derivative
 *  ( df, given a displacement dx ).
 */
template<class TDataTypes>
class ForceField : public BaseForceField
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ForceField, TDataTypes), BaseForceField);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    ForceField(MechanicalState<DataTypes> *mm = NULL);

    virtual ~ForceField();

    virtual void init();

    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes>* getMState() { return mstate; }

    /// @name Vector operations
    /// @{

    /// Given the current position and velocity states, update the current force
    /// vector by computing and adding the forces associated with this
    /// ForceField.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ f += B v + K x $
    ///
    /// This method retrieves the force, x and v vector from the MechanicalState
    /// and call the internal addForce(VecDeriv&,const VecCoord&,const VecDeriv&)
    /// method implemented by the component.
    virtual void addForce();

    /// Compute the force derivative given a small displacement from the
    /// position and velocity used in the previous call to addForce().
    ///
    /// The derivative should be directly derived from the computations
    /// done by addForce. Any forces neglected in addDForce will be integrated
    /// explicitly (i.e. using its value at the beginning of the timestep).
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ df += kFactor K dx + bFactor B dx $
    ///
    /// This method retrieves the force and dx vector from the MechanicalState
    /// and call the internal addDForce(VecDeriv&,const VecDeriv&,double,double)
    /// method implemented by the component.
    virtual void addDForce(double kFactor, double bFactor);

    /// Same as addDForce(), except the velocity vector should be used instead of dx.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ df += kFactor K v + bFactor B v $
    ///
    /// This method retrieves the force and velocity vector from the MechanicalState
    /// and call the internal addDForce(VecDeriv&,const VecDeriv&,double,double)
    /// method implemented by the component.
    virtual void addDForceV(double kFactor, double bFactor);

    /// Get the potential energy associated to this ForceField.
    ///
    /// Used to extimate the total energy of the system by some
    /// post-stabilization techniques.
    ///
    /// This method retrieves the x vector from the MechanicalState and call
    /// the internal getPotentialEnergy(const VecCoord&) method implemented by
    /// the component.
    virtual double getPotentialEnergy() const;

    /// Given the current position and velocity states, update the current force
    /// vector by computing and adding the forces associated with this
    /// ForceField.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ f += B v + K x $
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ForceField::addForce() method.
    virtual void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v) = 0;

    /// Compute the force derivative given a small displacement from the
    /// position and velocity used in the previous call to addForce().
    ///
    /// The derivative should be directly derived from the computations
    /// done by addForce. Any forces neglected in addDForce will be integrated
    /// explicitly (i.e. using its value at the beginning of the timestep).
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ df += K dx $
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ForceField::addDForce() method.
    ///
    /// @deprecated to more efficiently accumulate contributions from all terms
    ///   of the system equation, a new addDForce method allowing to pass two
    ///   coefficients for the stiffness and damping terms should now be used.
    virtual void addDForce(VecDeriv& df, const VecDeriv& dx);

    /// Compute the force derivative given a small displacement from the
    /// position and velocity used in the previous call to addForce().
    ///
    /// The derivative should be directly derived from the computations
    /// done by addForce. Any forces neglected in addDForce will be integrated
    /// explicitly (i.e. using its value at the beginning of the timestep).
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ df += kFactor K dx + bFactor B dx $
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ForceField::addDForce() method.
    ///
    /// To support old components that implement the deprecated addForce method
    /// without scalar coefficients, it defaults to using a temporaty vector to
    /// compute $ K dx $ and then manually scaling all values by kFactor.
    virtual void addDForce(VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);

    /// Get the potential energy associated to this ForceField.
    ///
    /// Used to extimate the total energy of the system by some
    /// post-stabilization techniques.
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ForceField::getPotentialEnergy() method.
    virtual double getPotentialEnergy(const VecCoord& x) const =0;

    /// @}

    /// @name Matrix operations
    /// @{

    /// @deprecated
    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * matrix, double kFact, unsigned int &offset);

    virtual void addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double kFact)
    {
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
        if (r)
            addKToMatrix(r.matrix, kFact, r.offset);
    }

    /// @deprecated
    virtual void addBToMatrix(sofa::defaulttype::BaseMatrix * matrix, double bFact, unsigned int &offset);

    virtual void addBToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double bFact)
    {
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
        if (r)
            addBToMatrix(r.matrix, bFact, r.offset);
    }

    /// @}

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

    static std::string templateName(const ForceField<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    MechanicalState<DataTypes> *mstate;
};

#if defined(WIN32) && !defined(SOFA_BUILD_CORE)
extern template class SOFA_CORE_API ForceField<defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API ForceField<defaulttype::Vec2dTypes>;
extern template class SOFA_CORE_API ForceField<defaulttype::Vec1dTypes>;
extern template class SOFA_CORE_API ForceField<defaulttype::Vec6dTypes>;
extern template class SOFA_CORE_API ForceField<defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API ForceField<defaulttype::Rigid2dTypes>;

extern template class SOFA_CORE_API ForceField<defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API ForceField<defaulttype::Vec2fTypes>;
extern template class SOFA_CORE_API ForceField<defaulttype::Vec1fTypes>;
extern template class SOFA_CORE_API ForceField<defaulttype::Vec6fTypes>;
extern template class SOFA_CORE_API ForceField<defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API ForceField<defaulttype::Rigid2fTypes>;
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
