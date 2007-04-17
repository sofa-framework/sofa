/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_CONSTRAINT_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_CONSTRAINT_H

#include <sofa/core/componentmodel/behavior/BaseConstraint.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{


template<class DataTypes>
class Constraint : public BaseConstraint
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecConst VecConst;

    Constraint(MechanicalState<DataTypes> *mm = NULL);

    virtual ~Constraint();

    DataField<Real> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    bool isActive() const; ///< if false, the constraint does nothing

    virtual void init();

    virtual void projectResponse(); ///< project dx to constrained space
    virtual void projectVelocity(); ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(); ///< project x to constrained space (x models a position)

    virtual void projectResponse(VecDeriv& dx) = 0; ///< project dx to constrained space
    virtual void projectVelocity(VecDeriv& dx)=0; ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& x)=0; ///< project x to constrained space (x models a position)

    virtual void applyConstraint(unsigned int & /*contactId*/); // Pure virtual would be better
    virtual void applyConstraint(VecConst& /*c*/, unsigned int & /*contactId*/) {};

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

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
