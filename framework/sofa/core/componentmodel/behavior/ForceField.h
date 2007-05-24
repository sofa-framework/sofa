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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_FORCEFIELD_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_FORCEFIELD_H

#include <sofa/core/componentmodel/behavior/BaseForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

/**
 *  \brief Component computing forces within a simulated body.
 */
template<class TDataTypes>
class ForceField : public BaseForceField
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    ForceField(MechanicalState<DataTypes> *mm = NULL);

    virtual ~ForceField();

    virtual void init();

    /// @name Vector operations
    /// @{

    virtual void addForce();

    virtual void addDForce();

    virtual double getPotentialEnergy();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v) = 0;

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx) = 0;

    virtual double getPotentialEnergy(const VecCoord& x) =0;

    /// @}

    /// @name Matrix operations
    /// @{

    virtual void contributeToMatrixDimension(unsigned int * const, unsigned int * const) {};

    virtual void computeMatrix(sofa::defaulttype::SofaBaseMatrix *, double , double , double, unsigned int &) {};

    virtual void computeVector(sofa::defaulttype::SofaBaseVector *, unsigned int & ) {};

    virtual void matResUpdatePosition(sofa::defaulttype::SofaBaseVector *, unsigned int & ) {};

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

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
