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
#ifndef SOFA_CORE_BEHAVIOR_BASECONSTRAINTSET_H
#define SOFA_CORE_BEHAVIOR_BASECONSTRAINTSET_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/VecId.h>
#include <sofa/core/core.h>


namespace sofa
{

namespace core
{

namespace behavior
{

class SOFA_CORE_API BaseConstraintSet : public virtual objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseConstraintSet, objectmodel::BaseObject);

    /// Description of the order of the constraint
    enum ConstOrder {POS,VEL,ACC};


    BaseConstraintSet()
        : group(initData(&group, 0, "group", "ID of the group containing this constraint. This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle."))
    {
    }

    virtual ~BaseConstraintSet() { }

    virtual void resetConstraint() {};

    /**
     *  \brief Construct the Jacobian Matrix
     *
     *  \param constraintId is the index of the next constraint equation: when building the constraint matrix, you have to use this index, and then update it
     *  \param x is the state vector containing the positions used to determine the line of the Jacobian Matrix
     **/
    virtual void buildConstraintMatrix(unsigned int &constraintId, core::VecId x=core::VecId::position())=0;


    /// says if the constraint is holonomic or not
    /// holonomic constraints can be processed using different methods such as :
    /// projection - reducing the degrees of freedom - simple lagrange multiplier process
    /// Non-holonomic constraints (like contact, friction...) need more specific treatments
    virtual bool isHolonomic() {return false; }

    /// If the constraint is applied only on a subset of particles.
    /// That way, we can optimize the time spent traversing the mappings
    /// Deactivated by default. The constraints using only a subset of particles should activate the mask,
    /// and during projectResponse(), insert the indices of the particles modified
    virtual bool useMask() {return false;}
protected:
    Data<int> group;
};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
