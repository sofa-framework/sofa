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
#ifndef SOFA_CORE_BEHAVIORMODEL_H
#define SOFA_CORE_BEHAVIORMODEL_H

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace core
{

/**
 *  \brief Abstract Interface of components defining the behavior of a simulated object.
 *
 *  This Interface is used by "black-box" objects (such as some fluid algorithms)
 *  that are present in a SOFA simulation but which do not use the internal
 *  behavior components (MechanicalState, ForceField, etc) defined in the
 *  sofa::core::behavior namespace.
 *
 *  A BehaviorModel simply has to implement the updatePosition method
 *  to compute a new simulation step.
 *
 */
class SOFA_CORE_API BehaviorModel : public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BehaviorModel, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BehaviorModel)
protected:
    BehaviorModel() {}
    /// Destructor
    virtual ~BehaviorModel() {}
	
private:
	BehaviorModel(const BehaviorModel& n) ;
	BehaviorModel& operator=(const BehaviorModel& n) ;
	
public:
    /// Computation of a new simulation step.
    virtual void updatePosition(SReal dt) = 0;

    virtual bool addBBox(SReal* /*minBBox*/, SReal* /*maxBBox*/)
    {
        return false;
    }

    virtual bool insertInNode( objectmodel::BaseNode* node ) override;
    virtual bool removeInNode( objectmodel::BaseNode* node ) override;

};

} // namespace core

} // namespace sofa

#endif
