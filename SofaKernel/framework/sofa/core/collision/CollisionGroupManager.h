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
#ifndef SOFA_CORE_COLLISION_COLLISIONGROUPMANAGER_H
#define SOFA_CORE_COLLISION_COLLISIONGROUPMANAGER_H

#include <sofa/core/collision/CollisionAlgorithm.h>
#include <sofa/core/collision/Contact.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace collision
{

//class Scene;
/**
 * @brief Given a set of contacts, create integration groups
 *
 * For each pair of objects in contacts :
 *
 * - Look which mechanical integration algorithm is used
 *
 * - If they are “compatible”, create a algorithm merging them
 *
 *   -# Often simply the most stable of the two
 *
 *	Explicit Euler + Explicit Runge Kutta -> Explicit Runge Kutta
 *
 *	Explicit * + Implicit Euler -> Implicit Euler
 *
 *
 */
class CollisionGroupManager : public virtual CollisionAlgorithm
{
public:
    SOFA_ABSTRACT_CLASS(CollisionGroupManager, CollisionAlgorithm);
    SOFA_BASE_CAST_IMPLEMENTATION(CollisionGroupManager)

protected:
    /// integration groups
    sofa::helper::vector<core::objectmodel::BaseContext::SPtr> groups;


    /// Destructor
    virtual ~CollisionGroupManager() { }
public:
    /// Create the integration groups
    virtual void createGroups(objectmodel::BaseContext* scene, const sofa::helper::vector<Contact::SPtr>& contacts) = 0;

    /// Clear the integration groups
    virtual void clearGroups(objectmodel::BaseContext* scene) = 0;

    /// Get the integration groups
    virtual const sofa::helper::vector<objectmodel::BaseContext::SPtr>& getGroups() { return groups; };

protected:

    std::map<Instance,sofa::helper::vector<core::objectmodel::BaseContext::SPtr> > storedGroups;

    virtual void changeInstance(Instance inst) override
    {
        storedGroups[instance].swap(groups);
        groups.swap(storedGroups[inst]);
    }
};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
