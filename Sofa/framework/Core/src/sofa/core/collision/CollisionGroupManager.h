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

#include <sofa/core/collision/CollisionAlgorithm.h>
#include <sofa/core/collision/Contact.h>

#include <vector>

namespace sofa::core::collision
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
    sofa::type::vector<core::objectmodel::BaseContext::SPtr> groups;


    /// Destructor
    ~CollisionGroupManager() override { }
public:
    /// Create the integration groups
    virtual void createGroups(objectmodel::BaseContext* scene, const sofa::type::vector<Contact::SPtr>& contacts) = 0;

    /// Clear the integration groups
    virtual void clearGroups(objectmodel::BaseContext* scene) = 0;

    /// Get the integration groups
    virtual const sofa::type::vector<objectmodel::BaseContext::SPtr>& getGroups() { return groups; };

protected:

    std::map<Instance,sofa::type::vector<core::objectmodel::BaseContext::SPtr> > storedGroups;

    void changeInstance(Instance inst) override
    {
        storedGroups[instance].swap(groups);
        groups.swap(storedGroups[inst]);
    }
};
} // namespace sofa::core::collision
