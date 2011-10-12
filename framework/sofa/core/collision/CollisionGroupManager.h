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

protected:
    /// integration groups
    sofa::helper::vector<core::objectmodel::BaseContext*> groups;


    /// Destructor
    virtual ~CollisionGroupManager() { }
public:
    /// Create the integration groups
    virtual void createGroups(objectmodel::BaseContext* scene, const sofa::helper::vector<Contact*>& contacts) = 0;

    /// Clear de integration groups
    virtual void clearGroups(objectmodel::BaseContext* scene) = 0;

    /// Get de integration groups
    virtual const sofa::helper::vector<objectmodel::BaseContext*>& getGroups() { return groups; };

protected:

    std::map<Instance,sofa::helper::vector<core::objectmodel::BaseContext*> > storedGroups;

    virtual void changeInstance(Instance inst)
    {
        storedGroups[instance].swap(groups);
        groups.swap(storedGroups[inst]);
    }
};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
