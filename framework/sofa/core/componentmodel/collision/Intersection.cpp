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
#include <sofa/core/componentmodel/collision/Intersection.inl>
#include <sofa/core/componentmodel/collision/DetectionOutput.h>
#include <sofa/helper/FnDispatcher.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

using namespace sofa::defaulttype;

helper::TypeInfo IntersectorMap::getType(core::CollisionModel* model)
{
    helper::TypeInfo t(typeid(*model));
    const std::map<helper::TypeInfo,helper::TypeInfo>::iterator it = castMap.find(t);
    if (it == castMap.end())
    {
        helper::TypeInfo t2 = t;
        for (std::set<const objectmodel::ClassInfo* >::iterator it = classes.begin(); it != classes.end(); ++it)
            if ((*it)->isInstance(model))
            {
                t2 = (*it)->type();
                break;
            }
        castMap.insert(std::make_pair(t,t2));
        return t2;
    }
    else return it->second;
}

ElementIntersector* IntersectorMap::get(core::CollisionModel* model1, core::CollisionModel* model2)
{
    helper::TypeInfo t1 = getType(model1);
    helper::TypeInfo t2 = getType(model2);
    iterator it =
        this->find(std::make_pair(t1,t2));
    if (it == this->end())
    {
        std::cerr << "ERROR: Element Intersector "
                << gettypename(t1) << "-"
                << gettypename(t2) << " NOT FOUND.\n";
        (*this)[std::make_pair(t1,t2)] = NULL;
        return NULL;
    }
    else
        return it->second;
}

Intersection::~Intersection()
{
}

/// Test if intersection between 2 types of elements is supported, i.e. an intersection test is implemented for this combinaison of types.
/// Note that this method is deprecated in favor of findIntersector
bool Intersection::isSupported(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
{
    ElementIntersector* i = findIntersector(elem1.getCollisionModel(), elem2.getCollisionModel());
    return i != NULL;
}

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

