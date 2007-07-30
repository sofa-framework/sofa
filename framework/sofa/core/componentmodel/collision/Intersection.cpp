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

ElementIntersector* IntersectorMap::get(core::CollisionModel* model1, core::CollisionModel* model2)
{
    iterator it =
        this->find(std::make_pair(TypeInfo(typeid(*model1)),TypeInfo(typeid(*model2))));
    if (it == this->end())
    {
        std::cerr << "ERROR: Element Intersector "
                << gettypename(typeid(*model1)) << "-"
                << gettypename(typeid(*model2)) << " NOT FOUND.\n";
        (*this)[std::make_pair(TypeInfo(typeid(*model1)),TypeInfo(typeid(*model2)))] = NULL;
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
/*
/// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present).
/// Note that this method is deprecated in favor of findIntersector
bool Intersection::canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
{
    ElementIntersector* i = findIntersector(elem1.getCollisionModel(), elem2.getCollisionModel());
    if (i == NULL)
        return false;
    else
        return i->canIntersect(elem1, elem2);
}

/// Compute the intersection between 2 elements.
/// Note that this method is deprecated in favor of findIntersector
int Intersection::intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, DetectionOutputVector& contacts)
{
    ElementIntersector* i = findIntersector(elem1.getCollisionModel(), elem2.getCollisionModel());
    if (i == NULL)
        return 0;
    else
        return i->intersect(elem1, elem2, contacts);
}
*/
} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

