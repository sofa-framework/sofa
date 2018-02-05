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
#include <sofa/core/collision/Intersection.inl>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/helper/FnDispatcher.h>

namespace sofa
{

namespace core
{

namespace collision
{

using namespace sofa::defaulttype;

IntersectorMap::~IntersectorMap()
{
    for(InternalMap::const_iterator it = intersectorsMap.begin(), itEnd = intersectorsMap.end(); it != itEnd; ++it)
    {
        delete it->second;
    }
}

helper::TypeInfo IntersectorMap::getType(core::CollisionModel* model)
{
    helper::TypeInfo t(typeid(*model));
    const std::map<helper::TypeInfo,helper::TypeInfo>::iterator it = castMap.find(t);
    if (it == castMap.end())
    {
        helper::TypeInfo t2 = t;
        for (std::set<const objectmodel::ClassInfo* >::iterator it = classes.begin(); it != classes.end(); ++it)
        {
            if ((*it)->isInstance(model))
            {
                t2 = (*it)->type();
                break;
            }
        }
        castMap.insert(std::make_pair(t,t2));
        return t2;
    }
    else return it->second;
}

ElementIntersector* IntersectorMap::get(core::CollisionModel* model1, core::CollisionModel* model2, bool& swapModels)
{
    helper::TypeInfo t1 = getType(model1);
    helper::TypeInfo t2 = getType(model2);
    InternalMap::iterator it = intersectorsMap.find(std::make_pair(t1,t2));
    if (it != intersectorsMap.end())
    {
        swapModels = false;
        return it->second;
    }

    it = intersectorsMap.find(std::make_pair(t2,t1));
    if (it != intersectorsMap.end())
    {
        swapModels = true;
        return it->second;
    }


    std::stringstream tmp;
    for(InternalMap::const_iterator it = intersectorsMap.begin(), itEnd = intersectorsMap.end(); it != itEnd; ++it)
    {
        helper::TypeInfo t1 = it->first.first;
        helper::TypeInfo t2 = it->first.second;
        tmp << "  "
                << gettypename(t1) << "-"
                << gettypename(t2);
        ElementIntersector* i = it->second;
        if (!i)
            tmp << "  NULL";
        else
            tmp << "  " << i->name();
        tmp << msgendl;
    }
    tmp << msgendl;

    msg_warning("IntersectorMap")
            << "Element Intersector " << gettypename(t1) << "-" << gettypename(t2) << " NOT FOUND within :" << tmp.str() ;


    insert(t1, t2, 0);
    return 0;
}

void IntersectorMap::add_impl(const objectmodel::ClassInfo& c1,
        const objectmodel::ClassInfo& c2,
        ElementIntersector* intersector)
{
    classes.insert(&c1);
    classes.insert(&c2);
    castMap.clear();
    // rebuild castMap
    for (std::set<const objectmodel::ClassInfo* >::iterator it = classes.begin(); it != classes.end(); ++it)
    {
        castMap.insert(std::make_pair((*it)->type(),(*it)->type()));
    }

    insert(c1.type(), c2.type(), intersector);
}

void IntersectorMap::insert(const helper::TypeInfo& t1, const helper::TypeInfo& t2, ElementIntersector* intersector)
{
    const MapValue mapValue(MapKey(t1, t2), intersector);
    InternalMap::iterator it = intersectorsMap.find(mapValue.first);
    if(it != intersectorsMap.end())
    {
        delete it->second;
        it->second = mapValue.second;
    }
    else
    {
        intersectorsMap.insert(mapValue);
    }
}

Intersection::~Intersection()
{
}

/// Test if intersection between 2 types of elements is supported, i.e. an intersection test is implemented for this combinaison of types.
/// Note that this method is deprecated in favor of findIntersector
bool Intersection::isSupported(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
{
    bool swap;
    ElementIntersector* i = findIntersector(elem1.getCollisionModel(), elem2.getCollisionModel(), swap);
    return i != NULL;
}

} // namespace collision

} // namespace core

} // namespace sofa

