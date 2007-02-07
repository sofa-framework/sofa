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
DetectionOutput* Intersection::intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
{
    ElementIntersector* i = findIntersector(elem1.getCollisionModel(), elem2.getCollisionModel());
    if (i == NULL)
        return NULL;
    else
        return i->intersect(elem1, elem2);
}

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

