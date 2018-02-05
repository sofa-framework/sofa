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
#ifndef SOFA_CORE_COLLISION_INTERSECTION_H
#define SOFA_CORE_COLLISION_INTERSECTION_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/helper/FnDispatcher.h>

namespace sofa
{

namespace core
{

namespace collision
{

class BaseIntersector
{
public:

    BaseIntersector() {}

    ~BaseIntersector() {}

    template<class Model1, class Model2>
    sofa::core::collision::TDetectionOutputVector<Model1,Model2>* createOutputVector(Model1*, Model2*)
    {
        return new sofa::core::collision::TDetectionOutputVector<Model1,Model2>;
    }

    template<class Model1, class Model2>
    sofa::core::collision::TDetectionOutputVector<Model1,Model2>* getOutputVector(Model1*, Model2*, sofa::core::collision::DetectionOutputVector* contacts)
    {
        return static_cast<sofa::core::collision::TDetectionOutputVector<Model1,Model2>*>(contacts);
    }

    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    int beginIntersection(sofa::core::CollisionModel* /*model1*/, sofa::core::CollisionModel* /*model2*/, OutputVector* /*contacts*/)
    {
        return 0;
    }

    int endIntersection(sofa::core::CollisionModel* /*model1*/, sofa::core::CollisionModel* /*model2*/, OutputVector* /*contacts*/)
    {
        return 0;
    }

};

class ElementIntersector
{
public:
    virtual ~ElementIntersector() {}

    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    virtual bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2) = 0;

    /// Begin intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    /// If the given contacts vector is NULL, then this method should allocate it.
    virtual int beginIntersect(core::CollisionModel* model1, core::CollisionModel* model2, DetectionOutputVector*& contacts) = 0;

    /// Compute the intersection between 2 elements. Return the number of contacts written in the contacts vector.
    virtual int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, DetectionOutputVector* contacts) = 0;

    /// End intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    virtual int endIntersect(core::CollisionModel* model1, core::CollisionModel* model2, DetectionOutputVector* contacts) = 0;

    virtual std::string name() const = 0;
};

/// Table storing associations between types of collision models and intersectors implementing intersection tests
///
/// This class uses the new ClassInfo metaclass to be able to recognize derived classes. So it is no longer necessary
/// to register all derived collision models (i.e. an intersector registered for RayModel will also be used for RayPickIntersector).
class SOFA_CORE_API IntersectorMap
{
    typedef std::map<std::pair<helper::TypeInfo, helper::TypeInfo>, ElementIntersector*> InternalMap;
    typedef InternalMap::key_type MapKey;
    typedef InternalMap::value_type MapValue;
public:

    template<class Model1, class Model2, class T>
    void add(T* ptr);

    template<class Model1, class Model2>
    void ignore();

    ~IntersectorMap();

    helper::TypeInfo getType(core::CollisionModel* model);

    ElementIntersector* get(core::CollisionModel* model1, core::CollisionModel* model2, bool& swapModels);

protected:
    template<class Model1, class Model2>
    void add_impl(ElementIntersector* intersector);

    void add_impl(const objectmodel::ClassInfo& c1, const objectmodel::ClassInfo& c2, ElementIntersector* intersector);

    void insert(const helper::TypeInfo& t1, const helper::TypeInfo& t2, ElementIntersector* intersector);

    InternalMap intersectorsMap;
    std::map< helper::TypeInfo, helper::TypeInfo > castMap;
    std::set< const objectmodel::ClassInfo* > classes;
};

/** @brief Given 2 collision elements, test if an intersection is possible (for bounding volumes), or compute intersection points if any
*/

class SOFA_CORE_API Intersection : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(Intersection, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(Intersection)
protected:
    Intersection() {}
    virtual ~Intersection();
	
private:
	Intersection(const Intersection& n) ;
	Intersection& operator=(const Intersection& n) ;
	
public:
    /// Return the intersector class handling the given pair of collision models, or NULL if not supported.
    /// @param swapModels output value set to true if the collision models must be swapped before calling the intersector.
    virtual ElementIntersector* findIntersector(core::CollisionModel* object1, core::CollisionModel* object2, bool& swapModels) = 0;

    /// Test if intersection between 2 types of elements is supported, i.e. an intersection test is implemented for this combinaison of types.
    /// Note that this method is deprecated in favor of findIntersector
    virtual bool isSupported(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2);

    /// returns true if algorithm uses proximity detection
    virtual bool useProximity() const { return false; }

    /// returns true if algorithm uses continous detection
    virtual bool useContinuous() const { return false; }

    /// Return the alarm distance (must return 0 if useProximity() is false)
    virtual SReal getAlarmDistance() const { return (SReal)0.0; }

    /// Return the contact distance (must return 0 if useProximity() is false)
    virtual SReal getContactDistance() const { return (SReal)0.0; }

    /// Set the alarm distance (must return 0 if useProximity() is false)
    virtual void setAlarmDistance(SReal) { return; }

    /// Set the contact distance (must return 0 if useProximity() is false)
    virtual void setContactDistance(SReal) { return; }


    /// Actions to accomplish when the broadPhase is started. By default do nothing.
    virtual void beginBroadPhase()
    {
    }

    /// Actions to accomplish when the broadPhase is finished. By default do nothing.
    virtual void endBroadPhase()
    {
    }

    /// Actions to accomplish when the narrow Phase is started. By default do nothing.
    virtual void beginNarrowPhase()
    {
    }

    /// Actions to accomplish when the narrow Phase is finished. By default do nothing.
    virtual void endNarrowPhase()
    {
    }


};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
