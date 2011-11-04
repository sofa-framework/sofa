/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_NEWPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_NEWPROXIMITYINTERSECTION_H

#include <sofa/component/collision/DiscreteIntersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/CubeModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_BASE_COLLISION_API NewProximityIntersection : public DiscreteIntersection
{
public:
    SOFA_CLASS(NewProximityIntersection,DiscreteIntersection);

    Data<double> alarmDistance;
    Data<double> contactDistance;
    Data<bool> useLineLine;
protected:
    NewProximityIntersection();
public:

    typedef core::collision::IntersectorFactory<NewProximityIntersection> IntersectorFactory;

    virtual void init();

    /// Returns true if algorithm uses proximity
    virtual bool useProximity() const { return true; }

    /// Returns the alarm distance (must returns 0 if useProximity() is false)
    double getAlarmDistance() const { return alarmDistance.getValue(); }

    /// Returns the contact distance (must returns 0 if useProximity() is false)
    double getContactDistance() const { return contactDistance.getValue(); }

    /// Sets the alarm distance (if useProximity() is false, the alarm distance is equal to 0)
    void setAlarmDistance(double v) { alarmDistance.setValue(v); }

    /// Sets the contact distance (if useProximity() is false, the contact distance is equal to 0)
    void setContactDistance(double v) { contactDistance.setValue(v); }

    bool testIntersection(Cube& ,Cube&);
    template<class Sphere>
    bool testIntersection(Sphere&, Sphere&);

    int computeIntersection(Cube&, Cube&, OutputVector*);
    template<class Sphere>
    int computeIntersection(Sphere&, Sphere&, OutputVector*);

    static inline int doIntersectionPointPoint(double dist2, const Vector3& p, const Vector3& q, OutputVector* contacts, int id);

};

#if defined(WIN32) && !defined(SOFA_BUILD_BASE_COLLISION)
extern template class SOFA_BASE_COLLISION_API core::collision::IntersectorFactory<NewProximityIntersection>;
#endif


} // namespace collision

} // namespace component

} // namespace sofa

#endif
