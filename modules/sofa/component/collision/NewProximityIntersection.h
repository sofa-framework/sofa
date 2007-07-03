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
#ifndef SOFA_COMPONENT_COLLISION_NEWPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_NEWPROXIMITYINTERSECTION_H

#include <sofa/component/collision/DiscreteIntersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/RayModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

class NewProximityIntersection : public DiscreteIntersection
{
public:
    DataField<double> alarmDistance;
    DataField<double> contactDistance;

    NewProximityIntersection();

    virtual void init();

    /// Return the intersector class handling the given pair of collision models, or NULL if not supported.
    virtual core::componentmodel::collision::ElementIntersector* findIntersector(core::CollisionModel* object1, core::CollisionModel* object2);

    /// returns true if algorithm uses proximity
    virtual bool useProximity() const { return true; }

    /// Return the alarm distance (must return 0 if useNewProximity() is false)
    double getAlarmDistance() const { return alarmDistance.getValue(); }

    /// Return the contact distance (must return 0 if useNewProximity() is false)
    double getContactDistance() const { return contactDistance.getValue(); }

    void setAlarmDistance(double v) { alarmDistance.setValue(v); }

    void setContactDistance(double v) { contactDistance.setValue(v); }

    bool testIntersection(Cube& ,Cube&);
    bool testIntersection(Point&, Point&);
    bool testIntersection(Sphere&, Point&);
    bool testIntersection(Sphere&, Sphere&);
    bool testIntersection(Line&, Point&);
    bool testIntersection(Line&, Sphere&);
    bool testIntersection(Line&, Line&);
    bool testIntersection(Triangle&, Point&);
    bool testIntersection(Triangle&, Sphere&);
    bool testIntersection(Triangle&, Line&);
    bool testIntersection(Triangle&, Triangle&);
    bool testIntersection(Ray&, Triangle&);

    int computeIntersection(Cube&, Cube&, DetectionOutputVector&);
    int computeIntersection(Point&, Point&, DetectionOutputVector&);
    int computeIntersection(Sphere&, Point&, DetectionOutputVector&);
    int computeIntersection(Sphere&, Sphere&, DetectionOutputVector&);
    int computeIntersection(Line&, Point&, DetectionOutputVector&);
    int computeIntersection(Line&, Sphere&, DetectionOutputVector&);
    int computeIntersection(Line&, Line&, DetectionOutputVector&);
    int computeIntersection(Triangle&, Point&, DetectionOutputVector&);
    int computeIntersection(Triangle&, Sphere&, DetectionOutputVector&);
    int computeIntersection(Triangle&, Line&, DetectionOutputVector&);
    int computeIntersection(Triangle&, Triangle&, DetectionOutputVector&);
    int computeIntersection(Ray&, Triangle&, DetectionOutputVector&);

    static inline int doIntersectionTrianglePoint(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& n, const Vector3& q, DetectionOutputVector& contacts, bool swapElems = false);

    static inline int doIntersectionTrianglePoints(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& n, const Vector3& q1, const Vector3& q2, DetectionOutputVector& contacts, bool swapElems = false);

    static inline int doIntersectionTrianglePoints(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& n, const Vector3& q1, const Vector3& q2, const Vector3& q3, DetectionOutputVector& contacts, bool swapElems = false);

    static inline int doIntersectionLineLine(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& q1, const Vector3& q2, DetectionOutputVector& contacts);

    static inline int doIntersectionLinePoint(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& q, DetectionOutputVector& contacts, bool swapElems = false);

    static inline int doIntersectionPointPoint(double dist2, const Vector3& p, const Vector3& q, DetectionOutputVector& contacts);

protected:
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
