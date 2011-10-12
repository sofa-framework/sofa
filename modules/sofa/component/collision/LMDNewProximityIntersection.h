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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_LMDNEWPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_LMDNEWPROXIMITYINTERSECTION_H

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

class SOFA_CONSTRAINT_API LMDNewProximityIntersection : public DiscreteIntersection
{
public:
    SOFA_CLASS(LMDNewProximityIntersection,DiscreteIntersection);

    Data<double> alarmDistance;
    Data<double> contactDistance;
    Data<bool> useLineLine;
protected:
    LMDNewProximityIntersection();
public:
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
    bool testIntersection(Point&, Point&);
    template<class Sphere>
    bool testIntersection(Sphere&, Point&);
    template<class Sphere>
    bool testIntersection(Sphere&, Sphere&);
    bool testIntersection(Line&, Point&);
    template<class Sphere>
    bool testIntersection(Line&, Sphere&);
    bool testIntersection(Line&, Line&);
    bool testIntersection(Triangle&, Point&);
    template<class Sphere>
    bool testIntersection(Triangle&, Sphere&);
    bool testIntersection(Triangle&, Line&);
    bool testIntersection(Triangle&, Triangle&);
    bool testIntersection(Ray&, Triangle&);

    int computeIntersection(Cube&, Cube&, OutputVector*);
    int computeIntersection(Point&, Point&, OutputVector*);
    template<class Sphere>
    int computeIntersection(Sphere&, Point&, OutputVector*);
    template<class Sphere>
    int computeIntersection(Sphere&, Sphere&, OutputVector*);
    int computeIntersection(Line&, Point&, OutputVector*);
    template<class Sphere>
    int computeIntersection(Line&, Sphere&, OutputVector*);
    int computeIntersection(Line&, Line&, OutputVector*);
    int computeIntersection(Triangle&, Point&, OutputVector*);
    template<class Sphere>
    int computeIntersection(Triangle&, Sphere&, OutputVector*);
    int computeIntersection(Triangle&, Line&, OutputVector*);
    int computeIntersection(Triangle&, Triangle&, OutputVector*);
    int computeIntersection(Ray&, Triangle&, OutputVector*);

    template< class TFilter1, class TFilter2 >
    static inline int doIntersectionLineLine(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& q1, const Vector3& q2, OutputVector* contacts, int id, int indexLine1, int indexLine2, TFilter1 &f1, TFilter2 &f2);

    template< class TFilter1, class TFilter2 >
    static inline int doIntersectionLinePoint(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& q, OutputVector* contacts, int id, int indexLine1, int indexPoint2, TFilter1 &f1, TFilter2 &f2, bool swapElems = false);

    template< class TFilter1, class TFilter2 >
    static inline int doIntersectionPointPoint(double dist2, const Vector3& p, const Vector3& q, OutputVector* contacts, int id, int indexPoint1, int indexPoint2, TFilter1 &f1, TFilter2 &f2);

    template< class TFilter1, class TFilter2 >
    static inline int doIntersectionTrianglePoint(double dist2, int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& n, const Vector3& q, OutputVector* contacts, int id, Triangle &e1, unsigned int *edgesIndices, int indexPoint2, TFilter1 &f1, TFilter2 &f2, bool swapElems = false);


//	static inline int doIntersectionLineLine(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& q1, const Vector3& q2, OutputVector* contacts, int id);

//	static inline int doIntersectionLinePoint(double dist2, const Vector3& p1, const Vector3& p2, const Vector3& q, OutputVector* contacts, int id, bool swapElems = false);

//	static inline int doIntersectionPointPoint(double dist2, const Vector3& p, const Vector3& q, OutputVector* contacts, int id);

//	static inline int doIntersectionTrianglePoint(double dist2, int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& n, const Vector3& q, OutputVector* contacts, int id, bool swapElems = false);


    /**
     * @brief Method called at the beginning of the collision detection between model1 and model2.
     * Checks if LMDFilters are associated to the CollisionModels.
     * @TODO Optimization.
     */
//	int beginIntersection(TriangleModel* /*model1*/, TriangleModel* /*model2*/, DiscreteIntersection::OutputVector* /*contacts*/);

protected:
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_COLLISION_LMDNEWPROXIMITYINTERSECTION_H
