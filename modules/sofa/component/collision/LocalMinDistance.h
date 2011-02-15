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
#ifndef SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_H
#define SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_H

#include <sofa/component/collision/DiscreteIntersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/BSplineModel.h>
#include <sofa/component/collision/RayModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_COMPONENT_COLLISION_API LocalMinDistance : public DiscreteIntersection
{
public:
    SOFA_CLASS(LocalMinDistance,DiscreteIntersection);

    // Data<bool> useSphereTriangle;
    // Data<bool> usePointPoint;
    Data<double> alarmDistance;
    Data<double> contactDistance;
    Data<bool> filterIntersection;
    Data<double> angleCone;
    Data<double> coneFactor;
    Data<bool> useLMDFilters;



    LocalMinDistance();

    virtual void init();

    /// returns true if algorithm uses proximity
    virtual bool useProximity() const { return true; }

    /// Return the alarm distance (must return 0 if useMinProximity() is false)
    double getAlarmDistance() const { return alarmDistance.getValue(); }

    /// Return the contact distance (must return 0 if useMinProximity() is false)
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
    bool testIntersection(Ray&, Sphere&);
    bool testIntersection(Ray&, Triangle&);

    int computeIntersection(Cube&, Cube&, OutputVector*);
    int computeIntersection(Point&, Point&, OutputVector*);
    int computeIntersection(Sphere&, Point&, OutputVector*);
    int computeIntersection(Sphere&, Sphere&, OutputVector*);
    int computeIntersection(Line&, Point&, OutputVector*);
    int computeIntersection(Line&, Sphere&, OutputVector*);
    int computeIntersection(Line&, Line&, OutputVector*);
    int computeIntersection(Triangle&, Point&, OutputVector*);
    int computeIntersection(Triangle&, Sphere&, OutputVector*);
    int computeIntersection(Ray&, Sphere&, OutputVector*);
    int computeIntersection(Ray&, Triangle&, OutputVector*);

    /// These methods check the validity of a found intersection.
    /// According to the local configuration around the found intersected primitive,
    /// we build a "Region Of Interest" geometric cone.
    /// Pertinent intersections have to belong to this cone, others are not taking into account anymore.
    bool testValidity(Point&, const Vector3&);
    bool testValidity(Line&, const Vector3&);
    bool testValidity(Triangle&, const Vector3&);

    //Copy of Line computation. TODO_Spline : finding adaptive and optimized computation for Spline
    bool testValidity(CubicBezierCurve&, const Vector3&);
    bool testIntersection(CubicBezierCurve&, Point&);
    int computeIntersection(CubicBezierCurve&, Point&, OutputVector*);
    bool testIntersection(CubicBezierCurve&, Sphere&);
    int computeIntersection(CubicBezierCurve&, Sphere&, OutputVector*);
    //bool testIntersection(CubicBezierCurve&, Sphere&);
    //bool testIntersection(CubicBezierCurve&, CubicBezierCurve&);


    void draw();

    /// Actions to accomplish when the broadPhase is started. By default do nothing.
    virtual void beginBroadPhase() {}

    int beginIntersection(sofa::core::CollisionModel* /*model1*/, sofa::core::CollisionModel* /*model2*/, OutputVector* /*contacts*/)
    {
        //std::cout << "beginIntersection\n";
        return 0;
    }

private:
    double mainAlarmDistance;
    double mainContactDistance;
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
