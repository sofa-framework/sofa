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
#ifndef SOFA_COMPONENT_COLLISION_MINPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_MINPROXIMITYINTERSECTION_H

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

class SOFA_BASE_COLLISION_API MinProximityIntersection : public DiscreteIntersection
{
public:
    SOFA_CLASS(MinProximityIntersection,DiscreteIntersection);
    Data<bool> useSphereTriangle;
    Data<bool> usePointPoint;
    Data<double> alarmDistance;
    Data<double> contactDistance;
protected:
    MinProximityIntersection();
public:

    typedef core::collision::IntersectorFactory<MinProximityIntersection> IntersectorFactory;

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
    bool testIntersection(Sphere&, Sphere&);
    //bool testIntersection(Ray&, Triangle&);

    int computeIntersection(Cube&, Cube&, OutputVector*);
    int computeIntersection(Sphere&, Sphere&, OutputVector*);
    //int computeIntersection(Ray&, Triangle&, OutputVector*);

    void draw(const core::visual::VisualParams* vparams);

private:
    double mainAlarmDistance;
    double mainContactDistance;
};

#if defined(WIN32) && !defined(SOFA_BUILD_BASE_COLLISION)
extern template class SOFA_BASE_COLLISION_API core::collision::IntersectorFactory<MinProximityIntersection>;
#endif


} // namespace collision

} // namespace component

} // namespace sofa

#endif
