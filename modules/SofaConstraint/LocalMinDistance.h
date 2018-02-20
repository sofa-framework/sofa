/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_H
#define SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_H
#include "config.h"

#include <SofaBaseCollision/BaseProximityIntersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaUserInteraction/RayModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_CONSTRAINT_API LocalMinDistance : public BaseProximityIntersection
{
public:
    SOFA_CLASS(LocalMinDistance,BaseProximityIntersection);

    typedef core::collision::IntersectorFactory<LocalMinDistance> IntersectorFactory;

    // Data<bool> useSphereTriangle;
    // Data<bool> usePointPoint;
    Data<bool> filterIntersection; ///< Activate LMD filter
    Data<double> angleCone; ///< Filtering cone extension angle
    Data<double> coneFactor; ///< Factor for filtering cone angle computation
    Data<bool> useLMDFilters; ///< Use external cone computation (Work in Progress)


protected:
    LocalMinDistance();
public:
    virtual void init() override;

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
    bool testValidity(Sphere&, const defaulttype::Vector3&) { return true; }
    bool testValidity(Point&, const defaulttype::Vector3&);
    bool testValidity(Line&, const defaulttype::Vector3&);
    bool testValidity(Triangle&, const defaulttype::Vector3&);

    void draw(const core::visual::VisualParams* vparams) override;

    /// Actions to accomplish when the broadPhase is started. By default do nothing.
    virtual void beginBroadPhase() override {}

    int beginIntersection(sofa::core::CollisionModel* /*model1*/, sofa::core::CollisionModel* /*model2*/, OutputVector* /*contacts*/)
    {
        return 0;
    }

private:
    double mainAlarmDistance;
    double mainContactDistance;
};

} // namespace collision

} // namespace component

namespace core
{
namespace collision
{
#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_CPP)
extern template class SOFA_CONSTRAINT_API IntersectorFactory<component::collision::LocalMinDistance>;
#endif
}
}

} // namespace sofa

#endif /* SOFA_COMPONENT_COLLISION_LOCALMINDISTANCE_H */
