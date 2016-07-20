/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "config.h"

#include <SofaBaseCollision/BaseProximityIntersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaBaseCollision/BaseIntTool.h>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_BASE_COLLISION_API MinProximityIntersection : public BaseProximityIntersection
{
public:
    SOFA_CLASS(MinProximityIntersection,BaseProximityIntersection);
    Data<bool> useSphereTriangle;
    Data<bool> usePointPoint;
    Data<bool> useSurfaceNormals;
    Data<bool> useLinePoint;
    Data<bool> useLineLine;
    Data<bool> useTriangleLine;

protected:
    MinProximityIntersection();
public:
    typedef core::collision::IntersectorFactory<MinProximityIntersection> IntersectorFactory;

    virtual void init();

    bool getUseSurfaceNormals();

    void draw(const core::visual::VisualParams* vparams);

private:
    SReal mainAlarmDistance;
    SReal mainContactDistance;
};

} // namespace collision

} // namespace component

namespace core
{
namespace collision
{
#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_MINPROXIMITYINTERSECTION_CPP)
extern template class SOFA_BASE_COLLISION_API IntersectorFactory<component::collision::MinProximityIntersection>;
#endif
}
}

} // namespace sofa

#endif
