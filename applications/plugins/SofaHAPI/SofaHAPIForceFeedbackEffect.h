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
#ifndef SOFAHAPI_SOFAHAPIFORCEFEEDBACKEFFECT_H
#define SOFAHAPI_SOFAHAPIFORCEFEEDBACKEFFECT_H

#include <sofa/SofaHAPI.h>

//HAPI include
#include <HAPI/HAPIForceEffect.h>
#include <H3DUtil/AutoRef.h>

#include <sofa/defaulttype/SolidTypes.h>
#include <SofaHaptics/ForceFeedback.h>

namespace SofaHAPI
{

using sofa::helper::vector;
using sofa::defaulttype::Vec3d;
using sofa::defaulttype::Quat;
typedef sofa::defaulttype::SolidTypes<double>::Transform Transform;
using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::BaseLink;
using sofa::core::objectmodel::SingleLink;
using sofa::core::behavior::MechanicalState;
using sofa::component::controller::ForceFeedback;

/// Data necessary to transform positions and forces between simulation and device space
class ForceFeedbackTransform
{
public:
    Transform endDevice_H_virtualTool;
    Transform world_H_baseDevice;
    double forceScale;
    double scale;
    ForceFeedbackTransform()
        : forceScale(1.0), scale(1.0)
    {
        endDevice_H_virtualTool.identity();
        world_H_baseDevice.identity();
    }
};

/// Implement HAPIForceEffect using a Sofa ForceFeedback component
class ForceFeedbackEffect : public HAPI::HAPIForceEffect
{
public:
    ForceFeedbackEffect(ForceFeedback* forceFeedback);
    ~ForceFeedbackEffect();

    virtual EffectOutput calculateForces( const EffectInput &input );

    void setTransform(const ForceFeedbackTransform& xf)
    {
        data = xf;
    }

    ForceFeedback* forceFeedback;
    ForceFeedbackTransform data;
    bool permanent_feedback;
};

/// Encapsulate ForceFeedbackEffect within a Sofa graph component
class SofaHAPIForceFeedbackEffect : public sofa::core::objectmodel::BaseObject
{
public:

    SOFA_CLASS(SofaHAPIForceFeedbackEffect, sofa::core::objectmodel::BaseObject);

    void setForceFeedback(ForceFeedback* ffb);
    ForceFeedback* getForceFeedback();
    int getIndice();

    ForceFeedbackEffect* getEffect();

    SingleLink<SofaHAPIForceFeedbackEffect,ForceFeedback,BaseLink::FLAG_STRONGLINK> forceFeedback;

protected:
    SofaHAPIForceFeedbackEffect();
    virtual ~SofaHAPIForceFeedbackEffect();

    H3DUtil::AutoRef<ForceFeedbackEffect> data;

public:
};

} // namespace SofaHAPI

#endif // SOFAHAPI_SOFAHAPIFORCEFEEDBACKEFFECT_H
