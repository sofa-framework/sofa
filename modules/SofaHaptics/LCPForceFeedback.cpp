/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_CPP

#include <SofaHaptics/LCPForceFeedback.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>
/*
namespace
{

#ifndef SOFA_FLOAT
using sofa::defaulttype::Rigid3dTypes;

template<>
bool derivVectors<Rigid3dTypes>(const Rigid3dTypes::VecCoord& x0, const Rigid3dTypes::VecCoord& x1, Rigid3dTypes::VecDeriv& d, bool derivRotation)
{
    return derivRigid3Vectors<Rigid3dTypes>(x0, x1, d, derivRotation);
}

template <>
double computeDot<Rigid3dTypes>(const Rigid3dTypes::Deriv& v0, const Rigid3dTypes::Deriv& v1)
{
    return dot(getVCenter(v0),getVCenter(v1)) + dot(getVOrientation(v0), getVOrientation(v1));
}

#endif

#ifndef SOFA_DOUBLE
using sofa::defaulttype::Rigid3fTypes;

template<>
bool derivVectors<Rigid3fTypes>(const Rigid3fTypes::VecCoord& x0, const Rigid3fTypes::VecCoord& x1, Rigid3fTypes::VecDeriv& d, bool derivRotation)
{
    return derivRigid3Vectors<Rigid3fTypes>(x0, x1, d, derivRotation);
}

template <>
double computeDot<Rigid3fTypes>(const Rigid3fTypes::Deriv& v0, const Rigid3fTypes::Deriv& v1)
{
    return dot(getVCenter(v0),getVCenter(v1)) + dot(getVOrientation(v0), getVOrientation(v1));
}

#endif

} */// anonymous namespace


namespace sofa
{

namespace component
{

namespace controller
{

#ifndef SOFA_DOUBLE

using sofa::defaulttype::Rigid3fTypes;

template <>
void LCPForceFeedback< Rigid3fTypes >::computeForce(SReal x, SReal y, SReal z, SReal, SReal, SReal, SReal, SReal& fx, SReal& fy, SReal& fz)
{
    Rigid3fTypes::VecCoord state;
    Rigid3fTypes::VecDeriv forces;
    state.resize(1);
    state[0].getCenter() = sofa::defaulttype::Vec3f((float)x,(float)y,(float)z);
    computeForce(state,forces);
    fx = getVCenter(forces[0]).x();
    fy = getVCenter(forces[0]).y();
    fz = getVCenter(forces[0]).z();
}

#endif // SOFA_DOUBLE

#ifndef SOFA_FLOAT

using sofa::defaulttype::Rigid3dTypes;

template <>
void LCPForceFeedback< Rigid3dTypes >::computeForce(double x, double y, double z, double, double, double, double, double& fx, double& fy, double& fz)
{
    Rigid3dTypes::VecCoord state;
    Rigid3dTypes::VecDeriv forces;
    state.resize(1);
    state[0].getCenter() = sofa::defaulttype::Vec3d(x,y,z);
    computeForce(state,forces);
    fx = getVCenter(forces[0]).x();
    fy = getVCenter(forces[0]).y();
    fz = getVCenter(forces[0]).z();
}


template <>
void LCPForceFeedback< Rigid3dTypes >::computeWrench(const sofa::defaulttype::SolidTypes<double>::Transform &world_H_tool,
        const sofa::defaulttype::SolidTypes<double>::SpatialVector &/*V_tool_world*/,
        sofa::defaulttype::SolidTypes<double>::SpatialVector &W_tool_world )
{
    //msg_info()<<"WARNING : LCPForceFeedback::computeWrench is not implemented"<<std::endl;

    if (!this->f_activate.getValue())
    {
        return;
    }


    Rigid3dTypes::VecCoord state;
    Rigid3dTypes::VecDeriv forces;
    state.resize(1);
    state[0].getCenter()	  = world_H_tool.getOrigin();
    state[0].getOrientation() = world_H_tool.getOrientation();


    computeForce(state,forces);

    W_tool_world.setForce(getVCenter(forces[0]));
    W_tool_world.setTorque(getVOrientation(forces[0]));



    //Vec3d Force(0.0,0.0,0.0);

    //this->computeForce(world_H_tool.getOrigin()[0], world_H_tool.getOrigin()[1],world_H_tool.getOrigin()[2],
    //				   world_H_tool.getOrientation()[0], world_H_tool.getOrientation()[1], world_H_tool.getOrientation()[2], world_H_tool.getOrientation()[3],
    //				   Force[0],  Force[1], Force[2]);

    //W_tool_world.setForce(Force);
}

#endif // SOFA_FLOAT



int lCPForceFeedbackClass = sofa::core::RegisterObject("LCP force feedback for the device")
#ifndef SOFA_FLOAT
        .add< LCPForceFeedback<defaulttype::Vec1dTypes> >()
        .add< LCPForceFeedback<defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LCPForceFeedback<defaulttype::Vec1fTypes> >()
        .add< LCPForceFeedback<defaulttype::Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_HAPTICS_API LCPForceFeedback<defaulttype::Vec1dTypes>;
template class SOFA_HAPTICS_API LCPForceFeedback<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_HAPTICS_API LCPForceFeedback<defaulttype::Vec1fTypes>;
template class SOFA_HAPTICS_API LCPForceFeedback<defaulttype::Rigid3fTypes>;
#endif

SOFA_DECL_CLASS(LCPForceFeedback)


} // namespace controller

} // namespace component

} // namespace sofa
