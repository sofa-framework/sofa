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
#define SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_CPP

#include <SofaHaptics/LCPForceFeedback.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>
/*
namespace
{

using sofa::defaulttype::Rigid3Types;

template<>
bool derivVectors<Rigid3Types>(const Rigid3Types::VecCoord& x0, const Rigid3Types::VecCoord& x1, Rigid3Types::VecDeriv& d, bool derivRotation)
{
    return derivRigid3Vectors<Rigid3Types>(x0, x1, d, derivRotation);
}

template <>
double computeDot<Rigid3Types>(const Rigid3Types::Deriv& v0, const Rigid3Types::Deriv& v1)
{
    return dot(getVCenter(v0),getVCenter(v1)) + dot(getVOrientation(v0), getVOrientation(v1));
}



#ifdef SOFA_WITH_FLOAT
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


using sofa::defaulttype::Rigid3Types;

template <>
void LCPForceFeedback< Rigid3Types >::computeForce(double x, double y, double z, double, double, double, double, double& fx, double& fy, double& fz)
{
    Rigid3Types::VecCoord state;
    Rigid3Types::VecDeriv forces;
    state.resize(1);
    state[0].getCenter() = sofa::defaulttype::Vec3d(x,y,z);
    computeForce(state,forces);
    fx = getVCenter(forces[0]).x();
    fy = getVCenter(forces[0]).y();
    fz = getVCenter(forces[0]).z();
}


template <>
void LCPForceFeedback< Rigid3Types >::computeWrench(const sofa::defaulttype::SolidTypes<double>::Transform &world_H_tool,
        const sofa::defaulttype::SolidTypes<double>::SpatialVector &/*V_tool_world*/,
        sofa::defaulttype::SolidTypes<double>::SpatialVector &W_tool_world )
{
    //msg_info()<<"WARNING : LCPForceFeedback::computeWrench is not implemented"<<std::endl;

    if (!this->f_activate.getValue())
    {
        return;
    }


    Rigid3Types::VecCoord state;
    Rigid3Types::VecDeriv forces;
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

 



int lCPForceFeedbackClass = sofa::core::RegisterObject("LCP force feedback for the device")
        .add< LCPForceFeedback<defaulttype::Vec1Types> >()
        .add< LCPForceFeedback<defaulttype::Rigid3Types> >()

        ;

template class SOFA_HAPTICS_API LCPForceFeedback<defaulttype::Vec1Types>;
template class SOFA_HAPTICS_API LCPForceFeedback<defaulttype::Rigid3Types>;


} // namespace controller

} // namespace component

} // namespace sofa
