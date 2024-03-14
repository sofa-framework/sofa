/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <MultiThreading/component/mapping/linear/BeamLinearMapping_mt.inl>
#include <sofa/core/ObjectFactory.h>
//#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.inl>
#include <MultiThreading/ParallelImplementationsRegistry.h>

namespace multithreading::component::mapping::linear
{

const bool isBeamLinearMapping_mtImplementationRegistered =
    multithreading::ParallelImplementationsRegistry::addEquivalentImplementations("BeamLinearMapping", "BeamLinearMapping_mt");

//using namespace defaulttype;
// Register in the Factory
int BeamLinearMapping_mtClass = sofa::core::RegisterObject("Set the positions and velocities of points attached to a beam using linear interpolation between DOFs")
        .add< BeamLinearMapping_mt< Rigid3Types, Vec3Types > >()
        ;

template class BeamLinearMapping_mt< Rigid3Types, Vec3Types >;

}

