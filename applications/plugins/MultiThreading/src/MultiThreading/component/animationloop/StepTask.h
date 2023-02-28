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
#pragma once

#include <MultiThreading/config.h>

#ifndef SOFA_BUILD_MULTITHREADING
SOFA_DEPRECATED_HEADER_NOT_REPLACED("v23.06", "v23.12")
#endif

#include <sofa/simulation/CpuTask.h>

// forawrd declaraion
namespace sofa::core::behavior
{
class BaseAnimationLoop;
}


namespace multithreading::component::animationloop
{

class SOFA_ATTRIBUTE_DEPRECATED("v23.06", "v23.12", "This class is no longer used.")
StepTask : public sofa::simulation::CpuTask
{
public:
    StepTask(sofa::core::behavior::BaseAnimationLoop* aloop, const double t, sofa::simulation::CpuTask::Status* pStatus);

    ~StepTask() override;

    MemoryAlloc run() final;

private:

    sofa::core::behavior::BaseAnimationLoop* animationloop;
    const double dt;

};

}


