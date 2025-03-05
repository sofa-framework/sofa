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
#ifndef SOFA_SIMULATION_PIPELINEIMPL_H
#define SOFA_SIMULATION_PIPELINEIMPL_H

#include <sofa/core/collision/Pipeline.h>
#include <sofa/simulation/config.h>


namespace sofa::simulation
{

class SOFA_SIMULATION_CORE_API PipelineImpl : public sofa::core::collision::Pipeline
{

protected:
    PipelineImpl();

    ~PipelineImpl() override;
public:
    void init() override;

    void reset() override;

    /// Remove collision response from last step
    void computeCollisionReset() override;
    /// Detect new collisions. Note that this step must not modify the simulation graph
    void computeCollisionDetection() override;
    /// Add collision response in the simulation graph
    void computeCollisionResponse() override;
};


} // namespace sofa::simulation


#endif
