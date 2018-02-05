/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_COLLISION_DEFAULTPIPELINE_H
#define SOFA_COMPONENT_COLLISION_DEFAULTPIPELINE_H
#include "config.h"

#include <sofa/simulation/PipelineImpl.h>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_BASE_COLLISION_API DefaultPipeline : public sofa::simulation::PipelineImpl
{
public:
    SOFA_CLASS(DefaultPipeline,sofa::simulation::PipelineImpl);

    Data<bool> d_doPrintInfoMessage;
    Data<bool> d_doDebugDraw;
    Data<int>  d_depth;
protected:
    DefaultPipeline();
public:
    void init() override;
    void draw(const core::visual::VisualParams* vparams) override;

    /// get the set of response available with the current collision pipeline
    std::set< std::string > getResponseList() const override;
protected:
    // -- Pipeline interface
    /// Remove collision response from last step
    virtual void doCollisionReset() override;
    /// Detect new collisions. Note that this step must not modify the simulation graph
    virtual void doCollisionDetection(const sofa::helper::vector<core::CollisionModel*>& collisionModels) override;
    /// Add collision response in the simulation graph
    virtual void doCollisionResponse() override;

    virtual void checkDataValues() ;
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
