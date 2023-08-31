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

#include <sofa/component/animationloop/config.h>

#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/simulation/CollisionAnimationLoop.h>

namespace sofa::component::animationloop
{

/** Simple animation loop that given a list of tags, animate the graph one tag after another.
*/
class SOFA_COMPONENT_ANIMATIONLOOP_API MultiTagAnimationLoop : public sofa::simulation::CollisionAnimationLoop
{
public:
    typedef sofa::simulation::CollisionAnimationLoop Inherit;
    SOFA_CLASS(MultiTagAnimationLoop,sofa::simulation::CollisionAnimationLoop);

    MultiTagAnimationLoop();

    void init() override;

    virtual void clear();

    ~MultiTagAnimationLoop() override;

    void step (const sofa::core::ExecParams* params, SReal dt) override;

private:
    sofa::core::objectmodel::TagSet tagList;
};

} // namespace sofa::component::animationloop
