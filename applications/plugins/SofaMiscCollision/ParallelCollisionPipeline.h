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
#ifndef SOFA_COMPONENT_COLLISION_PARALLELCOLLISIONPIPELINE_H
#define SOFA_COMPONENT_COLLISION_PARALLELCOLLISIONPIPELINE_H
#include "config.h"

#include <sofa/core/collision/ParallelPipeline.h>
#include <sofa/simulation/PipelineImpl.h>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_MISC_COLLISION_API ParallelCollisionPipeline : public sofa::simulation::PipelineImpl, public sofa::core::collision::ParallelPipeline
{
public:
    SOFA_CLASS2(ParallelCollisionPipeline,sofa::simulation::PipelineImpl,sofa::core::collision::ParallelPipeline);

    Data<bool> bVerbose;
    Data<bool> bDraw;
    Data<int> depth;

    ParallelCollisionPipeline();

    virtual void parallelComputeCollisions();

    void draw(const core::visual::VisualParams* vparams);
    long int contactSum;
    sofa::helper::vector<int> procs;
    sofa::helper::vector<core::ParallelCollisionModel*> parallelCollisionModels;

    sofa::helper::vector< a1::Shared<bool>* > parallelBoundingTreeDone;
    a1::Shared<int> parallelBoundingTreeDoneAll;

    /// get the set of response available with the current collision pipeline
    std::set< std::string > getResponseList() const;
protected:
    // -- Pipeline interface

    /// Remove collision response from last step
    virtual void doCollisionReset();
    virtual void doRealCollisionReset();

    /// Detect new collisions. Note that this step must not modify the simulation graph
    virtual void doCollisionDetection(const sofa::helper::vector<core::CollisionModel*>& collisionModels);
    /// Add collision response in the simulation graph
    virtual void doCollisionResponse();
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
