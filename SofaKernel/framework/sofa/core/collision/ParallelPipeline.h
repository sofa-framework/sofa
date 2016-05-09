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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_COLLISION_PARALLELPIPELINE_H
#define SOFA_CORE_COLLISION_PARALLELPIPELINE_H

#include <sofa/core/collision/Pipeline.h>
#include <sofa/core/ParallelCollisionModel.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace collision
{

class SOFA_CORE_API ParallelPipeline : public virtual Pipeline
{
protected:

    ParallelPipeline();

    virtual ~ParallelPipeline();

public:

    virtual void parallelComputeCollisions() = 0;
};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
