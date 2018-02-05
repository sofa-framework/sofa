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
#ifndef SOFA_COMPONENT_COLLISION_TRIANGLEMODELINREGULARGRID_H
#define SOFA_COMPONENT_COLLISION_TRIANGLEMODELINREGULARGRID_H
#include "config.h"

#include <SofaMeshCollision/TriangleModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa
{

namespace component
{

namespace collision
{


class TriangleModelInRegularGrid : public TriangleModel
{
public:
    SOFA_CLASS(TriangleModelInRegularGrid, TriangleModel);

    virtual void init() override;
    virtual void computeBoundingTree ( int maxDepth=0 ) override;

    sofa::core::topology::BaseMeshTopology* _topology;
    sofa::core::topology::BaseMeshTopology* _higher_topo;
    core::behavior::MechanicalState<defaulttype::Vec3Types>* _higher_mstate;

protected:
    TriangleModelInRegularGrid();
    ~TriangleModelInRegularGrid();
};

}

}

}

#endif
