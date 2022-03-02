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

#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::core::topology
{

class BaseTopologicalMapping : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseTopologicalMapping, objectmodel::BaseObject);

    /// Input Topology
    using In = BaseMeshTopology;
    /// Output Topology
    using Out = BaseMeshTopology;

    /// Method called at each topological changes propagation which comes from the INPUT topologies to adapt the OUTPUT topologies :
    virtual void updateTopologicalMappingTopDown() = 0;

    /// Method called at each topological changes propagation which comes from the OUTPUT topologies to adapt the INPUT topologies :
    virtual void updateTopologicalMappingBottomUp() {}

    /// Return true if this mapping is able to propagate topological changes from input to output topologies
    virtual bool propagateFromInputToOutputModel() { return true; }

    /// Return true if this mapping is able to propagate topological changes from output to input topologies
    virtual bool propagateFromOutputToInputModel() { return false; }

    virtual bool isTopologyAnInput(core::topology::Topology* topology) { return false; }
    virtual bool isTopologyAnOutput(core::topology::Topology* topology) { return false; }

};

} // namespace sofa::core::topology
