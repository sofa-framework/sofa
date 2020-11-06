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

#include <SofaMiscTopology/config.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::misc
{

/** 
*
*/
class SOFA_SOFAMISCTOPOLOGY_API TopologyChecker: public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(TopologyChecker, core::objectmodel::BaseObject);

    void init() override;

    void reinit() override;

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    void draw(const core::visual::VisualParams* vparams) override;

    bool checkContainer();

protected:
    TopologyChecker();

    ~TopologyChecker() override;


    bool checkHexahedronTopology();
    bool checkTetrahedronTopology();
    bool checkQuadTopology();
    bool checkTriangleTopology();
    bool checkEdgeTopology();
    
    
public:
    ///< draw information
    Data<bool> m_draw;

    Data<bool> d_eachStep;

    /// Link to be set to the topology container in the component graph.
    SingleLink<TopologyChecker, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;


protected:
    core::topology::BaseMeshTopology* m_topology;

};


} // namespace sofa::component::misc
