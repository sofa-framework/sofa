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

#include <sofa/component/topology/utility/config.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::topology::utility
{

/** 
* The class TopologyChecker represents a SOFA component which can be added in a scene graph to test a given Topology.
* The topology component to be tested need to be linked using @sa l_topology. 
* If the Data @sa d_eachStep is set to true, the topology will be tested at each step using the generic method @sa checkTopology
* 
* Otherwise each method can be called manually: 
* @CheckTopology will call the appropriate Check{TopologyType}Topology then call the lower level of CheckTopology. 
*   - i.e for a Tetrahedron Topology, CheckTopology with call @sa checkTetrahedronTopology then @sa checkTriangleTopology and finally @sa checkEdgeTopology
*   - At each level the topology is checked through the main element container and also the cross topology containers
*   - Each method return a bool and will display msg_error if problems are detected.
*/
class SOFA_COMPONENT_TOPOLOGY_UTILITY_API TopologyChecker: public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(TopologyChecker, core::objectmodel::BaseObject);

    void init() override;

    void reinit() override;

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    void draw(const core::visual::VisualParams* vparams) override;

    bool checkTopology();


    /// Edge methods
    ///{
    /// Full method to check Edge Topology. Will call @sa checkEdgeContainer and @sa checkEdgeToVertexCrossContainer
    bool checkEdgeTopology();

    /// Method to test Edge container concistency
    bool checkEdgeContainer();

    /// Method to test Edge to vertex cross container concistency
    bool checkEdgeToVertexCrossContainer();
    ///}


    /// Triangle methods
    ///{
    /// Full method to check Triangle Topology. Will call @sa checkTriangleContainer, @sa checkTriangleToEdgeCrossContainer and @sa checkTriangleToVertexCrossContainer
    bool checkTriangleTopology();

    /// Method to test Triangle container concistency
    bool checkTriangleContainer();

    /// Method to test triangles to edges cross container concistency
    bool checkTriangleToEdgeCrossContainer();

    /// Method to test triangles to vertices cross container concistency
    bool checkTriangleToVertexCrossContainer();
    ///}


    /// Quad methods
    ///{
    /// Full method to check Quad Topology. Will call @sa checkQuadContainer, @sa checkQuadToEdgeCrossContainer and @sa checkQuadToVertexCrossContainer
    bool checkQuadTopology();

    /// Method to test quad container concistency
    bool checkQuadContainer();

    /// Method to test quads to edges cross container concistency
    bool checkQuadToEdgeCrossContainer();

    /// Method to test quads to vertices cross container concistency
    bool checkQuadToVertexCrossContainer();
    /// }
    

    /// Tetrahedron methods
    ///{
    /// Full method to check Tetrahedron Topology. Will call @sa checkTetrahedronContainer, @sa checkTetrahedronToTriangleCrossContainer
    /// @sa checkTetrahedronToEdgeCrossContainer and @sa checkTetrahedronToVertexCrossContainer
    bool checkTetrahedronTopology();

    /// Method to test Tetrahedron container concistency
    bool checkTetrahedronContainer();

    /// Method to test Tetrahedron to triangles cross container concistency
    bool checkTetrahedronToTriangleCrossContainer();

    /// Method to test Tetrahedron to edges cross container concistency
    bool checkTetrahedronToEdgeCrossContainer();

    /// Method to test Tetrahedron to vertices cross container concistency
    bool checkTetrahedronToVertexCrossContainer();
    /// }


    /// Hexahedron methods
    ///{
    /// Full method to check Hexahedron Topology. Will call @sa checkHexahedronContainer, @sa checkHexahedronToQuadCrossContainer
    /// @sa checkHexahedronToEdgeCrossContainer and @sa checkHexahedronToVertexCrossContainer
    bool checkHexahedronTopology();

    /// Method to test Hexahedron container concistency
    bool checkHexahedronContainer();

    /// Method to test Hexahedron to quads cross container concistency
    bool checkHexahedronToQuadCrossContainer();

    /// Method to test Hexahedron to edges cross container concistency
    bool checkHexahedronToEdgeCrossContainer();

    /// Method to test Hexahedron to vertices cross container concistency
    bool checkHexahedronToVertexCrossContainer();
    /// }

   

protected:
    TopologyChecker();

    ~TopologyChecker() override;


public:
    /// bool to check topology at each step.
    Data<bool> d_eachStep;

    /// Link to be set to the topology container in the component graph.
    SingleLink<TopologyChecker, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;


protected:
    core::topology::BaseMeshTopology::SPtr m_topology;

};


} // namespace sofa::component::topology::utility
