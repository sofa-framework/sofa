/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TEST_FAKE_TOPOLOGYSCENE_H
#define SOFA_COMPONENT_TOPOLOGY_TEST_FAKE_TOPOLOGYSCENE_H

#include <sofa/core/topology/Topology.h>
#include <SofaSimulationGraph/SimpleApi.h>

class fake_TopologyScene
{
public:
    /**
    * Default constructor, take the filepath of the mesh file to load, the type of topology and if the topology is static (MeshTopology)
    */
    fake_TopologyScene(const std::string& filename, sofa::core::topology::TopologyObjectType topoType, bool staticTopo = false);

    /// Method to load the mesh and fill the topology asked
    bool loadMeshFile();

    /// Method to get acces to node containing the meshLoader and the toplogy container.
    sofa::simulation::Node::SPtr getNode() { return m_root; }

protected:
    /// Simulation object
    sofa::simulation::Simulation::SPtr m_simu;
    /// Node containing the topology
    sofa::simulation::Node::SPtr m_root;

    /// Type of topology asked
    sofa::core::topology::TopologyObjectType m_topoType;
    /// filepath of the mesh to load
    std::string m_filename;
    /// Bool storing if static or dynamyc topology.
    bool m_staticTopology;
};


#endif // SOFA_COMPONENT_TOPOLOGY_TEST_FAKE_TOPOLOGYSCENE_H