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
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/Utils.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/simulation/graph/SimpleApi.h>
#include <sofa/simulation/Node.h>

class fake_TopologyScene
{
public:
    /**
    * Default constructor, take the filepath of the mesh file to load, the type of topology and if the topology is static (MeshTopology)
    */
    fake_TopologyScene(const std::string& filename, sofa::geometry::ElementType topoType, bool staticTopo = false)
        : m_topoType(topoType)
        , m_filename(filename)
        , m_staticTopology(staticTopo)
    {
        //force load sofabase
        sofa::helper::system::DataRepository.addFirstPath(SOFA_COMPONENT_TOPOLOGY_TESTING_RESOURCES_DIR);

        loadMeshFile();
    }

    /// Method to load the mesh and fill the topology asked
    bool loadMeshFile()
    {
        using namespace sofa::simpleapi;
        using namespace sofa::core::topology;

        m_simu = createSimulation("DAG");
        m_root = createRootNode(m_simu, "root");

        sofa::simpleapi::importPlugin("Sofa.Component.IO.Mesh");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Constant");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Dynamic");
        sofa::simpleapi::importPlugin("Sofa.Component.Mass");

        createObject(m_root, "DefaultAnimationLoop");

        std::string loaderType = "MeshOBJLoader";
        if (m_topoType == sofa::geometry::ElementType::TETRAHEDRON || m_topoType == sofa::geometry::ElementType::HEXAHEDRON)
            loaderType = "MeshGmshLoader";


        auto loader = createObject(m_root, loaderType, {
            { "name","loader" },
            { "filename", sofa::helper::system::DataRepository.getFile(m_filename) } });

        auto meca = createObject(m_root, "MechanicalObject", {
            { "name", "dof" },
            { "position", "@loader.position"} });


        if (m_staticTopology)
        {
            auto topo = createObject(m_root, "MeshTopology", {
                { "name", "topoCon" },
                { "src", "@loader" }
                });
        }
        else
        {
            std::string topoType = "";
            if (m_topoType == sofa::geometry::ElementType::POINT)
                topoType = "Point";
            else if (m_topoType == sofa::geometry::ElementType::EDGE)
                topoType = "Edge";
            else if (m_topoType == sofa::geometry::ElementType::TRIANGLE)
                topoType = "Triangle";
            else if (m_topoType == sofa::geometry::ElementType::QUAD)
                topoType = "Quad";
            else if (m_topoType == sofa::geometry::ElementType::TETRAHEDRON)
                topoType = "Tetrahedron";
            else if (m_topoType == sofa::geometry::ElementType::HEXAHEDRON)
                topoType = "Hexahedron";

            // create topology components
            auto topo = createObject(m_root, topoType + "SetTopologyContainer", {
                { "name", "topoCon" },
                { "src", "@loader" }
                });

            createObject(m_root, topoType + "SetTopologyModifier", { { "name", "topoMod" } });
            createObject(m_root, topoType + "SetGeometryAlgorithms", { { "name", "topoGeo" } });

            // Add some mechanical components
            createObject(m_root, "MeshMatrixMass");

            if (m_topoType == sofa::geometry::ElementType::EDGE) {
                sofa::simpleapi::importPlugin("Sofa.Component.SolidMechanics.Spring");
                createObject(m_root, "VectorSpringForceField", { {"useTopology", "true"} });
            }
                
                
        }

        sofa::simulation::node::initRoot(m_root.get());

        return true;
    }

    /// Method to get acces to node containing the meshLoader and the toplogy container.
    sofa::simulation::Node::SPtr getNode() { return m_root; }

private:
    /// Simulation object
    sofa::simulation::Simulation::SPtr m_simu;
    /// Node containing the topology
    sofa::simulation::Node::SPtr m_root;

    /// Type of topology asked
    sofa::geometry::ElementType m_topoType;
    /// filepath of the mesh to load
    std::string m_filename;
    /// Bool storing if static or dynamyc topology.
    bool m_staticTopology;
};
