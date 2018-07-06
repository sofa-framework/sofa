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

#include <sofa/helper/system/FileRepository.h>
#include <SofaLoader/MeshObjLoader.h>
#include <sofa/helper/Utils.h>
#include "fake_TopologyScene.h"

using namespace sofa::simpleapi;
using namespace sofa::simpleapi::components;
using namespace sofa::core::topology;

fake_TopologyScene::fake_TopologyScene(const std::string& filename, TopologyObjectType topoType, bool staticTopo)
    : m_filename(filename)
    , m_topoType(topoType)
    , m_staticTopology(staticTopo)
{
    loadMeshFile();
}

bool fake_TopologyScene::loadMeshFile()
{
    m_simu = createSimulation("DAG");
    m_root = createRootNode(m_simu, "root");

    std::string loaderType = "MeshObjLoader";
    if (m_topoType == TopologyObjectType::TETRAHEDRON || m_topoType == TopologyObjectType::HEXAHEDRON)
        loaderType = "MeshGmshLoader";


    auto loader = createObject(m_root, loaderType, {
        { "name","loader" },
        { "filename", sofa::helper::system::DataRepository.getFile(m_filename) } });

    // could do better but will work for now
    std::string topoConType = "";
    if (m_staticTopology)
        topoConType = "MeshTopology";
    else if (m_topoType == TopologyObjectType::POINT)
        topoConType = "PointSetTopologyContainer";
    else if (m_topoType == TopologyObjectType::EDGE)
        topoConType = "EdgeSetTopologyContainer";
    else if (m_topoType == TopologyObjectType::TRIANGLE)
        topoConType = "TriangleSetTopologyContainer";
    else if (m_topoType == TopologyObjectType::QUAD)
        topoConType = "QuadSetTopologyContainer";
    else if (m_topoType == TopologyObjectType::TETRAHEDRON)
        topoConType = "TetrahedronSetTopologyContainer";
    else if (m_topoType == TopologyObjectType::HEXAHEDRON)
        topoConType = "HexahedronSetTopologyContainer";


    auto topo = createObject(m_root, topoConType, {
        { "name", "topoCon" },
        { "src", "@loader" }
    });
   
    return true;
}

