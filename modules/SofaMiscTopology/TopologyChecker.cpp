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
#include <SofaMiscTopology/TopologyChecker.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace misc
{

using namespace defaulttype;
using namespace sofa::core::topology;


int TopologyCheckerClass = core::RegisterObject("Read topological Changes and process them.")
        .add< TopologyChecker >();


TopologyChecker::TopologyChecker()
    : m_draw( initData(&m_draw, false, "draw", "draw information"))
    , l_topology(initLink("topology", "link to the topology container"))
    , m_topology(nullptr)
{
    this->f_listening.setValue(true);
}


TopologyChecker::~TopologyChecker()
{

}


void TopologyChecker::init()
{
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }


    checkContainer();
}

void TopologyChecker::reinit()
{

}


bool TopologyChecker::checkContainer()
{
    bool result = false;
    if (m_topology->getTopologyType() == TopologyObjectType::TETRAHEDRON)
        result = checkTetrahedronTopology();
    else if (m_topology->getTopologyType() == TopologyObjectType::TRIANGLE)
        result = checkTriangleTopology();
    else if (m_topology->getTopologyType() == TopologyObjectType::TRIANGLE)
        result = checkEdgeTopology();

    return result;
}


bool TopologyChecker::checkTetrahedronTopology()
{
    bool ret = true;

    return ret && checkTriangleTopology();
}


bool TopologyChecker::checkTriangleTopology()
{
    bool ret = true;

    return ret && checkEdgeTopology();
}


bool TopologyChecker::checkEdgeTopology()
{
    bool ret = true;
    int nbE = m_topology->getNbEdges();
    const sofa::core::topology::BaseMeshTopology::SeqEdges& my_edges = m_topology->getEdges();
    
    if (nbE != my_edges.size())
    {
        msg_error() << "CheckEdgeTopology failed: not the good number of edges, getNbEdges returns " << nbE << " whereas edge array size is: " << my_edges.size();
        return false;
    }
        

    // check edge buffer
    for (std::size_t i = 0; i < nbE; ++i)
    {
        const auto& edge = my_edges[i];
        if (edge[0] == edge[1]) {
            msg_error() << "CheckEdgeTopology failed: edge " << i << " has 2 identical vertices: " << edge;
            ret = false;
        }
    }

    // check cross element
    std::size_t nbP = m_topology->getNbPoints();    
    
    std::set<int> edgeSet;
    for (std::size_t i = 0; i < nbP; ++i)
    {
        const auto& EdgesAV = m_topology->getEdgesAroundVertex(i);
        for (size_t j = 0; j < EdgesAV.size(); ++j)
        {
            const Topology::Edge& edge = my_edges[EdgesAV[j]];
            if (!(edge[0] == i || edge[1] == i))
            {
                msg_error() << "CheckEdgeTopology failed: edge " << EdgesAV[j] << ": [" << edge << "] not around vertex: " << i;
                ret = false;
            }

            // count number of edge
            edgeSet.insert(EdgesAV[j]);
        }
    }

    if (edgeSet.size() != nbE)
    {
        msg_error() << "CheckEdgeTopology failed: found " << edgeSet.size() << " edges in m_edgesAroundVertex out of " << nbE;
        ret = false;
    }

    return ret;
}


void TopologyChecker::checkCrossTopology()
{

}



void TopologyChecker::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::AnimateBeginEvent* ev = */simulation::AnimateBeginEvent::checkEventType(event))
    {
        //if (m_useDataInputs.getValue())
        //    processTopologicalChanges(this->getTime());
        //else
        //    processTopologicalChanges();
    }
    if (/* simulation::AnimateEndEvent* ev = */simulation::AnimateEndEvent::checkEventType(event))
    {

    }
}


void TopologyChecker::draw(const core::visual::VisualParams* vparams)
{
    if (!m_topology)
        return;

    if(!m_draw.getValue())
        return;

    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    //sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
    //m_topology->getContext()->get(triangleGeo);

    //if (!triangleGeo)
    //    return;

    //size_t nbTriangles = m_topology->getNbTriangles();

    //std::vector< Vector3 > trianglesToDraw;
    //std::vector< Vector3 > pointsToDraw;

    //for (size_t i = 0 ; i < triangleIncisionInformation.size() ; i++)
    //{
    //    for (size_t j = 0 ; j < triangleIncisionInformation[i].triangleIndices.size() ; j++)
    //    {
    //        unsigned int triIndex = triangleIncisionInformation[i].triangleIndices[j];

    //        if ( triIndex > nbTriangles -1)
    //            break;

    //        Vec3Types::Coord coord[3];
    //        triangleGeo->getTriangleVertexCoordinates(triIndex, coord);

    //        for(unsigned int k = 0 ; k < 3 ; k++)
    //            trianglesToDraw.push_back(coord[k]);

    //        Vector3 a;
    //        a.clear();
    //        for (unsigned k = 0 ; k < 3 ; k++)
    //            a += coord[k] * triangleIncisionInformation[i].barycentricCoordinates[j][k];

    //        pointsToDraw.push_back(a);
    //    }
    //}

    //vparams->drawTool()->drawTriangles(trianglesToDraw, Vec<4,float>(0.0,0.0,1.0,1.0));
    //vparams->drawTool()->drawPoints(pointsToDraw, 15.0,  Vec<4,float>(1.0,0.0,1.0,1.0));

    //if (!errorTrianglesIndices.empty())
    //{
    //    trianglesToDraw.clear();
    //    /* initialize random seed: */
    //    srand ( (unsigned int)time(nullptr) );

    //    for (size_t i = 0 ; i < errorTrianglesIndices.size() ; i++)
    //    {
    //        Vec3Types::Coord coord[3];
    //        triangleGeo->getTriangleVertexCoordinates(errorTrianglesIndices[i], coord);

    //        for(unsigned int k = 0 ; k < 3 ; k++)
    //            trianglesToDraw.push_back(coord[k]);
    //    }

    //    vparams->drawTool()->drawTriangles(trianglesToDraw,
    //            Vec<4,float>(1.0f,(float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX, 1.0f));
    //}
}

} // namespace misc

} // namespace component

} // namespace sofa
