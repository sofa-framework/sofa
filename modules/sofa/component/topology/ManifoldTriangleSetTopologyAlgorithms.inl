/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYALGORITHMS_INL

#include <sofa/component/topology/ManifoldTriangleSetTopologyContainer.h>
#include <sofa/component/topology/ManifoldTriangleSetTopologyModifier.h>
#include <sofa/component/topology/ManifoldTriangleSetTopologyAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>

#include <algorithm>
#include <functional>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

template<class DataTypes>
void ManifoldTriangleSetTopologyAlgorithms< DataTypes >::init()
{
    TriangleSetTopologyAlgorithms< DataTypes >::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    this->getContext()->get(m_geometryAlgorithms);
}



template<class DataTypes>
int ManifoldTriangleSetTopologyAlgorithms< DataTypes >::SplitAlongPath(unsigned int pa, Coord& a, unsigned int pb, Coord& b,
        sofa::helper::vector< sofa::core::componentmodel::topology::TopologyObjectType>& topoPath_list,
        sofa::helper::vector<unsigned int>& indices_list,
        sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& coords_list,
        sofa::helper::vector<EdgeID>& new_edges, double epsilonSnapPath, double epsilonSnapBorder)
{

    /// force the creation of TriangleEdgeShellArray
    m_container->getTriangleEdgeShellArray();
    /// force the creation of TriangleVertexShellArray
    m_container->getTriangleVertexShellArray();

    int result = TriangleSetTopologyAlgorithms< DataTypes >::SplitAlongPath (pa, a, pb, b, topoPath_list, indices_list, coords_list, new_edges, epsilonSnapPath, epsilonSnapBorder);

    return result;
}






template<class DataTypes>
bool ManifoldTriangleSetTopologyAlgorithms< DataTypes >::InciseAlongEdgeList (const sofa::helper::vector<unsigned int>& edges, sofa::helper::vector<unsigned int>& new_points, sofa::helper::vector<unsigned int>& end_points)
{
    // std::cout << "ManifoldTriangleSetTopologyAlgorithms::InciseAlongEdgeList()" << std::endl;

    //// STEP 1 - Incise with the TriangleSetTopologyAlgorithms function. Addremovetriangles() automatically reorder the mesh
    bool ok = TriangleSetTopologyAlgorithms< DataTypes >::InciseAlongEdgeList (edges, new_points, end_points);

    //// STEP 2 - Create a ROI of edges (without double) around the incision
    sofa::helper::vector< PointID > listVertex;
    sofa::helper::vector< EdgeID > listEdges;
    bool doublon;

    // Old vertices from the list of edges
    for (unsigned int i = 0; i<edges.size(); i++)
    {
        Edge theEdge = m_container->getEdge (edges[i]);

        for (unsigned int j = 0; j<2; j++)
        {
            const sofa::helper::vector< PointID >& shell = m_container->getVertexVertexShell( theEdge[j] );

            for (unsigned int k = 0; k<shell.size(); k++)
            {
                doublon = false;
                for (unsigned int u = 0; u<listVertex.size(); u++)
                {
                    if (listVertex[u] == shell[k])
                    {
                        doublon = true;
                        break;
                    }
                }

                if (!doublon)
                    listVertex.push_back (shell[k]);
            }
        }
    }


    // New points, from vertices just created:
    for (unsigned int i = 0; i < new_points.size(); i++)
    {
        const sofa::helper::vector< PointID >& shell = m_container->getVertexVertexShell( i );

        for (unsigned int j = 0; j<shell.size(); j++)
        {
            doublon = false;
            for (unsigned int k = 0; k<listVertex.size(); k++)
            {
                if (listVertex[k] == shell[j])
                {
                    doublon = true;
                    break;
                }
            }

            if (!doublon)
                listVertex.push_back (shell[j]);
        }
    }


    // Creating ROI of edges from list of vertices.
    for (unsigned int i = 0; i<listVertex.size(); i++)
    {
        const sofa::helper::vector< unsigned int >& shell = m_container->getEdgeVertexShell (listVertex[i]);

        for (unsigned int j = 0; j<shell.size(); j++)
        {
            doublon = false;
            for (unsigned int k = 0; k<listEdges.size(); k++)
            {
                if (listEdges[k] == shell[j])
                {
                    doublon = true;
                    break;
                }
            }

            if (!doublon)
                listEdges.push_back (shell[j]);
        }
    }


    //// STEP 3 - Test the ROI mesh for swapremeshing

    // recompute elements border list to take into account the incision
    if (m_container->hasBorderElementLists())
        m_container->createElementsOnBorder();


    m_modifier->swapRemeshing (listEdges);

    // std::cout <<"end incision"<<std::endl;

    return ok;
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_MANIFOLDEDGESETTOPOLOGYALGORITHMS_INL
