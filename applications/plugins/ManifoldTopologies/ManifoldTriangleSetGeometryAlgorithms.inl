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

#include <ManifoldTopologies/ManifoldTriangleSetGeometryAlgorithms.h>
#include <ManifoldTopologies/ManifoldTriangleSetTopologyContainer.h>
#include <ManifoldTopologies/ManifoldTriangleSetTopologyModifier.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::core::behavior;


template<class DataTypes>
void ManifoldTriangleSetGeometryAlgorithms< DataTypes >::init()
{
    TriangleSetGeometryAlgorithms< DataTypes >::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    this->getContext()->get(m_geometryAlgorithms);
}


template<class DataTypes>
void ManifoldTriangleSetGeometryAlgorithms< DataTypes >::reinit()
{
    if(!(m_triSwap.getValue()).empty() && this->getContext()->getAnimate()) //temporarly test for the funciton edgeSwap
    {
        edgeSwapProcess (m_triSwap.getValue());
    }

    if(m_swapMesh.getValue() && this->getContext()->getAnimate())
    {
        swapRemeshing();
    }
}




template<class DataTypes>
void ManifoldTriangleSetGeometryAlgorithms< DataTypes >::edgeSwapProcess (const sofa::type::vector <EdgeID>& listEdges)
{

    for (unsigned int i = 0; i<listEdges.size(); i++)
    {
        edgeSwap(listEdges[i]);
        //   m_modifier->propagateTopologicalChanges();
    }
}



template<class DataTypes>
void ManifoldTriangleSetGeometryAlgorithms< DataTypes >::edgeSwapProcess (const TriangleID& indexTri1, const TriangleID& indexTri2)
{
    sofa::type::vector < unsigned int > listVertex;
    unsigned int cpt = 0;
    int commonEdgeIndex;
    bool test = true;
    Edge commonEdge;

    Triangle vertexTriangle1 = m_container->getTriangleArray()[indexTri1];
    for (unsigned int i = 0; i < 3; i++)
        listVertex.push_back(vertexTriangle1[i]);

    Triangle vertexTriangle2 = m_container->getTriangleArray()[indexTri2];
    for (unsigned int i = 0; i <3; i++)
    {
        test =true;
        for (unsigned int j = 0; j <3; j++)
        {
            if (vertexTriangle2[i] == listVertex[j])
            {
                commonEdge[cpt] = vertexTriangle2[i];
                cpt++;
                test = false;
                break;
            }
        }

        if (test)
            listVertex.push_back(vertexTriangle2[i]);
    }


    if (commonEdge[0] < commonEdge[1])
        commonEdgeIndex = m_container->getEdgeIndex(commonEdge[0], commonEdge[1]);
    else
        commonEdgeIndex = m_container->getEdgeIndex(commonEdge[1], commonEdge[0]);

    if (commonEdgeIndex == -1 || listVertex.size() > 4)
    {
        msg_error() << "EdgeSwapProcess: the two selected triangles are not adjacent";
        return;
    }
    else
    {
        edgeSwap(commonEdgeIndex);
        //   m_modifier->propagateTopologicalChanges();
    }
}



template<class DataTypes>
bool ManifoldTriangleSetGeometryAlgorithms< DataTypes >::edgeSwap(const EdgeID& edgeIndex)
{

    sofa::type::vector< Triangle > triToAdd; triToAdd.resize (2);
    sofa::type::vector< TriangleID > triToAddID; triToAddID.resize (2);
    sofa::type::vector< sofa::type::vector<Index > > ancestors; ancestors.resize(2);
    sofa::type::vector< sofa::type::vector< double > > baryCoefs; baryCoefs.resize (2);
    sofa::type::vector< TriangleID > trianglesIndex2remove; trianglesIndex2remove.resize(2);

    trianglesIndex2remove = m_container->getTrianglesAroundEdgeArray()[edgeIndex];

    if(trianglesIndex2remove.size()>2)
    {
        msg_error() << "EdgeSwap: the topology is not manifold around the input edge: " << edgeIndex;
        return false;
    }
    else if (trianglesIndex2remove.size() == 1)
    {
        msg_error() << "EdgeSwap: the edge: "<< edgeIndex << " is on the border of the mesh. Swaping this edge is impossible";
        return false;
    }

    int edgeInTri1 = m_container->getEdgeIndexInTriangle ( m_container->getEdgesInTriangle (trianglesIndex2remove[0]), edgeIndex);
    int edgeInTri2 = m_container->getEdgeIndexInTriangle ( m_container->getEdgesInTriangle (trianglesIndex2remove[1]), edgeIndex);
    Triangle vertexTriangle1 = m_container->getTriangle (trianglesIndex2remove[0]);
    Triangle vertexTriangle2 = m_container->getTriangle (trianglesIndex2remove[1]);

    Triangle newTri;
    const typename DataTypes::VecCoord& coords = m_geometryAlgorithms->getDOF()->read(core::ConstVecCoordId::position())->getValue();
    typename DataTypes::Coord tri1[3], tri2[3];

    newTri[0] = vertexTriangle1[ edgeInTri1 ];
    newTri[1] = vertexTriangle1[ (edgeInTri1+1)%3 ];
    newTri[2] = vertexTriangle2[ edgeInTri2 ];
    triToAdd[0] = newTri;

    for (unsigned int i = 0; i<3; i++)
        tri1[i] = coords[ newTri[i]];

    newTri[0] = vertexTriangle2[ edgeInTri2 ];
    newTri[1] = vertexTriangle2[ (edgeInTri2+1)%3 ];
    newTri[2] = vertexTriangle1[ edgeInTri1 ];
    triToAdd[1] = newTri;

    for (unsigned int i = 0; i<3; i++)
        tri2[i] = coords[ newTri[i]];

    if (!m_geometryAlgorithms->isDiagonalsIntersectionInQuad (tri1, tri2) )
    {
        msg_error() << "EdgeSwap: the new edge swaped will be outside the quad.";
        return false;
    }

    for (unsigned int i = 0; i <2; i++)
    {
        ancestors[i].push_back (trianglesIndex2remove[0]); baryCoefs[i].push_back (0.5);
        ancestors[i].push_back (trianglesIndex2remove[1]); baryCoefs[i].push_back (0.5);
    }
    triToAddID[0] = TriangleID(m_container->getNbTriangles());
    triToAddID[1] = TriangleID(m_container->getNbTriangles()+1);

    m_modifier->addRemoveTriangles (triToAdd.size(), triToAdd, triToAddID, ancestors, baryCoefs, trianglesIndex2remove);

    return true;
}



template<class DataTypes>
void ManifoldTriangleSetGeometryAlgorithms< DataTypes >::swapRemeshing()
{
    // All the mesh is about to be remeshed by swaping edges. So passing a simple list.
    sofa::type::vector <EdgeID> listEdges;
    for(unsigned int i = 0; i<m_container->getNumberOfEdges(); i++)
        listEdges.push_back (i);

    swapRemeshing(listEdges);
}



template<class DataTypes>
void ManifoldTriangleSetGeometryAlgorithms< DataTypes >::swapRemeshing(sofa::type::vector <EdgeID>& listEdges)
{
    //sofa::type::vector <EdgeID> edgeToSwap;
    bool allDone = false;
    bool test = true;

    while (!allDone && test)
    {
        allDone = true;
        test = false;

        for (unsigned int edgeIndex = 0; edgeIndex<listEdges.size() ; edgeIndex++)
        {
            const sofa::type::vector <TriangleID>& shell = m_container->getTrianglesAroundEdgeArray()[listEdges[edgeIndex]];

            if (shell.size() == 2)
            {
                sofa::type::vector <unsigned int> listVertex;
                const sofa::type::vector <PointID>& border = m_container->getPointsOnBorder();
                TriangleID indexTri1, indexTri2;

                indexTri1 = shell[0];
                indexTri2 = shell[1];

                int edgeInTri1 = m_container->getEdgeIndexInTriangle ( m_container->getEdgesInTriangle (indexTri1), listEdges[edgeIndex]);
                int edgeInTri2 = m_container->getEdgeIndexInTriangle ( m_container->getEdgesInTriangle (indexTri2), listEdges[edgeIndex]);
                Triangle vertexTriangle1 = m_container->getTriangleArray()[indexTri1];
                Triangle vertexTriangle2 = m_container->getTriangleArray()[indexTri2];

                listVertex.push_back( vertexTriangle1[edgeInTri1] );
                listVertex.push_back( vertexTriangle2[edgeInTri2] );
                listVertex.push_back( vertexTriangle1[ (edgeInTri1+1)%3 ] );
                listVertex.push_back( vertexTriangle2[ (edgeInTri2+1)%3 ] );

                int sum = 0;

                sum = int((m_container->getTrianglesAroundVertexArray()[ listVertex[0] ]).size());
                sum += int((m_container->getTrianglesAroundVertexArray()[ listVertex[1] ]).size());
                sum -= int((m_container->getTrianglesAroundVertexArray()[ listVertex[2] ]).size());
                sum -= int((m_container->getTrianglesAroundVertexArray()[ listVertex[3] ]).size());

                for (unsigned int i = 0; i <2; i++)
                {
                    for (unsigned int j = 0; j <border.size(); j++)
                    {
                        if(listVertex[i] == border[j])
                        {
                            sum+=2;
                            break;
                        }
                    }
                }

                for (unsigned int i = 2; i <4; i++)
                {
                    for (unsigned int j = 0; j <border.size(); j++)
                    {
                        if(listVertex[i] == border[j])
                        {
                            sum-=2;
                            break;
                        }
                    }
                }

                if (sum < -2)
                {
                    //edgeToSwap.push_back (listEdges[edgeIndex]);
                    if (edgeSwap (listEdges[edgeIndex]))
                        test = true;
                    //	    m_modifier->propagateTopologicalChanges();
                    allDone = false;
                }
            }
        }

        //edgeSwapProcess (edgeToSwap);
        //edgeToSwap.clear();
    }
}










template<class DataTypes>
int ManifoldTriangleSetGeometryAlgorithms< DataTypes >::SplitAlongPath(Index pa, Coord& a, Index pb, Coord& b,
        sofa::type::vector< sofa::core::topology::TopologyElementType>& topoPath_list,
        sofa::type::vector<Index>& indices_list,
        sofa::type::vector< sofa::type::Vec<3, double> >& coords_list,
        sofa::type::vector<EdgeID>& new_edges, double epsilonSnapPath, double epsilonSnapBorder)
{

    /// force the creation of TrianglesAroundEdgeArray
    m_container->getTrianglesAroundEdgeArray();
    /// force the creation of TrianglesAroundVertexArray
    m_container->getTrianglesAroundVertexArray();

    int result = TriangleSetGeometryAlgorithms< DataTypes >::SplitAlongPath (pa, a, pb, b, topoPath_list, indices_list, coords_list, new_edges, epsilonSnapPath, epsilonSnapBorder);

    return result;
}






template<class DataTypes>
bool ManifoldTriangleSetGeometryAlgorithms< DataTypes >::InciseAlongEdgeList (const sofa::type::vector<Index>& edges, sofa::type::vector<Index>& new_points, sofa::type::vector<Index>& end_points, bool& reachBorder)
{
    //// STEP 1 - Incise with the TriangleSetGeometryAlgorithms function. Addremovetriangles() automatically reorder the mesh
    bool ok = TriangleSetGeometryAlgorithms< DataTypes >::InciseAlongEdgeList (edges, new_points, end_points, reachBorder);

    //// STEP 2 - Create a ROI of edges (without double) around the incision
    sofa::type::vector< PointID > listVertex;
    sofa::type::vector< EdgeID > listEdges;
    bool doublon;

    // Old vertices from the list of edges
    for (unsigned int i = 0; i<edges.size(); i++)
    {
        Edge theEdge = m_container->getEdge (edges[i]);

        for (unsigned int j = 0; j<2; j++)
        {
            const sofa::type::vector< PointID >& shell = m_container->getVerticesAroundVertex( theEdge[j] );

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
        const sofa::type::vector< PointID >& shell = m_container->getVerticesAroundVertex( i );

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
        const auto& shell = m_container->getEdgesAroundVertex (listVertex[i]);

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


    swapRemeshing (listEdges);

    return ok;
}


} // namespace topology

} // namespace component

} // namespace sofa
