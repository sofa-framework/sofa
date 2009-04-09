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
#include <sofa/component/topology/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/TriangleSetTopologyChange.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace topology
{
SOFA_DECL_CLASS(TriangleSetTopologyModifier)
int TriangleSetTopologyModifierClass = core::RegisterObject("Triangle set topology modifier")
        .add< TriangleSetTopologyModifier >()
        ;

using namespace std;
using namespace sofa::defaulttype;


void TriangleSetTopologyModifier::init()
{

    EdgeSetTopologyModifier::init();
    this->getContext()->get(m_container);
}



void TriangleSetTopologyModifier::addTriangleProcess(Triangle t)
{
    sofa::helper::vector <Triangle> triangles;
    triangles.push_back(t);

    if (addTrianglesPreconditions(triangles))// Test if the topology will still fullfil the conditions if this triangles is added.
    {
        addSingleTriangleProcess(t); // add the triangle
        addTrianglesPostProcessing(triangles); // Apply postprocessing to arrange the topology.
    }
    else
    {
        std::cout << " TriangleSetTopologyModifier::addTriangleProcess(), preconditions for adding this triangle are not fullfil. " << std::endl;
    }
}


void TriangleSetTopologyModifier::addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles)
{
    if (addTrianglesPreconditions(triangles)) // Test if the topology will still fullfil the conditions if these triangles are added.
    {
        m_container->m_triangle.reserve(m_container->m_triangle.size() + triangles.size());

        for(unsigned int i=0; i<triangles.size(); ++i)
        {
            addSingleTriangleProcess(triangles[i]); //add triangle one by one.
        }

        addTrianglesPostProcessing(triangles); // Apply postprocessing to arrange the topology.
    }
    else
    {
        std::cout << " TriangleSetTopologyModifier::addTrianglesProcess(), preconditions for adding these triangles are not fullfil. " << std::endl;
    }
}


void TriangleSetTopologyModifier::addSingleTriangleProcess(Triangle t)
{

#ifndef NDEBUG
    // check if the 3 vertices are different
    if((t[0]==t[1]) || (t[0]==t[2]) || (t[1]==t[2]) )
    {
        sout << "Error: [TriangleSetTopologyModifier::addTriangle] : invalid quad: "
                << t[0] << ", " << t[1] << ", " << t[2] <<  endl;
        return;
    }

    // check if there already exists a triangle with the same indices
    // Important: getEdgeIndex creates the quad vertex shell array
    if(m_container->hasTriangleVertexShell())
    {
        if(m_container->getTriangleIndex(t[0],t[1],t[2]) != -1)
        {
            sout << "Error: [TriangleSetTopologyModifier::addTriangle] : Triangle "
                    << t[0] << ", " << t[1] << ", " << t[2] << " already exists." << endl;
            return;
        }
    }
#endif

    const unsigned int triangleIndex = m_container->getNumberOfTriangles();

    if(m_container->hasTriangleVertexShell())
    {
        for(unsigned int j=0; j<3; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->getTriangleVertexShellForModification( t[j] );
            shell.push_back( triangleIndex );
        }
    }

    if(m_container->hasEdges())
    {
        for(unsigned int j=0; j<3; ++j)
        {
            int edgeIndex = m_container->getEdgeIndex(t[(j+1)%3], t[(j+2)%3]);

            if(edgeIndex == -1)
            {
                // first create the edges
                sofa::helper::vector< Edge > v(1);
                Edge e1 (t[(j+1)%3], t[(j+2)%3]);
                v[0] = e1;

                addEdgesProcess((const sofa::helper::vector< Edge > &) v);

                edgeIndex = m_container->getEdgeIndex(t[(j+1)%3],t[(j+2)%3]);
                sofa::helper::vector< unsigned int > edgeIndexList;
                edgeIndexList.push_back((unsigned int) edgeIndex);
                addEdgesWarning( v.size(), v, edgeIndexList);
            }

            if(m_container->hasTriangleEdges())
            {
                m_container->m_triangleEdge.resize(triangleIndex+1);
                m_container->m_triangleEdge[triangleIndex][j]= edgeIndex;
            }

            if(m_container->hasTriangleEdgeShell())
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_triangleEdgeShell[m_container->m_triangleEdge[triangleIndex][j]];
                shell.push_back( triangleIndex );
            }
        }
    }

    m_container->m_triangle.push_back(t);
}


void TriangleSetTopologyModifier::addTrianglesWarning(const unsigned int nTriangles,
        const sofa::helper::vector< Triangle >& trianglesList,
        const sofa::helper::vector< unsigned int >& trianglesIndexList)
{
    // Warning that quads just got created
    TrianglesAdded *e = new TrianglesAdded(nTriangles, trianglesList, trianglesIndexList);
    addTopologyChange(e);
}


void TriangleSetTopologyModifier::addTrianglesWarning(const unsigned int nTriangles,
        const sofa::helper::vector< Triangle >& trianglesList,
        const sofa::helper::vector< unsigned int >& trianglesIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that triangles just got created
    TrianglesAdded *e=new TrianglesAdded(nTriangles, trianglesList,trianglesIndexList,ancestors,baryCoefs);
    addTopologyChange(e);
}


void TriangleSetTopologyModifier::addPointsProcess(const unsigned int nPoints)
{
    // start by calling the parent's method.
    EdgeSetTopologyModifier::addPointsProcess( nPoints );

    // now update the local container structures.
    if(m_container->hasTriangleVertexShell())
        m_container->m_triangleVertexShell.resize( m_container->getNbPoints() );
}

void TriangleSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    if(!m_container->hasEdges())
    {
        m_container->createEdgeSetArray();
    }

    // start by calling the parent's method.
    EdgeSetTopologyModifier::addEdgesProcess( edges );

    if(m_container->hasTriangleEdgeShell())
        m_container->m_triangleEdgeShell.resize( m_container->m_triangleEdgeShell.size() + edges.size() );
}




void TriangleSetTopologyModifier::removeItems(sofa::helper::vector< unsigned int >& items)
{

    removeTriangles(items, true, true); // remove triangles
}


void TriangleSetTopologyModifier::removeTriangles(sofa::helper::vector< unsigned int >& triangles,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    for (unsigned int i = 0; i < triangles.size(); i++)
    {
        if( triangles[i] >= m_container->m_triangle.size())
        {
            std::cout << "Error: TriangleSetTopologyModifier::removeTriangles: Triangle: "<< triangles[i] <<" is out of bound" << std::endl;
            return;
        }
    }


    if (removeTrianglesPreconditions(triangles)) // Test if the topology will still fullfil the conditions if these triangles are removed.
    {
        /// add the topological changes in the queue
        removeTrianglesWarning(triangles);
        // inform other objects that the triangles are going to be removed
        propagateTopologicalChanges();
        // now destroy the old triangles.
        removeTrianglesProcess(  triangles ,removeIsolatedEdges, removeIsolatedPoints);

        m_container->checkTopology();
    }
    else
    {
        std::cout << " TriangleSetTopologyModifier::removeItems(), preconditions for removal are not fullfil. " << std::endl;
    }

}


void TriangleSetTopologyModifier::removeTrianglesWarning(sofa::helper::vector<unsigned int> &triangles)
{
    /// sort vertices to remove in a descendent order
    std::sort( triangles.begin(), triangles.end(), std::greater<unsigned int>() );

    // Warning that these triangles will be deleted
    TrianglesRemoved *e=new TrianglesRemoved(triangles);
    addTopologyChange(e);
}


void TriangleSetTopologyModifier::removeTrianglesProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{

    if(!m_container->hasTriangles()) // this method should only be called when triangles exist
    {
#ifndef NDEBUG
        sout << "Error. [TriangleSetTopologyModifier::removeTrianglesProcess] triangle array is empty." << endl;
#endif
        return;
    }


    if(m_container->hasEdges() && removeIsolatedEdges)
    {

        if(!m_container->hasTriangleEdges())
            m_container->createTriangleEdgeArray();

        if(!m_container->hasTriangleEdgeShell())
            m_container->createTriangleEdgeShellArray();
    }

    if(removeIsolatedPoints)
    {

        if(!m_container->hasTriangleVertexShell())
            m_container->createTriangleVertexShellArray();
    }

    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    unsigned int lastTriangle = m_container->getNumberOfTriangles() - 1;
    for(unsigned int i = 0; i<indices.size(); ++i, --lastTriangle)
    {
        Triangle &t = m_container->m_triangle[ indices[i] ];
        Triangle &q = m_container->m_triangle[ lastTriangle ];

        if(m_container->hasTriangleVertexShell())
        {
            for(unsigned int j=0; j<3; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_triangleVertexShell[ t[j] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedPoints && shell.empty())
                    vertexToBeRemoved.push_back(t[j]);
            }
        }

        if(m_container->hasTriangleEdgeShell())
        {
            for(unsigned int j=0; j<3; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_triangleEdgeShell[ m_container->m_triangleEdge[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedEdges && shell.empty())
                    edgeToBeRemoved.push_back(m_container->m_triangleEdge[indices[i]][j]);
            }
        }

        if(indices[i] < lastTriangle)
        {

            if(m_container->hasTriangleVertexShell())
            {

                for(unsigned int j=0; j<3; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = m_container->m_triangleVertexShell[ q[j] ];
                    replace(shell.begin(), shell.end(), lastTriangle, indices[i]);
                }
            }

            if(m_container->hasTriangleEdgeShell())
            {

                for(unsigned int j=0; j<3; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = m_container->m_triangleEdgeShell[ m_container->m_triangleEdge[lastTriangle][j]];
                    replace(shell.begin(), shell.end(), lastTriangle, indices[i]);
                }
            }
        }

        // removes the triangleEdges from the triangleEdgesArray
        if(m_container->hasTriangleEdges())
        {

            m_container->m_triangleEdge[ indices[i] ] = m_container->m_triangleEdge[ lastTriangle ]; // overwriting with last valid value.
            m_container->m_triangleEdge.resize( lastTriangle ); // resizing to erase multiple occurence of the triangle.
        }

        // removes the triangle from the triangleArray
        m_container->m_triangle[ indices[i] ] = m_container->m_triangle[ lastTriangle ]; // overwriting with last valid value.
        m_container->m_triangle.resize( lastTriangle ); // resizing to erase multiple occurence of the triangle.
    }


    removeTrianglesPostProcessing(edgeToBeRemoved, vertexToBeRemoved); // Arrange the current topology.

    if(!edgeToBeRemoved.empty())
    {

        /// warn that edges will be deleted
        removeEdgesWarning(edgeToBeRemoved);
        propagateTopologicalChanges();
        /// actually remove edges without looking for isolated vertices
        removeEdgesProcess(edgeToBeRemoved, false);
    }

    if(!vertexToBeRemoved.empty())
    {

        removePointsWarning(vertexToBeRemoved);
        /// propagate to all components
        propagateTopologicalChanges();
        removePointsProcess(vertexToBeRemoved);
    }


#ifndef NDEBUG // TO BE REMOVED WHEN SURE.
    Debug();
#endif
}



void TriangleSetTopologyModifier::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{

    // Note: this does not check if an edge is removed from an existing triangle (it should never happen)

    if(m_container->hasTriangleEdges()) // this method should only be called when edges exist
    {
        if(!m_container->hasTriangleEdgeShell())
            m_container->createTriangleEdgeShellArray();

        unsigned int lastEdge = m_container->getNumberOfEdges() - 1;
        for(unsigned int i = 0; i < indices.size(); ++i, --lastEdge)
        {
            // updating the triangles connected to the edge replacing the removed one:
            // for all triangles connected to the last point
            for(sofa::helper::vector<unsigned int>::iterator itt = m_container->m_triangleEdgeShell[lastEdge].begin();
                itt != m_container->m_triangleEdgeShell[lastEdge].end(); ++itt)
            {
                unsigned int edgeIndex = m_container->getEdgeIndexInTriangle(m_container->m_triangleEdge[(*itt)], lastEdge);
                m_container->m_triangleEdge[(*itt)][edgeIndex] = indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_triangleEdgeShell[ indices[i] ] = m_container->m_triangleEdgeShell[ lastEdge ];
        }

        m_container->m_triangleEdgeShell.resize( m_container->m_triangleEdgeShell.size() - indices.size() );
    }

    // call the parent's method.
    EdgeSetTopologyModifier::removeEdgesProcess(indices, removeIsolatedItems);
}



void TriangleSetTopologyModifier::removePointsProcess( sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{

    if(m_container->hasTriangles())
    {
        if(!m_container->hasTriangleVertexShell())
            m_container->createTriangleVertexShellArray();

        unsigned int lastPoint = m_container->getNbPoints() - 1;
        for(unsigned int i=0; i<indices.size(); ++i, --lastPoint)
        {
            // updating the triangles connected to the point replacing the removed one:
            // for all triangles connected to the last point

            sofa::helper::vector<unsigned int> &shell = m_container->m_triangleVertexShell[lastPoint];
            for(unsigned int j=0; j<shell.size(); ++j)
            {
                const unsigned int q = shell[j];
                for(unsigned int k=0; k<3; ++k)
                {
                    if(m_container->m_triangle[q][k] == lastPoint)
                        m_container->m_triangle[q][k] = indices[i];
                }
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_triangleVertexShell[ indices[i] ] = m_container->m_triangleVertexShell[ lastPoint ];
        }

        m_container->m_triangleVertexShell.resize( m_container->m_triangleVertexShell.size() - indices.size() );
    }

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent's method.
    EdgeSetTopologyModifier::removePointsProcess( indices, removeDOF );
}


void TriangleSetTopologyModifier::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{

    if(m_container->hasTriangles())
    {
        if(m_container->hasTriangleVertexShell())
        {
            sofa::helper::vector< sofa::helper::vector< unsigned int > > triangleVertexShell_cp = m_container->m_triangleVertexShell;
            for(unsigned int i=0; i<index.size(); ++i)
            {
                m_container->m_triangleVertexShell[i] = triangleVertexShell_cp[ index[i] ];
            }
        }

        for(unsigned int i=0; i<m_container->m_triangle.size(); ++i)
        {
            m_container->m_triangle[i][0] = inv_index[ m_container->m_triangle[i][0] ];
            m_container->m_triangle[i][1] = inv_index[ m_container->m_triangle[i][1] ];
            m_container->m_triangle[i][2] = inv_index[ m_container->m_triangle[i][2] ];
        }
    }

    // call the parent's method
    EdgeSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}




void TriangleSetTopologyModifier::renumberPoints( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index)
{

    /// add the topological changes in the queue
    renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    propagateTopologicalChanges();
    // now renumber the points
    renumberPointsProcess(index, inv_index);

    m_container->checkTopology();
}


bool TriangleSetTopologyModifier::removeTrianglesPreconditions(const sofa::helper::vector< unsigned int >& items)
{
    (void)items;
    return true;
}

void TriangleSetTopologyModifier::removeTrianglesPostProcessing(const sofa::helper::vector< unsigned int >& edgeToBeRemoved, const sofa::helper::vector< unsigned int >& vertexToBeRemoved )
{
    (void)vertexToBeRemoved;
    (void)edgeToBeRemoved;
}


bool TriangleSetTopologyModifier::addTrianglesPreconditions(const sofa::helper::vector <Triangle>& triangles)
{
    (void)triangles;
    return true;
}

void TriangleSetTopologyModifier::addTrianglesPostProcessing(const sofa::helper::vector <Triangle>& triangles)
{
    (void)triangles;
}

} // namespace topology

} // namespace component

} // namespace sofa


