/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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


TriangleSetTopologyContainer* TriangleSetTopologyModifier::getTriangleSetTopologyContainer() const
{
    return static_cast<TriangleSetTopologyContainer* > (this->m_topologyContainer);
}


void TriangleSetTopologyModifier::addTriangle(Triangle t)
{
    TriangleSetTopologyContainer *container = getTriangleSetTopologyContainer();

#ifndef NDEBUG
    // check if the 3 vertices are different
    if((t[0]==t[1]) || (t[0]==t[2]) || (t[1]==t[2]) )
    {
        cout << "Error: [TriangleSetTopologyModifier::addTriangle] : invalid quad: "
                << t[0] << ", " << t[1] << ", " << t[2] <<  endl;

        return;
    }

    // check if there already exists a triangle with the same indices
    // Important: getEdgeIndex creates the quad vertex shell array
    if(container->hasTriangleVertexShell())
    {
        if(container->getTriangleIndex(t[0],t[1],t[2]) != -1)
        {
            cout << "Error: [TriangleSetTopologyModifier::addTriangle] : Triangle "
                    << t[0] << ", " << t[1] << ", " << t[2] << " already exists." << endl;
            return;
        }
    }
#endif

    const unsigned int triangleIndex = container->m_triangle.size();

    if(container->hasEdges())
    {
        for(unsigned int j=0; j<3; ++j)
        {
            int edgeIndex = container->getEdgeIndex(t[(j+1)%3], t[(j+2)%3]);

            if(edgeIndex == -1)
            {
                // first create the edges
                sofa::helper::vector< Edge > v(1);
                Edge e1 (t[(j+1)%3], t[(j+2)%3]);
                v[0] = e1;

                addEdgesProcess((const sofa::helper::vector< Edge > &) v);

                edgeIndex = container->getEdgeIndex(t[(j+1)%3],t[(j+2)%3]);
                sofa::helper::vector< unsigned int > edgeIndexList;
                edgeIndexList.push_back((unsigned int) edgeIndex);
                this->addEdgesWarning( v.size(), v, edgeIndexList);
            }

            if(container->hasTriangleEdges())
            {
                container->m_triangleEdge.resize(triangleIndex+1);
                container->m_triangleEdge[triangleIndex][j]= edgeIndex;
            }

            if(container->hasTriangleEdgeShell())
            {
                sofa::helper::vector< unsigned int > &shell = container->m_triangleEdgeShell[container->m_triangleEdge[triangleIndex][j]];
                shell.push_back( triangleIndex );
            }
        }
    }

    if(container->hasTriangleVertexShell())
    {
        for(unsigned int j=0; j<3; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = container->getTriangleVertexShellForModification( t[j] );
            shell.push_back( triangleIndex );
        }
    }

    container->m_triangle.push_back(t);
}


void TriangleSetTopologyModifier::addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles)
{
    TriangleSetTopologyContainer *container = getTriangleSetTopologyContainer();
    container->m_triangle.reserve(container->m_triangle.size() + triangles.size());

    for(unsigned int i=0; i<triangles.size(); ++i)
    {
        addTriangle(triangles[i]);
    }
}


void TriangleSetTopologyModifier::addTrianglesWarning(const unsigned int nTriangles,
        const sofa::helper::vector< Triangle >& trianglesList,
        const sofa::helper::vector< unsigned int >& trianglesIndexList)
{
    // Warning that quads just got created
    TrianglesAdded *e = new TrianglesAdded(nTriangles, trianglesList, trianglesIndexList);

    this->addTopologyChange(e);
}


void TriangleSetTopologyModifier::addTrianglesWarning(const unsigned int nTriangles,
        const sofa::helper::vector< Triangle >& trianglesList,
        const sofa::helper::vector< unsigned int >& trianglesIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that triangles just got created
    TrianglesAdded *e=new TrianglesAdded(nTriangles, trianglesList,trianglesIndexList,ancestors,baryCoefs);

    this->addTopologyChange(e);
}


void TriangleSetTopologyModifier::removeTrianglesWarning( sofa::helper::vector<unsigned int> &triangles)
{
    /// sort vertices to remove in a descendent order
    std::sort( triangles.begin(), triangles.end(), std::greater<unsigned int>() );

    // Warning that these triangles will be deleted
    TrianglesRemoved *e=new TrianglesRemoved(triangles);

    this->addTopologyChange(e);
}


void TriangleSetTopologyModifier::removeTrianglesProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    TriangleSetTopologyContainer *container = getTriangleSetTopologyContainer();

    if(!container->hasTriangles()) // TODO : this method should only be called when triangles exist
    {
#ifndef NDEBUG
        cout << "Error. [TriangleSetTopologyModifier::removeTrianglesProcess] triangle array is empty." << endl;
#endif
        container->createTriangleSetArray();
    }

    if(container->hasEdges() && removeIsolatedEdges)
    {
        if(!container->hasTriangleEdges())
            container->createTriangleEdgeArray();

        if(!container->hasTriangleEdgeShell())
            container->createTriangleEdgeShellArray();
    }

    if(removeIsolatedPoints)
    {
        if(!container->hasTriangleVertexShell())
            container->createTriangleVertexShellArray();
    }

    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    for(unsigned int i = 0; i<indices.size(); ++i)
    {
        const unsigned int lastTriangle = container->m_triangle.size() - 1;
        Triangle &t = container->m_triangle[ indices[i] ];
        Triangle &q = container->m_triangle[ lastTriangle ];

        if(container->hasTriangleVertexShell())
        {
            for(unsigned int j=0; j<3; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_triangleVertexShell[ t[j] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if((removeIsolatedPoints) && shell.empty())
                    vertexToBeRemoved.push_back(t[j]);
            }
        }

        if(container->hasTriangleEdgeShell())
        {
            for(unsigned int j=0; j<3; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if((removeIsolatedEdges) && shell.empty())
                    edgeToBeRemoved.push_back(container->m_triangleEdge[indices[i]][j]);
            }
        }

        // now updates the shell information of the triangle at the end of the array
        if(indices[i] < lastTriangle)
        {
            if(container->hasTriangleVertexShell())
            {
                for(unsigned int j=0; j<3; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = container->m_triangleVertexShell[ q[j] ];
                    replace(shell.begin(), shell.end(), lastTriangle, indices[i]);
                }
            }

            if(container->hasTriangleEdgeShell())
            {
                for(unsigned int j=0; j<3; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = container->m_triangleEdgeShell[ container->m_triangleEdge[lastTriangle][j]];
                    replace(shell.begin(), shell.end(), lastTriangle, indices[i]);
                }
            }
        }

        // removes the triangleEdges from the triangleEdgesArray
        if(container->hasTriangleEdges())
        {
            container->m_triangleEdge[ indices[i] ] = container->m_triangleEdge[ lastTriangle ]; // overwriting with last valid value.
            container->m_triangleEdge.resize( lastTriangle ); // resizing to erase multiple occurence of the triangle.
        }

        // removes the triangle from the triangleArray
        container->m_triangle[ indices[i] ] = container->m_triangle[ lastTriangle ]; // overwriting with last valid value.
        container->m_triangle.resize( lastTriangle ); // resizing to erase multiple occurence of the triangle.
    }

    if(!edgeToBeRemoved.empty())
    {
        /// warn that edges will be deleted
        this->removeEdgesWarning(edgeToBeRemoved);
        container->propagateTopologicalChanges();
        /// actually remove edges without looking for isolated vertices
        this->removeEdgesProcess(edgeToBeRemoved, false);
    }

    if(!vertexToBeRemoved.empty())
    {
        this->removePointsWarning(vertexToBeRemoved);
        /// propagate to all components
        container->propagateTopologicalChanges();
        this->removePointsProcess(vertexToBeRemoved);
    }
}

void TriangleSetTopologyModifier::addPointsProcess(const unsigned int nPoints, const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if edges exist, otherwise call the PointSet method
    EdgeSetTopologyModifier::addPointsProcess( nPoints, addDOF );

    // now update the local container structures.
    TriangleSetTopologyContainer *container = getTriangleSetTopologyContainer();

    if(container->hasTriangleVertexShell())
        container->m_triangleVertexShell.resize( container->m_triangleVertexShell.size() + nPoints );
}

void TriangleSetTopologyModifier::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if edges exist, otherwise call the PointSet method
    EdgeSetTopologyModifier::addPointsProcess( nPoints, ancestors, baryCoefs, addDOF );

    // now update the local container structures.
    TriangleSetTopologyContainer *container = getTriangleSetTopologyContainer();

    if(container->hasTriangleVertexShell())
        container->m_triangleVertexShell.resize( container->m_triangleVertexShell.size() + nPoints );
}

void TriangleSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // now update the local container structures.
    TriangleSetTopologyContainer *container = getTriangleSetTopologyContainer();

    if(!container->hasEdges())
    {
        container->createEdgeSetArray();
    }

    // start by calling the parent's method.
    EdgeSetTopologyModifier::addEdgesProcess( edges );

    if(container->hasTriangleEdgeShell())
        container->m_triangleEdgeShell.resize( container->m_triangleEdgeShell.size() + edges.size() );
}

void TriangleSetTopologyModifier::removePointsProcess( sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    TriangleSetTopologyContainer *container = getTriangleSetTopologyContainer();

    // force the creation of the triangle vertex shell array before any point is deleted
    if(!container->hasTriangleVertexShell())
        container->createTriangleVertexShellArray();

    unsigned int lastPoint = container->getNbPoints() - 1;
    for(unsigned int i=0; i<indices.size(); ++i, --lastPoint)
    {
        // updating the triangles connected to the point replacing the removed one:
        // for all triangles connected to the last point

        sofa::helper::vector<unsigned int> &shell = container->m_triangleVertexShell[lastPoint];
        for(unsigned int j=0; j<shell.size(); ++j)
        {
            const unsigned int q = shell[j];
            for(unsigned int k=0; k<3; ++k)
            {
                if(container->m_triangle[q][k] == lastPoint)
                    container->m_triangle[q][k] = indices[i];
            }
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_triangleVertexShell[ indices[i] ] = container->m_triangleVertexShell[ lastPoint ];
    }

    container->m_triangleVertexShell.resize( container->m_triangleVertexShell.size() - indices.size() );

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent's method.
    // TODO : only if edges exist, otherwise call PointSetMethod
    EdgeSetTopologyModifier::removePointsProcess( indices, removeDOF );
}

void TriangleSetTopologyModifier::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    // TODO : clarify what exactly has to happen here (what if an edge is removed from an existing triangle?)

    // now update the local container structures
    TriangleSetTopologyContainer *container = getTriangleSetTopologyContainer();

    if(!container->hasEdges()) // TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyModifier::removeEdgesProcess] edge array is empty." << endl;
#endif
        container->createEdgeSetArray();
    }

    if(!container->hasTriangleEdgeShell())
        container->createTriangleEdgeShellArray();

    if(!container->hasTriangleEdges())
        container->createTriangleEdgeArray();

    unsigned int lastEdge = container->getNumberOfEdges() - 1;
    for(unsigned int i = 0; i < indices.size(); ++i, --lastEdge)
    {
        // updating the triangles connected to the edge replacing the removed one:
        // for all triangles connected to the last point
        for(sofa::helper::vector<unsigned int>::iterator itt = container->m_triangleEdgeShell[lastEdge].begin();
            itt != container->m_triangleEdgeShell[lastEdge].end(); ++itt)
        {
            int edgeIndex = container->getEdgeIndexInTriangle(container->m_triangleEdge[(*itt)], lastEdge);
            assert((int)edgeIndex!= -1);
            container->m_triangleEdge[(*itt)][(unsigned int) edgeIndex] = indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_triangleEdgeShell[ indices[i] ] = container->m_triangleEdgeShell[ lastEdge ];
    }

    container->m_triangleEdgeShell.resize( container->m_triangleEdgeShell.size() - indices.size() );

    // call the parent's method.
    EdgeSetTopologyModifier::removeEdgesProcess(indices, removeIsolatedItems);
}

void TriangleSetTopologyModifier::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    // now update the local container structures.
    TriangleSetTopologyContainer *container = getTriangleSetTopologyContainer();

    if(!container->hasTriangles()) // TODO : this method should only be called when triangles exist
    {
#ifndef NDEBUG
        cout << "Error. [TriangleSetTopologyModifier::renumberPointsProcess] triangle array is empty." << endl;
#endif
        container->createTriangleSetArray();
    }

    if(container->hasTriangleVertexShell())
    {
        sofa::helper::vector< sofa::helper::vector< unsigned int > > triangleVertexShell_cp = container->m_triangleVertexShell;
        for(unsigned int i=0; i<index.size(); ++i)
        {
            container->m_triangleVertexShell[i] = triangleVertexShell_cp[ index[i] ];
        }
    }

    for(unsigned int i=0; i<container->m_triangle.size(); ++i)
    {
        container->m_triangle[i][0] = inv_index[ container->m_triangle[i][0] ];
        container->m_triangle[i][1] = inv_index[ container->m_triangle[i][1] ];
        container->m_triangle[i][2] = inv_index[ container->m_triangle[i][2] ];
    }

    // call the parent's method
    if(container->hasEdges())
        EdgeSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
    else
        PointSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}

} // namespace topology

} // namespace component

} // namespace sofa
