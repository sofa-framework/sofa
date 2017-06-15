/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
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
using namespace sofa::core::topology;

void TriangleSetTopologyModifier::init()
{

    EdgeSetTopologyModifier::init();
    this->getContext()->get(m_container);
}

void TriangleSetTopologyModifier::reinit()
{
    const sofa::helper::vector <unsigned int>& vertexToBeRemoved = this->list_Out.getValue();


    sofa::helper::vector <unsigned int> trianglesToBeRemoved;
    const sofa::helper::vector<Triangle>& listTri = this->m_container->d_triangle.getValue();

    for (unsigned int i = 0; i<listTri.size(); ++i)
    {
        Triangle the_tri = listTri[i];
        bool find = false;
        for (unsigned int j = 0; j<3; ++j)
        {
            unsigned int the_point = the_tri[j];
            for (unsigned int k = 0; k<vertexToBeRemoved.size(); ++k)
                if (the_point == vertexToBeRemoved[k])
                {
                    find = true;
                    break;
                }

            if (find)
            {
                trianglesToBeRemoved.push_back(i);
                break;
            }
        }
    }

    this->removeItems(trianglesToBeRemoved);
}



void TriangleSetTopologyModifier::addTriangles(const sofa::helper::vector<Triangle> &triangles)
{
    unsigned int nTriangles = m_container->getNbTriangles();

    // Test if the topology will still fulfill the conditions if this triangles is added.
    if (addTrianglesPreconditions(triangles))
    {
        /// effectively add triangles in the topology container
        addTrianglesProcess(triangles);

        // Apply postprocessing to arrange the topology.
        addTrianglesPostProcessing(triangles);

        sofa::helper::vector<unsigned int> trianglesIndex;
        trianglesIndex.reserve(triangles.size());

        for (unsigned int i=0; i<triangles.size(); ++i)
            trianglesIndex.push_back(nTriangles+i);

        // add topology event in the stack of topological events
        addTrianglesWarning((unsigned int)triangles.size(), triangles, trianglesIndex);

        // inform other objects that the edges are already added
        propagateTopologicalChanges();
    }
    else
    {
        std::cout << " TriangleSetTopologyModifier::addTriangleProcess(), preconditions for adding this triangle are not fulfilled. " << std::endl;
    }
}


void TriangleSetTopologyModifier::addTriangles(const sofa::helper::vector<Triangle> &triangles,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &baryCoefs)
{
    unsigned int nTriangles = m_container->getNbTriangles();

    // Test if the topology will still fulfill the conditions if this triangles is added.
    if (addTrianglesPreconditions(triangles))
    {
        /// actually add triangles in the topology container
        addTrianglesProcess(triangles);

        // Apply postprocessing to arrange the topology.
        addTrianglesPostProcessing(triangles);

        sofa::helper::vector<unsigned int> trianglesIndex;
        trianglesIndex.reserve(triangles.size());

        for (unsigned int i=0; i<triangles.size(); ++i)
            trianglesIndex.push_back(nTriangles+i);

        // add topology event in the stack of topological events
        addTrianglesWarning((unsigned int)triangles.size(), triangles, trianglesIndex, ancestors, baryCoefs);

        // inform other objects that the edges are already added
        propagateTopologicalChanges();
    }
    else
    {
        std::cout << " TriangleSetTopologyModifier::addTriangleProcess(), preconditions for adding this triangle are not fulfilled. " << std::endl;
    }
}


void TriangleSetTopologyModifier::addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles)
{
    for(unsigned int i=0; i<triangles.size(); ++i)
        addTriangleProcess(triangles[i]); //add triangle one by one.
}


void TriangleSetTopologyModifier::addTriangleProcess(Triangle t)
{

#ifndef NDEBUG
    // check if the 3 vertices are different
    if((t[0]==t[1]) || (t[0]==t[2]) || (t[1]==t[2]) )
    {
        serr << "Error: [TriangleSetTopologyModifier::addTriangle] : invalid triangle: "
                << t[0] << ", " << t[1] << ", " << t[2] <<  sendl;
        //return;
    }

    // check if there already exists a triangle with the same indices
    // Important: getEdgeIndex creates the quad vertex shell array
    if(m_container->hasTrianglesAroundVertex())
    {
        int previd = m_container->getTriangleIndex(t[0],t[1],t[2]);
        if( previd != -1)
        {
            serr << "Error: [TriangleSetTopologyModifier::addTriangle] : Triangle "
                    << t[0] << ", " << t[1] << ", " << t[2] << " already exists with index " << previd << "." << sendl;
            //return;
        }
    }
#endif

    const unsigned int triangleIndex = m_container->getNumberOfTriangles();
    helper::WriteAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = m_container->d_triangle;

    if(m_container->hasTrianglesAroundVertex())
    {
        for(unsigned int j=0; j<3; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->getTrianglesAroundVertexForModification( t[j] );
            shell.push_back( triangleIndex );
        }
    }

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

        if(m_container->hasEdgesInTriangle())
        {
            m_container->m_edgesInTriangle.resize(triangleIndex+1);
            m_container->m_edgesInTriangle[triangleIndex][j]= edgeIndex;
        }

        if(m_container->hasTrianglesAroundEdge())
        {
            sofa::helper::vector< unsigned int > &shell = m_container->m_trianglesAroundEdge[m_container->m_edgesInTriangle[triangleIndex][j]];
            shell.push_back( triangleIndex );
        }
    }

    m_triangle.push_back(t);
}


void TriangleSetTopologyModifier::addTrianglesWarning(const unsigned int nTriangles,
        const sofa::helper::vector< Triangle >& trianglesList,
        const sofa::helper::vector< unsigned int >& trianglesIndexList)
{
    m_container->setTriangleTopologyToDirty();

    // Warning that triangles just got created
    TrianglesAdded *e = new TrianglesAdded(nTriangles, trianglesList, trianglesIndexList);
    addTopologyChange(e);
}


void TriangleSetTopologyModifier::addTrianglesWarning(const unsigned int nTriangles,
        const sofa::helper::vector< Triangle >& trianglesList,
        const sofa::helper::vector< unsigned int >& trianglesIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    m_container->setTriangleTopologyToDirty();

    // Warning that triangles just got created
    TrianglesAdded *e=new TrianglesAdded(nTriangles, trianglesList,trianglesIndexList,ancestors,baryCoefs);
    addTopologyChange(e);
}


void TriangleSetTopologyModifier::addPointsProcess(const unsigned int nPoints)
{
    // start by calling the parent's method.
    EdgeSetTopologyModifier::addPointsProcess( nPoints );

    // now update the local container structures.
    if(m_container->hasTrianglesAroundVertex())
        m_container->m_trianglesAroundVertex.resize( m_container->getNbPoints() );
}

void TriangleSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    if(!m_container->hasEdges())
    {
        m_container->createEdgeSetArray();
    }

    // start by calling the parent's method.
    EdgeSetTopologyModifier::addEdgesProcess( edges );

    if(m_container->hasTrianglesAroundEdge())
        m_container->m_trianglesAroundEdge.resize( m_container->m_trianglesAroundEdge.size() + edges.size() );
}




void TriangleSetTopologyModifier::removeItems(const sofa::helper::vector<unsigned int> &items)
{
    removeTriangles(items, true, true); // remove triangles
}


void TriangleSetTopologyModifier::removeTriangles(const sofa::helper::vector<unsigned int> &triangleIds,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{    
    sofa::helper::vector<unsigned int> triangleIds_filtered;
    for (unsigned int i = 0; i < triangleIds.size(); i++)
    {
        if( triangleIds[i] >= m_container->getNumberOfTriangles())
            std::cout << "Warning: TriangleSetTopologyModifier::removeTriangles: Triangle: "<< triangleIds[i] <<" is out of bound and won't be removed." << std::endl;
        else
            triangleIds_filtered.push_back(triangleIds[i]);
    }

    if (removeTrianglesPreconditions(triangleIds_filtered)) // Test if the topology will still fulfill the conditions if these triangles are removed.
    {
        /// add the topological changes in the queue
        removeTrianglesWarning(triangleIds_filtered);
        // inform other objects that the triangles are going to be removed
        propagateTopologicalChanges();
        // now destroy the old triangles.
        removeTrianglesProcess(triangleIds_filtered ,removeIsolatedEdges, removeIsolatedPoints);

        m_container->checkTopology();
    }
    else
    {
        std::cout << " TriangleSetTopologyModifier::removeItems(), preconditions for removal are not fulfilled. " << std::endl;
    }

}


void TriangleSetTopologyModifier::removeTrianglesWarning(sofa::helper::vector<unsigned int> &triangles)
{
    m_container->setTriangleTopologyToDirty();


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
        serr << "Error. [TriangleSetTopologyModifier::removeTrianglesProcess] triangle array is empty." << sendl;
#endif
        return;
    }


    if(m_container->hasEdges() && removeIsolatedEdges)
    {

        if(!m_container->hasEdgesInTriangle())
            m_container->createEdgesInTriangleArray();

        if(!m_container->hasTrianglesAroundEdge())
            m_container->createTrianglesAroundEdgeArray();
    }

    if(removeIsolatedPoints)
    {

        if(!m_container->hasTrianglesAroundVertex())
            m_container->createTrianglesAroundVertexArray();
    }

    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;
    helper::WriteAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = m_container->d_triangle;

    unsigned int lastTriangle = m_container->getNumberOfTriangles() - 1;
    for(unsigned int i = 0; i<indices.size(); ++i, --lastTriangle)
    {
        Triangle &t = m_triangle[ indices[i] ];
        Triangle &q = m_triangle[ lastTriangle ];

        if(m_container->hasTrianglesAroundVertex())
        {
            for(unsigned int j=0; j<3; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_trianglesAroundVertex[ t[j] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedPoints && shell.empty())
                    vertexToBeRemoved.push_back(t[j]);
            }
        }

        if(m_container->hasTrianglesAroundEdge())
        {
            for(unsigned int j=0; j<3; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_trianglesAroundEdge[ m_container->m_edgesInTriangle[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedEdges && shell.empty())
                    edgeToBeRemoved.push_back(m_container->m_edgesInTriangle[indices[i]][j]);
            }
        }

        if(indices[i] < lastTriangle)
        {

            if(m_container->hasTrianglesAroundVertex())
            {

                for(unsigned int j=0; j<3; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = m_container->m_trianglesAroundVertex[ q[j] ];
                    replace(shell.begin(), shell.end(), lastTriangle, indices[i]);
                }
            }

            if(m_container->hasTrianglesAroundEdge())
            {

                for(unsigned int j=0; j<3; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = m_container->m_trianglesAroundEdge[ m_container->m_edgesInTriangle[lastTriangle][j]];
                    replace(shell.begin(), shell.end(), lastTriangle, indices[i]);
                }
            }
        }

        // removes the edgesInTriangles from the edgesInTrianglesArray
        if(m_container->hasEdgesInTriangle())
        {

            m_container->m_edgesInTriangle[ indices[i] ] = m_container->m_edgesInTriangle[ lastTriangle ]; // overwriting with last valid value.
            m_container->m_edgesInTriangle.resize( lastTriangle ); // resizing to erase multiple occurence of the triangle.
        }

        // removes the triangle from the triangleArray
        m_triangle[ indices[i] ] = m_triangle[ lastTriangle ]; // overwriting with last valid value.
        m_triangle.resize( lastTriangle ); // resizing to erase multiple occurence of the triangle.
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

    /*
    #ifndef NDEBUG // TO BE REMOVED WHEN SURE.
    	Debug();
    #endif
    */
}



void TriangleSetTopologyModifier::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{

    // Note: this does not check if an edge is removed from an existing triangle (it should never happen)

    if(m_container->hasEdgesInTriangle()) // this method should only be called when edges exist
    {
        if(!m_container->hasTrianglesAroundEdge())
            m_container->createTrianglesAroundEdgeArray();

        unsigned int lastEdge = m_container->getNumberOfEdges() - 1;
        for(unsigned int i = 0; i < indices.size(); ++i, --lastEdge)
        {
            // updating the triangles connected to the edge replacing the removed one:
            // for all triangles connected to the last point
            for(sofa::helper::vector<unsigned int>::iterator itt = m_container->m_trianglesAroundEdge[lastEdge].begin();
                itt != m_container->m_trianglesAroundEdge[lastEdge].end(); ++itt)
            {
                unsigned int edgeIndex = m_container->getEdgeIndexInTriangle(m_container->m_edgesInTriangle[(*itt)], lastEdge);
                m_container->m_edgesInTriangle[(*itt)][edgeIndex] = indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_trianglesAroundEdge[ indices[i] ] = m_container->m_trianglesAroundEdge[ lastEdge ];
        }

        m_container->m_trianglesAroundEdge.resize( m_container->m_trianglesAroundEdge.size() - indices.size() );
    }

    // call the parent's method.
    EdgeSetTopologyModifier::removeEdgesProcess(indices, removeIsolatedItems);
}



void TriangleSetTopologyModifier::removePointsProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{

    if(m_container->hasTriangles())
    {
        if(!m_container->hasTrianglesAroundVertex())
            m_container->createTrianglesAroundVertexArray();

        helper::WriteAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = m_container->d_triangle;

        unsigned int lastPoint = m_container->getNbPoints() - 1;
        for(unsigned int i=0; i<indices.size(); ++i, --lastPoint)
        {
            // updating the triangles connected to the point replacing the removed one:
            // for all triangles connected to the last point

            sofa::helper::vector<unsigned int> &shell = m_container->m_trianglesAroundVertex[lastPoint];
            for(unsigned int j=0; j<shell.size(); ++j)
            {
                const unsigned int q = shell[j];
                for(unsigned int k=0; k<3; ++k)
                {
                    if(m_triangle[q][k] == lastPoint)
                        m_triangle[q][k] = indices[i];
                }
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_trianglesAroundVertex[ indices[i] ] = m_container->m_trianglesAroundVertex[ lastPoint ];
        }

        m_container->m_trianglesAroundVertex.resize( m_container->m_trianglesAroundVertex.size() - indices.size() );
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
        if(m_container->hasTrianglesAroundVertex())
        {
            sofa::helper::vector< sofa::helper::vector< unsigned int > > trianglesAroundVertex_cp = m_container->m_trianglesAroundVertex;
            for(unsigned int i=0; i<index.size(); ++i)
            {
                m_container->m_trianglesAroundVertex[i] = trianglesAroundVertex_cp[ index[i] ];
            }
        }
        helper::WriteAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = m_container->d_triangle;

        for(unsigned int i=0; i<m_triangle.size(); ++i)
        {
            m_triangle[i][0] = inv_index[ m_triangle[i][0] ];
            m_triangle[i][1] = inv_index[ m_triangle[i][1] ];
            m_triangle[i][2] = inv_index[ m_triangle[i][2] ];
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



void TriangleSetTopologyModifier::addRemoveTriangles( const unsigned int nTri2Add,
        const sofa::helper::vector< Triangle >& triangles2Add,
        const sofa::helper::vector< unsigned int >& trianglesIndex2Add,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        sofa::helper::vector< unsigned int >& trianglesIndex2remove)
{

    // Create all the triangles registered to be created
    this->addTrianglesProcess(triangles2Add); // WARNING called after the creation process by the method "addTrianglesProcess"

    // Warn for the creation of all the triangles registered to be created
    this->addTrianglesWarning (nTri2Add, triangles2Add, trianglesIndex2Add, ancestors, baryCoefs);

    // Propagate the topological changes *** not necessary ??
    //m_modifier->propagateTopologicalChanges();

    // Remove all the triangles registered to be removed
    this->removeTriangles(trianglesIndex2remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

}


void TriangleSetTopologyModifier::movePointsProcess (const sofa::helper::vector <unsigned int>& id,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs,
        const bool moveDOF)
{
    m_container->setTriangleTopologyToDirty();

    (void)moveDOF;
    const size_t nbrVertex = id.size();
    bool doublet;
    sofa::helper::vector< unsigned int > trianglesAroundVertex2Move;
    sofa::helper::vector< Triangle > trianglesArray;


    // Step 1/4 - Creating trianglesAroundVertex to moved due to moved points:
    for (unsigned int i = 0; i<nbrVertex; ++i)
    {
        const sofa::helper::vector <unsigned int>& trianglesAroundVertex = m_container->getTrianglesAroundVertex( id[i] );

        for (unsigned int j = 0; j<trianglesAroundVertex.size(); ++j)
        {
            doublet = false;

            for (unsigned int k =0; k<trianglesAroundVertex2Move.size(); ++k) //Avoid double
            {
                if (trianglesAroundVertex2Move[k] == trianglesAroundVertex[j])
                {
                    doublet = true;
                    break;
                }
            }

            if(!doublet)
                trianglesAroundVertex2Move.push_back (trianglesAroundVertex[j]);

        }
    }

    std::sort( trianglesAroundVertex2Move.begin(), trianglesAroundVertex2Move.end(), std::greater<unsigned int>() );


    // Step 2/4 - Create event to delete all elements before moving and propagate it:
    TrianglesMoved_Removing *ev1 = new TrianglesMoved_Removing (trianglesAroundVertex2Move);
    this->addTopologyChange(ev1);
    propagateTopologicalChanges();


    // Step 3/4 - Physically move all dof:
    PointSetTopologyModifier::movePointsProcess (id, ancestors, coefs);

    m_container->setTriangleTopologyToDirty();

    // Step 4/4 - Create event to recompute all elements concerned by moving and propagate it:

    // Creating the corresponding array of Triangles for ancestors
    for (unsigned int i = 0; i<trianglesAroundVertex2Move.size(); i++)
        trianglesArray.push_back (m_container->getTriangleArray()[ trianglesAroundVertex2Move[i] ]);

    TrianglesMoved_Adding *ev2 = new TrianglesMoved_Adding (trianglesAroundVertex2Move, trianglesArray);
    this->addTopologyChange(ev2); // This event should be propagated with global workflow
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


void TriangleSetTopologyModifier::propagateTopologicalEngineChanges()
{
    if (m_container->beginChange() == m_container->endChange()) // nothing to do if no event is stored
        return;

    if (!m_container->isTriangleTopologyDirty()) // triangle Data has not been touched
        return EdgeSetTopologyModifier::propagateTopologicalEngineChanges();

#ifndef NDEBUG
    std::cout << "triangles is dirty" << std::endl;
    std::cout << "TriangleSetTopologyModifier - Number of outputs for triangle array: " << m_container->m_enginesList.size() << std::endl;
#endif

    std::list<sofa::core::topology::TopologyEngine *>::iterator it;
 //   for ( it = m_container->m_enginesList.begin(); it!=m_container->m_enginesList.end(); ++it)
	 for ( it = m_container->m_topologyEngineList.begin(); it!=m_container->m_topologyEngineList.end(); ++it)
    {
        sofa::core::topology::TopologyEngine* topoEngine = (*it);
        if (topoEngine->isDirty())
        {
#ifndef NDEBUG
            std::cout << "TriangleSetTopologyModifier::performing: " << topoEngine->getName() << std::endl;
#endif
            topoEngine->update();
        }
    }

    m_container->cleanTriangleTopologyFromDirty();
    EdgeSetTopologyModifier::propagateTopologicalEngineChanges();
}

} // namespace topology

} // namespace component

} // namespace sofa


