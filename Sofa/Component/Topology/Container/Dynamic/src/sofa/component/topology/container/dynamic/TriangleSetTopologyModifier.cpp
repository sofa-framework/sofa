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
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>
#include <sofa/core/topology/TopologyHandler.h>

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>

#include <algorithm>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::topology::container::dynamic
{
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

    if(!m_container)
    {
        msg_error() << "TriangleSetTopologyContainer not found in current node: " << this->getContext()->getName();
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
}

void TriangleSetTopologyModifier::reinit()
{
    const sofa::type::vector<TriangleID>& vertexToBeRemoved = this->list_Out.getValue();


    sofa::type::vector<TriangleID> trianglesToBeRemoved;
    const sofa::type::vector<Triangle>& listTri = this->m_container->d_triangle.getValue();

    for (size_t i = 0; i<listTri.size(); ++i)
    {
        Triangle the_tri = listTri[i];
        bool find = false;
        for (unsigned int j = 0; j<3; ++j)
        {
            const PointID the_point = the_tri[j];
            for (size_t k = 0; k<vertexToBeRemoved.size(); ++k)
                if (the_point == vertexToBeRemoved[k])
                {
                    find = true;
                    break;
                }

            if (find)
            {
                trianglesToBeRemoved.push_back((TriangleID)i);
                break;
            }
        }
    }

    this->removeItems(trianglesToBeRemoved);
}



void TriangleSetTopologyModifier::addTriangles(const sofa::type::vector<Triangle> &triangles)
{
    const size_t nTriangles = m_container->getNbTriangles();

    // Test if the topology will still fulfill the conditions if this triangles is added.
    if (addTrianglesPreconditions(triangles))
    {
        /// effectively add triangles in the topology container
        addTrianglesProcess(triangles);

        // Apply postprocessing to arrange the topology.
        addTrianglesPostProcessing(triangles);

        sofa::type::vector<TriangleID> trianglesIndex;
        trianglesIndex.reserve(triangles.size());

        for (size_t i=0; i<triangles.size(); ++i)
            trianglesIndex.push_back((TriangleID)(nTriangles+i));

        // add topology event in the stack of topological events
        addTrianglesWarning(sofa::Size(triangles.size()), triangles, trianglesIndex);

        // inform other objects that the edges are already added
        propagateTopologicalChanges();

        m_container->checkTopology();
    }
    else
    {
        msg_error() << " TriangleSetTopologyModifier::addTriangleProcess(), preconditions for adding this triangle are not fulfilled. ";
    }
}


void TriangleSetTopologyModifier::addTriangles(const sofa::type::vector<Triangle> &triangles,
        const sofa::type::vector<sofa::type::vector<TriangleID> > &ancestors,
        const sofa::type::vector<sofa::type::vector<SReal> > &baryCoefs)
{
    const size_t nTriangles = m_container->getNbTriangles();

    // Test if the topology will still fulfill the conditions if this triangles is added.
    if (addTrianglesPreconditions(triangles))
    {
        /// actually add triangles in the topology container
        addTrianglesProcess(triangles);

        // Apply postprocessing to arrange the topology.
        addTrianglesPostProcessing(triangles);

        sofa::type::vector<TriangleID> trianglesIndex;
        trianglesIndex.reserve(triangles.size());

        for (size_t i=0; i<triangles.size(); ++i)
            trianglesIndex.push_back((TriangleID)(nTriangles+i));

        // add topology event in the stack of topological events
        addTrianglesWarning(sofa::Size(triangles.size()), triangles, trianglesIndex, ancestors, baryCoefs);

        // inform other objects that the edges are already added
        propagateTopologicalChanges();

        m_container->checkTopology();
    }
    else
    {
		msg_error() << "Preconditions for adding this triangle are not fulfilled. ";
    }
}


void TriangleSetTopologyModifier::addTrianglesProcess(const sofa::type::vector< Triangle > &triangles)
{
    for(size_t i=0; i<triangles.size(); ++i)
        addTriangleProcess(triangles[i]); //add triangle one by one.
}


void TriangleSetTopologyModifier::addTriangleProcess(Triangle t)
{

	if (m_container->d_checkTopology.getValue())
	{
		// check if the 3 vertices are different
		if ((t[0] == t[1]) || (t[0] == t[2]) || (t[1] == t[2]))
		{
			msg_error() << "Invalid triangle: "	<< t[0] << ", " << t[1] << ", " << t[2];
		}

		// check if there already exists a triangle with the same indices
		// Important: getEdgeIndex creates the quad vertex shell array
		if (m_container->hasTrianglesAroundVertex())
		{
            const TriangleID previd = m_container->getTriangleIndex(t[0], t[1], t[2]);
            if (previd != sofa::InvalidID)
			{
				msg_error() << "Triangle " << t[0] << ", " << t[1] << ", " << t[2] << " already exists with index " << previd << ".";
			}
		}
	}

    const TriangleID triangleIndex = (TriangleID)m_container->getNumberOfTriangles();
    helper::WriteAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = m_container->d_triangle;

    // update nbr point if needed
    unsigned int nbrP = m_container->getNbPoints();
    for(unsigned int i=0; i<3; ++i)
        if (t[i] + 1 > nbrP) // point not well init
        {
            nbrP = t[i] + 1;
            m_container->setNbPoints(nbrP);
        }

    // update m_trianglesAroundVertex
    if (m_container->m_trianglesAroundVertex.size() < nbrP)
        m_container->m_trianglesAroundVertex.resize(nbrP);

    for(unsigned int j=0; j<3; ++j)
    {
        sofa::type::vector< TriangleID > &shell = m_container->m_trianglesAroundVertex[t[j]];
        shell.push_back( triangleIndex );
    }


    // update edge-triangle cross buffers
    if (m_container->m_edgesInTriangle.size() < triangleIndex+1)
        m_container->m_edgesInTriangle.resize(triangleIndex+1);

    for(unsigned int j=0; j<3; ++j)
    {
        EdgeID edgeIndex = m_container->getEdgeIndex(t[(j+1)%3], t[(j+2)%3]);

        if(edgeIndex == sofa::InvalidID)
        {
            // first create the edges
            sofa::type::vector< Edge > v(1);
            Edge e1 (t[(j+1)%3], t[(j+2)%3]);
            v[0] = e1;

            addEdgesProcess((const sofa::type::vector< Edge > &) v);

            edgeIndex = m_container->getEdgeIndex(t[(j+1)%3],t[(j+2)%3]);
            assert (edgeIndex != sofa::InvalidID);
            if (edgeIndex == sofa::InvalidID)
            {
                msg_error() << "Edge creation: " << e1 << " failed in addTriangleProcess. Edge will not be added in buffers.";
                continue;
            }

            sofa::type::vector< EdgeID > edgeIndexList;
            edgeIndexList.push_back((EdgeID) edgeIndex);
            addEdgesWarning(sofa::Size(v.size()), v, edgeIndexList);
        }

        // update m_edgesInTriangle
        m_container->m_edgesInTriangle[triangleIndex][j]= edgeIndex;

        // update m_trianglesAroundEdge
        if (m_container->m_trianglesAroundEdge.size() < m_container->getNbEdges())
            m_container->m_trianglesAroundEdge.resize(m_container->getNbEdges());

        sofa::type::vector< TriangleID > &shell = m_container->m_trianglesAroundEdge[edgeIndex];
        shell.push_back( triangleIndex );
    }

    m_triangle.push_back(t);
}


void TriangleSetTopologyModifier::addTrianglesWarning(const sofa::Size nTriangles,
        const sofa::type::vector< Triangle >& trianglesList,
        const sofa::type::vector< TriangleID >& trianglesIndexList)
{
    m_container->setTriangleTopologyToDirty();

    // Warning that triangles just got created
    const TrianglesAdded *e = new TrianglesAdded(nTriangles, trianglesList, trianglesIndexList);
    addTopologyChange(e);
}


void TriangleSetTopologyModifier::addTrianglesWarning(const sofa::Size nTriangles,
        const sofa::type::vector< Triangle >& trianglesList,
        const sofa::type::vector< TriangleID >& trianglesIndexList,
        const sofa::type::vector< sofa::type::vector< TriangleID > > & ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
{
    m_container->setTriangleTopologyToDirty();

    // Warning that triangles just got created
    const TrianglesAdded *e=new TrianglesAdded(nTriangles, trianglesList,trianglesIndexList,ancestors,baryCoefs);
    addTopologyChange(e);
}


void TriangleSetTopologyModifier::addPointsProcess(const sofa::Size nPoints)
{
    // start by calling the parent's method.
    EdgeSetTopologyModifier::addPointsProcess( nPoints );

    // now update the local container structures.
    if(m_container->hasTrianglesAroundVertex())
        m_container->m_trianglesAroundVertex.resize( m_container->getNbPoints() );
}

void TriangleSetTopologyModifier::addEdgesProcess(const sofa::type::vector< Edge > &edges)
{
    // start by calling the parent's method.
    EdgeSetTopologyModifier::addEdgesProcess( edges );

    if(m_container->hasTrianglesAroundEdge())
        m_container->m_trianglesAroundEdge.resize( m_container->m_trianglesAroundEdge.size() + edges.size() );
}

void TriangleSetTopologyModifier::removeItems(const sofa::type::vector<TriangleID> &items)
{
    removeTriangles(items, true, true); // remove triangles
}


void TriangleSetTopologyModifier::removeTriangles(const sofa::type::vector<TriangleID> &triangleIds,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    SCOPED_TIMER_VARNAME(removeTrianglesTimer, "removeTriangles");

    sofa::type::vector<TriangleID> triangleIds_filtered;
    for (size_t i = 0; i < triangleIds.size(); i++)
    {
        if( triangleIds[i] >= m_container->getNumberOfTriangles())
            dmsg_warning() << "RemoveTriangles: Triangle: "<< triangleIds[i] <<" is out of bound and won't be removed.";
        else
            triangleIds_filtered.push_back(triangleIds[i]);
    }

    if (removeTrianglesPreconditions(triangleIds_filtered)) // Test if the topology will still fulfill the conditions if these triangles are removed.
    {
        /// add the topological changes in the queue
        {
            SCOPED_TIMER("removeTrianglesWarning");
            removeTrianglesWarning(triangleIds_filtered);
        }

        // inform other objects that the triangles are going to be removed
        {
            SCOPED_TIMER("propagateTopologicalChanges");
            propagateTopologicalChanges();
        }

        // now destroy the old triangles.
        {
            SCOPED_TIMER("removeTrianglesProcess");
            removeTrianglesProcess(triangleIds_filtered ,removeIsolatedEdges, removeIsolatedPoints);
        }

        m_container->checkTopology();
    }
    else
    {
		msg_warning() << "Preconditions for removal are not fulfilled. ";
    }
}


void TriangleSetTopologyModifier::removeTrianglesWarning(sofa::type::vector<TriangleID> &triangles)
{
    m_container->setTriangleTopologyToDirty();


    /// sort vertices to remove in a descendent order
    std::sort( triangles.begin(), triangles.end(), std::greater<TriangleID>() );

    // Warning that these triangles will be deleted
    const TrianglesRemoved *e=new TrianglesRemoved(triangles);
    addTopologyChange(e);
}


void TriangleSetTopologyModifier::removeTrianglesProcess(const sofa::type::vector<TriangleID> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{

    if(!m_container->hasTriangles()) // this method should only be called when triangles exist
    {
		msg_error() << "Triangle array is empty.";
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

    sofa::type::vector<EdgeID> edgeToBeRemoved;
    sofa::type::vector<PointID> vertexToBeRemoved;
    helper::WriteAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = m_container->d_triangle;

    size_t lastTriangle = m_container->getNumberOfTriangles() - 1;
    for(size_t i = 0; i<indices.size(); ++i, --lastTriangle)
    {
        Triangle &t = m_triangle[ indices[i] ];
        Triangle &q = m_triangle[ lastTriangle ];

        if(m_container->hasTrianglesAroundVertex())
        {
            for(unsigned int j=0; j<3; ++j)
            {
                sofa::type::vector< TriangleID > &shell = m_container->m_trianglesAroundVertex[ t[j] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedPoints && shell.empty())
                    vertexToBeRemoved.push_back(t[j]);
            }
        }

        if(m_container->hasTrianglesAroundEdge())
        {
            for(unsigned int j=0; j<3; ++j)
            {
                sofa::type::vector< TriangleID > &shell = m_container->m_trianglesAroundEdge[ m_container->m_edgesInTriangle[indices[i]][j]];
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
                    sofa::type::vector< TriangleID > &shell = m_container->m_trianglesAroundVertex[ q[j] ];
                    replace(shell.begin(), shell.end(), (TriangleID)lastTriangle, indices[i]);
                }
            }

            if(m_container->hasTrianglesAroundEdge())
            {

                for(unsigned int j=0; j<3; ++j)
                {
                    sofa::type::vector< TriangleID > &shell = m_container->m_trianglesAroundEdge[ m_container->m_edgesInTriangle[lastTriangle][j]];
                    replace(shell.begin(), shell.end(), (TriangleID)lastTriangle, indices[i]);
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
        removePointsProcess(vertexToBeRemoved, d_propagateToDOF.getValue());
    }
}



void TriangleSetTopologyModifier::removeEdgesProcess( const sofa::type::vector<EdgeID> &indices,
        const bool removeIsolatedItems)
{

    // Note: this does not check if an edge is removed from an existing triangle (it should never happen)

    if(m_container->hasEdgesInTriangle()) // this method should only be called when edges exist
    {
        if(!m_container->hasTrianglesAroundEdge())
            m_container->createTrianglesAroundEdgeArray();

        size_t lastEdge = m_container->getNumberOfEdges() - 1;
        for(size_t i = 0; i < indices.size(); ++i, --lastEdge)
        {
            // updating the triangles connected to the edge replacing the removed one:
            // for all triangles connected to the last point
            for(sofa::type::vector<TriangleID>::iterator itt = m_container->m_trianglesAroundEdge[lastEdge].begin();
                itt != m_container->m_trianglesAroundEdge[lastEdge].end(); ++itt)
            {
                const EdgeID edgeIndex = m_container->getEdgeIndexInTriangle(m_container->m_edgesInTriangle[(*itt)], (EdgeID)lastEdge);
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



void TriangleSetTopologyModifier::removePointsProcess(const sofa::type::vector<PointID> &indices,
        const bool removeDOF)
{

    if(m_container->hasTriangles())
    {
        if(!m_container->hasTrianglesAroundVertex())
            m_container->createTrianglesAroundVertexArray();

        helper::WriteAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = m_container->d_triangle;

        size_t lastPoint = m_container->getNbPoints() - 1;
        for(size_t i=0; i<indices.size(); ++i, --lastPoint)
        {
            // updating the triangles connected to the point replacing the removed one:
            // for all triangles connected to the last point

            PointID pointID = indices[i];
            const sofa::type::vector<TriangleID> &oldShell = m_container->m_trianglesAroundVertex[pointID];
            if (!oldShell.empty())
                msg_error() << "m_trianglesAroundVertex is not empty around point: " << pointID << " with shell array: " << oldShell;
            sofa::type::vector<TriangleID> &shell = m_container->m_trianglesAroundVertex[lastPoint];


            for(size_t j=0; j<shell.size(); ++j)
            {
                const TriangleID q = shell[j];
                for(unsigned int k=0; k<3; ++k)
                {
                    if(m_triangle[q][k] == lastPoint)
                        m_triangle[q][k] = pointID;
                }
            }

            // updating the triangle shell itself (change the old index for the new one)
            m_container->m_trianglesAroundVertex[ pointID ] = m_container->m_trianglesAroundVertex[ lastPoint ];
        }

        m_container->m_trianglesAroundVertex.resize( m_container->m_trianglesAroundVertex.size() - indices.size() );
    }

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent's method.
    EdgeSetTopologyModifier::removePointsProcess( indices, removeDOF );
}


void TriangleSetTopologyModifier::renumberPointsProcess( const sofa::type::vector<PointID> &index,
        const sofa::type::vector<PointID> &inv_index,
        const bool renumberDOF)
{

    if(m_container->hasTriangles())
    {
        if(m_container->hasTrianglesAroundVertex())
        {
            sofa::type::vector< sofa::type::vector< TriangleID > > trianglesAroundVertex_cp = m_container->m_trianglesAroundVertex;
            for(sofa::Index i=0; i<index.size(); ++i)
            {
                m_container->m_trianglesAroundVertex[i] = trianglesAroundVertex_cp[ index[i] ];
            }
        }
        helper::WriteAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = m_container->d_triangle;

        for(sofa::Index i=0; i<m_triangle.size(); ++i)
        {
            m_triangle[i][0] = inv_index[ m_triangle[i][0] ];
            m_triangle[i][1] = inv_index[ m_triangle[i][1] ];
            m_triangle[i][2] = inv_index[ m_triangle[i][2] ];
        }
    }

    // call the parent's method
    EdgeSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}


void TriangleSetTopologyModifier::addRemoveTriangles( const sofa::Size nTri2Add,
        const sofa::type::vector< Triangle >& triangles2Add,
        const sofa::type::vector< TriangleID >& trianglesIndex2Add,
        const sofa::type::vector< sofa::type::vector< TriangleID > > & ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs,
        sofa::type::vector< TriangleID >& trianglesIndex2remove)
{

    // Create all the triangles registered to be created
    this->addTrianglesProcess(triangles2Add); // WARNING called after the creation process by the method "addTrianglesProcess"

    // Warn for the creation of all the triangles registered to be created
    this->addTrianglesWarning (nTri2Add, triangles2Add, trianglesIndex2Add, ancestors, baryCoefs);

    // Propagate the topological changes *** not necessary ?? => in some cases yes (chains of topology mapping and topology data containers depending on each other)
    propagateTopologicalChanges();

    // Remove all the triangles registered to be removed
    this->removeTriangles(trianglesIndex2remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

}


void TriangleSetTopologyModifier::movePointsProcess (const sofa::type::vector<PointID>& id,
        const sofa::type::vector< sofa::type::vector< PointID > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs,
        const bool moveDOF)
{
    m_container->setTriangleTopologyToDirty();

    (void)moveDOF;
    const size_t nbrVertex = id.size();
    bool doublet;
    sofa::type::vector< TriangleID > trianglesAroundVertex2Move;
    sofa::type::vector< Triangle > trianglesArray;


    // Step 1/4 - Creating trianglesAroundVertex to moved due to moved points:
    for (size_t i = 0; i<nbrVertex; ++i)
    {
        const sofa::type::vector<TriangleID>& trianglesAroundVertex = m_container->getTrianglesAroundVertex( id[i] );

        for (size_t j = 0; j<trianglesAroundVertex.size(); ++j)
        {
            doublet = false;

            for (size_t k =0; k<trianglesAroundVertex2Move.size(); ++k) //Avoid double
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

    std::sort( trianglesAroundVertex2Move.begin(), trianglesAroundVertex2Move.end(), std::greater<TriangleID>() );


    // Step 2/4 - Create event to delete all elements before moving and propagate it:
    const TrianglesMoved_Removing *ev1 = new TrianglesMoved_Removing (trianglesAroundVertex2Move);
    this->addTopologyChange(ev1);
    propagateTopologicalChanges();


    // Step 3/4 - Physically move all dof:
    PointSetTopologyModifier::movePointsProcess (id, ancestors, coefs);

    m_container->setTriangleTopologyToDirty();

    // Step 4/4 - Create event to recompute all elements concerned by moving and propagate it:

    // Creating the corresponding array of Triangles for ancestors
    for (TriangleID i = 0; i<trianglesAroundVertex2Move.size(); i++)
        trianglesArray.push_back (m_container->getTriangleArray()[ trianglesAroundVertex2Move[i] ]);

    const TrianglesMoved_Adding *ev2 = new TrianglesMoved_Adding (trianglesAroundVertex2Move, trianglesArray);
    this->addTopologyChange(ev2); // This event should be propagated with global workflow
}



// Duplicate the given edge. Only works of at least one of its points is adjacent to a border.
int TriangleSetTopologyModifier::InciseAlongEdge(EdgeID ind_edge, int* createdPoints)
{
    const Edge & edge0 = m_container->getEdge(ind_edge);
    PointID ind_pa = edge0[0];
    PointID ind_pb = edge0[1];

    const type::vector<TriangleID>& triangles0 = m_container->getTrianglesAroundEdge(ind_edge);
    if (triangles0.size() != 2)
    {
        msg_error() << "InciseAlongEdge: ERROR edge " << ind_edge << " is not attached to 2 triangles.";
        return -1;
    }

    // choose one triangle
    TriangleID ind_tri0 = triangles0[0];

    PointID ind_tria = ind_tri0;
    PointID ind_trib = ind_tri0;
    EdgeID ind_edgea = ind_edge;
    EdgeID ind_edgeb = ind_edge;

    type::vector<TriangleID> list_tria;
    type::vector<TriangleID> list_trib;

    for (;;)
    {
        const EdgesInTriangle& te = m_container->getEdgesInTriangle(ind_tria);

        // find the edge adjacent to a that is not ind_edgea
        int j = 0;
        for (j = 0; j < 3; ++j)
        {
            if (te[j] != ind_edgea && (m_container->getEdge(te[j])[0] == ind_pa || m_container->getEdge(te[j])[1] == ind_pa))
                break;
        }
        if (j == 3)
        {
            msg_error() << "InciseAlongEdge: ERROR in triangle " << ind_tria;
            return -1;
        }

        ind_edgea = te[j];
        if (ind_edgea == ind_edge)
            break; // full loop

        const auto& tes = m_container->getTrianglesAroundEdge(ind_edgea);
        if (tes.size() < 2)
            break; // border edge

        if (tes[0] == ind_tria)
            ind_tria = tes[1];
        else
            ind_tria = tes[0];
        list_tria.push_back(ind_tria);
    }

    for (;;)
    {
        const EdgesInTriangle& te = m_container->getEdgesInTriangle(ind_trib);

        // find the edge adjacent to b that is not ind_edgeb
        int j = 0;
        for (j = 0; j < 3; ++j)
        {
            if (te[j] != ind_edgeb && (m_container->getEdge(te[j])[0] == ind_pb || m_container->getEdge(te[j])[1] == ind_pb))
                break;
        }
        if (j == 3)
        {
            msg_error() << "InciseAlongEdge: ERROR in triangle " << ind_trib;
            return -1;
        }

        ind_edgeb = te[j];
        if (ind_edgeb == ind_edge)
            break; // full loop

        const auto& tes = m_container->getTrianglesAroundEdge(ind_edgeb);
        if (tes.size() < 2)
            break; // border edge

        if (tes[0] == ind_trib)
            ind_trib = tes[1];
        else
            ind_trib = tes[0];
        list_trib.push_back(ind_trib);
    }

    bool pa_is_on_border = (ind_edgea != ind_edge);
    bool pb_is_on_border = (ind_edgeb != ind_edge);

    if (!pa_is_on_border && !pb_is_on_border)
    {
        msg_error() << "InciseAlongEdge: ERROR edge " << ind_edge << " is not on border.";
        return -1;
    }

    // now we can split the edge

    /// force the creation of TrianglesAroundEdgeArray
    m_container->getTrianglesAroundEdgeArray();
    /// force the creation of TrianglesAroundVertexArray
    m_container->getTrianglesAroundVertexArray();

    const sofa::Size nb_points = sofa::Size(m_container->getTrianglesAroundVertexArray().size());
    const sofa::type::vector<Triangle> &vect_t = m_container->getTriangleArray();
    const sofa::Size nb_triangles = sofa::Size(vect_t.size());

    // Variables to accumulate the number of elements registered to be created (so as to remember their indices)
    PointID acc_nb_points = (PointID)nb_points;
    TriangleID acc_nb_triangles = (TriangleID)nb_triangles;

    // Variables to accumulate the elements registered to be created or to be removed
    sofa::type::vector< sofa::type::vector< PointID > > p_ancestors;
    sofa::type::vector< sofa::type::vector< SReal > > p_baryCoefs;
    sofa::type::vector< Triangle > triangles_to_create;
    sofa::type::vector< TriangleID > trianglesIndexList;
    sofa::type::vector< TriangleID > triangles_to_remove;

    sofa::type::vector<SReal> defaultCoefs; 
    defaultCoefs.push_back(1.0);

    unsigned new_pa, new_pb;

    if (pa_is_on_border)
    {
        sofa::type::vector<PointID> ancestors;
        new_pa = acc_nb_points++;
        ancestors.push_back(ind_pa);
        p_ancestors.push_back(ancestors);
        p_baryCoefs.push_back(defaultCoefs);
        if (createdPoints) *(createdPoints++) = new_pa;
    }
    else
        new_pa = ind_pa;

    sofa::type::vector<PointID> ancestors(1);

    if (pb_is_on_border)
    {
        new_pb = acc_nb_points++;
        ancestors[0] = ind_pb;
        p_ancestors.push_back(ancestors);
        p_baryCoefs.push_back(defaultCoefs);
        if (createdPoints) *(createdPoints++) = new_pb;
    }
    else
        new_pb = ind_pb;

    // we need to recreate at least tri0
    Triangle new_tri0 = m_container->getTriangle(ind_tri0);
    for (unsigned i = 0; i < 3; i++)
    {
        if (new_tri0[i] == ind_pa)
            new_tri0[i] = new_pa;
        else if (new_tri0[i] == ind_pb)
            new_tri0[i] = new_pb;
    }

    triangles_to_remove.push_back(ind_tri0);
    ancestors[0] = ind_tri0;
    triangles_to_create.push_back(new_tri0);

    trianglesIndexList.push_back(acc_nb_triangles);
    acc_nb_triangles += 1;

    // recreate list_tria iff pa is new
    if (new_pa != ind_pa)
    {
        for (unsigned j = 0; j < list_tria.size(); j++)
        {
            unsigned ind_tri = list_tria[j];
            Triangle new_tri = m_container->getTriangle(ind_tri);
            for (unsigned i = 0; i < 3; i++)
                if (new_tri[i] == ind_pa) new_tri[i] = new_pa;
            triangles_to_remove.push_back(ind_tri);
            ancestors[0] = ind_tri;
            triangles_to_create.push_back(new_tri);

            trianglesIndexList.push_back(acc_nb_triangles);
            acc_nb_triangles += 1;
        }
    }

    // recreate list_trib iff pb is new
    if (new_pb != ind_pb)
    {
        for (unsigned j = 0; j < list_trib.size(); j++)
        {
            unsigned ind_tri = list_trib[j];
            Triangle new_tri = m_container->getTriangle(ind_tri);
            for (unsigned i = 0; i < 3; i++)
                if (new_tri[i] == ind_pb) new_tri[i] = new_pb;
            triangles_to_remove.push_back(ind_tri);
            ancestors[0] = ind_tri;
            triangles_to_create.push_back(new_tri);

            trianglesIndexList.push_back(acc_nb_triangles);
            acc_nb_triangles += 1;
        }
    }

    // Create all the points registered to be created
    addPointsProcess(acc_nb_points - nb_points);

    // Warn for the creation of all the points registered to be created
    addPointsWarning(acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

    // Create all the triangles registered to be created
    addTrianglesProcess((const sofa::type::vector< Triangle > &) triangles_to_create); // WARNING called after the creation process by the method "addTrianglesProcess"

    // Warn for the creation of all the triangles registered to be created
    addTrianglesWarning(sofa::Size(triangles_to_create.size()), triangles_to_create, trianglesIndexList);

    // Propagate the topological changes *** not necessary
    //propagateTopologicalChanges();

    // Remove all the triangles registered to be removed
    removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

    return (pb_is_on_border ? 1 : 0) + (pa_is_on_border ? 1 : 0); // todo: get new edge indice
}



bool TriangleSetTopologyModifier::removeTrianglesPreconditions(const sofa::type::vector< TriangleID >& items)
{
    (void)items;
    return true;
}

void TriangleSetTopologyModifier::removeTrianglesPostProcessing(const sofa::type::vector< TriangleID >& edgeToBeRemoved, const sofa::type::vector< TriangleID >& vertexToBeRemoved )
{
    (void)vertexToBeRemoved;
    (void)edgeToBeRemoved;
}


bool TriangleSetTopologyModifier::addTrianglesPreconditions(const sofa::type::vector<Triangle>& triangles)
{
    (void)triangles;
    return true;
}

void TriangleSetTopologyModifier::addTrianglesPostProcessing(const sofa::type::vector<Triangle>& triangles)
{
    (void)triangles;
}


void TriangleSetTopologyModifier::propagateTopologicalEngineChanges()
{
    if (m_container->beginChange() == m_container->endChange()) // nothing to do if no event is stored
        return;

    if (!m_container->isTriangleTopologyDirty()) // triangle Data has not been touched
        return EdgeSetTopologyModifier::propagateTopologicalEngineChanges();

    SCOPED_TIMER("TriangleSetTopologyModifier::propagateTopologicalEngineChanges");

    auto& triangleTopologyHandlerList = m_container->getTopologyHandlerList(sofa::geometry::ElementType::TRIANGLE);
    for (const auto topoHandler : triangleTopologyHandlerList)
    {
        if (topoHandler->isDirty())
        {
            topoHandler->update();
        }
    }

    m_container->cleanTriangleTopologyFromDirty();
    EdgeSetTopologyModifier::propagateTopologicalEngineChanges();
}

} //namespace sofa::component::topology::container::dynamic
