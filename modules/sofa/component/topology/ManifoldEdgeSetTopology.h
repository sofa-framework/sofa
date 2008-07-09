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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGY_H

#include <sofa/component/topology/EdgeSetTopology.h>
#include <vector>
#include <map>

#include <list>

namespace sofa
{

namespace component
{

namespace topology
{

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::EdgeID EdgeID;

typedef BaseMeshTopology::Edge Edge;
typedef BaseMeshTopology::SeqEdges SeqEdges;
typedef BaseMeshTopology::VertexEdges VertexEdges;

/////////////////////////////////////////////////////////
/// ManifoldEdgeSetTopology objects
/////////////////////////////////////////////////////////


/** a class that stores a set of edges and provides access to the adjacency between points and edges.
    this topology is constraint by the manifold property : each vertex is adjacent either to one vertex or to two vertices. */
class ManifoldEdgeSetTopologyContainer : public EdgeSetTopologyContainer
{

protected:

    // Describe each connected component, which can be seen as an oriented line
    class ConnectedComponent
    {
    public:

        unsigned int FirstVertexIndex; // index of the first vertex on the line
        unsigned int LastVertexIndex; // index of the last vertex on the line

        unsigned int size; // number of the vertices on the line

        unsigned int ccIndex; // index of the connected component stored in the m_ConnectedComponentArray

        ConnectedComponent(unsigned int FirstVertexIndex=-1, unsigned int LastVertexIndex=-1, unsigned int size=0,unsigned int ccIndex=0)
            :FirstVertexIndex(FirstVertexIndex), LastVertexIndex(LastVertexIndex), size(size), ccIndex(ccIndex)
        {
        }
    };

private:
    /** \brief Creates the EdgeSetIndex.
     *
     * This function is only called if the EdgeShell member is required.
     * EdgeShell[i] contains the indices of all edges having the ith DOF as
     * one of their ends.
     */
    virtual void createEdgeVertexShellArray();

protected:

    /** \brief Creates the EdgeSet array.
     *
     * This function is only called by derived classes to create a list of edges from a set of triangles or tetrahedra
     */
    virtual void createEdgeSetArray() {}

    /*** The array that stores for each vertex index, the connected component the vertex belongs to */
    sofa::helper::vector< unsigned int > m_ComponentVertexArray;

    /*** The array that stores the connected components */
    sofa::helper::vector< ConnectedComponent > m_ConnectedComponentArray;

    /** \brief Returns the ComponentVertex array.
     *
     */
    const sofa::helper::vector< unsigned int > &getComponentVertexArray() const
    {
        return m_ComponentVertexArray;
    }

    /** \brief Returns the array of connected components.
     *
     */
    const sofa::helper::vector< ConnectedComponent > &getConnectedComponentArray() const
    {
        return m_ConnectedComponentArray;
    }

public:

    /** \brief Resets the array of connected components and the ComponentVertex array (which are not valide anymore).
     *
     */
    void resetConnectedComponent()
    {

        m_ComponentVertexArray.clear();
        m_ConnectedComponentArray.clear();
    }

    /** \brief Returns true iff the array of connected components and the ComponentVertex array are valide (ie : not void)
     *
     */
    bool isvoid_ConnectedComponent()
    {
        return m_ConnectedComponentArray.size()==0;
    }

    /** \brief Computes the array of connected components and the ComponentVertex array (which makes them valide).
     *
     */
    void computeConnectedComponent();


    /** \brief Returns the number of connected components.
     *
     */
    virtual int getNumberOfConnectedComponents()
    {
        computeConnectedComponent();
        return m_ConnectedComponentArray.size();
    }

    /** \brief Returns the FirstVertexIndex of the ith connected component.
     *
     */
    virtual int getFirstVertexIndex(unsigned int i)
    {
        computeConnectedComponent();
        assert(i<m_ConnectedComponentArray.size());
        return m_ConnectedComponentArray[i].FirstVertexIndex;
    }

    /** \brief Returns the LastVertexIndex of the ith connected component.
     *
     */
    virtual int getLastVertexIndex(unsigned int i)
    {
        computeConnectedComponent();
        assert(i<m_ConnectedComponentArray.size());
        return m_ConnectedComponentArray[i].LastVertexIndex;
    }

    /** \brief Returns the size of the ith connected component.
     *
     */
    virtual int getComponentSize(unsigned int i)
    {
        computeConnectedComponent();
        assert(i<m_ConnectedComponentArray.size());
        return m_ConnectedComponentArray[i].size;
    }

    /** \brief Returns the index of the ith connected component.
     *
     */
    virtual int getComponentIndex(unsigned int i)
    {
        computeConnectedComponent();
        assert(i<m_ConnectedComponentArray.size());
        return m_ConnectedComponentArray[i].ccIndex;
    }

    /** \brief Returns true iff the ith connected component is closed (ie : iff FirstVertexIndex == LastVertexIndex).
     *
     */
    virtual bool isComponentClosed(unsigned int i)
    {
        computeConnectedComponent();
        assert(i<m_ConnectedComponentArray.size());
        return (m_ConnectedComponentArray[i].FirstVertexIndex == m_ConnectedComponentArray[i].LastVertexIndex);
    }


    inline friend std::ostream& operator<< (std::ostream& out, const ManifoldEdgeSetTopologyContainer& t)
    {
        out << t.m_edge.size();
        for (unsigned int i=0; i<t.m_edge.size(); i++)
            out << " " << t.m_edge[i][0] << " " << t.m_edge[i][1] ;

        return out;
    }

    /// Needed to be compliant with Datas.
    inline friend std::istream& operator>>(std::istream& in, ManifoldEdgeSetTopologyContainer& t)
    {
        unsigned int s;
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            Edge T; in >> T;
            t.m_edge.push_back(T);
        }
        return in;
    }


    /** \brief Returns the indice of the vertex which is next to the vertex indexed by i.
     */
    int getNextVertex(const unsigned int i)
    {

        assert(getEdgeVertexShell(i).size()>0);
        if((getEdgeVertexShell(i)).size()==1 && (getEdge((getEdgeVertexShell(i))[0]))[1]==i)
        {
            return -1;
        }
        else
        {
            return (getEdge((getEdgeVertexShell(i))[0]))[1];
        }

    }

    /** \brief Returns the indice of the vertex which is previous to the vertex indexed by i.
     */
    int getPreviousVertex(const unsigned int i)
    {

        assert(getEdgeVertexShell(i).size()>0);
        if((getEdgeVertexShell(i)).size()==1 && (getEdge((getEdgeVertexShell(i))[0]))[0]==i)
        {
            return -1;
        }
        else
        {
            return (getEdge((getEdgeVertexShell(i))[0]))[0];
        }
    }

    /** \brief Returns the indice of the edge which is next to the edge indexed by i.
     */
    int getNextEdge(const unsigned int i)
    {
        if((getEdgeVertexShell(getEdge(i)[1])).size()==1)
        {
            return -1;
        }
        else
        {
            return (getEdgeVertexShell(getEdge(i)[1]))[1];
        }
    }

    /** \brief Returns the indice of the edge which is previous to the edge indexed by i.
     */
    int getPreviousEdge(const unsigned int i)
    {
        if((getEdgeVertexShell(getEdge(i)[0])).size()==1)
        {
            return -1;
        }
        else
        {
            return (getEdgeVertexShell(getEdge(i)[0]))[0];
        }
    }


    /** \brief Returns the ith Edge.
     *
     */
    virtual const Edge &getEdge(const unsigned int i);



    /** \brief Returns the number of edges in this topology.
     *
     */
    virtual unsigned int getNumberOfEdges() ;

    /** \brief Returns the Edge array.
     *
     */
    virtual const sofa::helper::vector<Edge> &getEdgeArray();


    /** \brief Checks if the Edge Set Topology is coherent
     *
     * Check if the Edge and the Edhe Shell arrays are coherent
     */
    virtual bool checkTopology() const;

protected:

    /** \brief Returns the Edge Shell array.
     *
     */
    virtual const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getEdgeVertexShellArray() ;



public:
    /** \brief Returns the edge shell of the ith DOF.
     *
     */
    virtual const sofa::helper::vector< unsigned int > &getEdgeVertexShell(const unsigned int i) ;
    /** \brief Returns the index of the edge joining vertex v1 and vertex v2; returns -1 if no edge exists
     *
     */
    virtual int getEdgeIndex(const unsigned int v1, const unsigned int v2);


    /** \brief Returns the number of connected components
    @param components the array containing the optimal vertex permutation according to the Reverse CuthillMckee algorithm
    */
    virtual int getNumberConnectedComponents(sofa::helper::vector<unsigned int>& components);

    ManifoldEdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top=NULL,
            const sofa::helper::vector< Edge >         &edges    = (const sofa::helper::vector< Edge >)        0 );

    template< typename DataTypes >
    friend class ManifoldEdgeSetTopologyModifier;
protected:
    /** \brief Returns a non-const edge shell of the ith DOF for subsequent modification
     *
     */
    virtual sofa::helper::vector< unsigned int > &getEdgeVertexShellForModification(const unsigned int i);

};


template <class DataTypes>
class ManifoldEdgeSetTopologyLoader;

/**
 * A class that can apply basic transformations on a set of points.
 */
template<class DataTypes>
class ManifoldEdgeSetTopologyModifier : public EdgeSetTopologyModifier <DataTypes>
{

public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    ManifoldEdgeSetTopologyModifier(core::componentmodel::topology::BaseTopology *top) : EdgeSetTopologyModifier<DataTypes>(top)
    {
    }

    //protected:
    /** \brief Build an edge set topology from a file : also modifies the MechanicalObject
     *
     */
    virtual bool load(const char *filename);

    /** \brief Sends a message to warn that some edges were added in this topology.
     *
     * \sa addEdgesProcess
     */
    virtual void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors= (const sofa::helper::vector< sofa::helper::vector<unsigned int > >) 0 ,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs= (const sofa::helper::vector< sofa::helper::vector< double > >)0);


    /** \brief Add some edges to this topology.
     *
     * \sa addEdgesWarning
     */
    virtual void addEdgesProcess(const sofa::helper::vector< Edge > &edges);

    /** \brief Sends a message to warn that some edges are about to be deleted.
     *
     * \sa removeEdgesProcess
     */
    void removeEdgesWarning( sofa::helper::vector<unsigned int> &edges);

    /** \brief Effectively Remove a subset of edges. Eventually remove isolated vertices
     *
     * Elements corresponding to these edges are removed form the mechanical object's state vectors.
     *
     * Important : some structures might need to be warned BEFORE the edges are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
     * \sa removeEdgesWarning
     *
     * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
     *
     * @param removeIsolatedItems if true isolated vertices are also removed
     */
    void removeEdgesProcess(const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems=false);

    /** \brief Add some points to this topology.
     *
     * Use a list of ancestors to create the new points.
     * Last parameter baryCoefs defines the coefficient used for the creation of the new edges.
     * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
     * for the edge being created.
     * Important : the points are actually added to the mechanical object's state vectors iff (addDOF == true)
     *
     * \sa addPointsWarning
     */
    virtual void addPointsProcess(const unsigned int nPoints,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs = (const sofa::helper::vector< sofa::helper::vector< double > >)0,
            const bool addDOF = true);


    /** \brief Add a new point (who has no ancestors) to this topology.
     *
     * \sa addPointsWarning
     */
    virtual void addNewPoint(unsigned int i, const sofa::helper::vector< double >& x);


    /** \brief Remove a subset of points
     *
     * these points are removed form the mechanical object's state vectors.
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
     * \sa removePointsWarning
     * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
     */
    virtual void removePointsProcess( sofa::helper::vector<unsigned int> &indices, const bool removeDOF = true);

    /** \brief Reorder this topology.
     *
     * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
     * \see MechanicalObject::renumberValues
     */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &/*inv_index*/, const bool renumberDOF = true);

    /** \brief Swap the edges.
     *
     */
    virtual void swapEdgesProcess(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs);

    /** \brief Fuse the edges.
     *
     * @param removeIsolatedItems if true isolated vertices are also removed
     */
    virtual void fuseEdgesProcess(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs, const bool removeIsolatedPoints = true);


    /** \brief Split the edges.
     *
     */
    virtual void splitEdgesProcess( sofa::helper::vector<unsigned int> &indices,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs = sofa::helper::vector< sofa::helper::vector< double > >(0),
            const bool removeIsolatedPoints = true);

    /** \brief Load an edge.
     */
    virtual void addEdge(Edge e);

public:
    //template <class DataTypes>
    friend class ManifoldEdgeSetTopologyLoader<DataTypes>;
};

/**
 * A class that performs topology algorithms on an EdgeSet.
 */
template < class DataTypes >
class ManifoldEdgeSetTopologyAlgorithms : public EdgeSetTopologyAlgorithms<DataTypes>
{

public:

    typedef typename DataTypes::Real Real;

    /** \brief Remove a set  of edges
        @param edges an array of edge indices to be removed (note that the array is not const since it needs to be sorted)
        *
        */
    virtual void removeEdges(sofa::helper::vector< unsigned int >& edges, const bool removeIsolatedPoints = true);

    /** \brief Generic method to remove a list of items.
     */
    virtual void removeItems(sofa::helper::vector< unsigned int >& items);

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> &/*index*/, const sofa::helper::vector<unsigned int> &/*inv_index*/);

    /** \brief add a set  of edges
        @param edges an array of pair of vertex indices describing the edge to be created
        @param ancestors for each edge to be created provides an array of edge ancestors (optional)
        @param baryCoefs for each edge provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
        *
        */
    virtual void addEdges(const sofa::helper::vector< Edge >& edges,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors= (const sofa::helper::vector< sofa::helper::vector<unsigned int > >) 0 ,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs= (const sofa::helper::vector< sofa::helper::vector< double > >)0) ;

    /** \brief Swap a list of pair edges, replacing each edge pair ((p11, p12), (p21, p22)) by the edge pair ((p11, p21), (p12, p22))
     *
     */
    virtual void swapEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs);

    /** \brief Fuse a list of pair edges.
     *
     */
    virtual void fuseEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPair, const bool removeIsolatedPoints = true);

    /** \brief Split an array of edges. On each edge, a vertex is created based on its barycentric coordinates
     *
     */
    virtual void splitEdges( sofa::helper::vector<unsigned int> &indices,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs = (const sofa::helper::vector< sofa::helper::vector< double > >)0 ,
            const bool removeIsolatedPoints = true);

    /** \brief Gives the optimal vertex permutation according to the Reverse CuthillMckee algorithm (use BOOST GRAPH LIBRAIRY)
    */
    //virtual void resortCuthillMckee(sofa::helper::vector<int>& inverse_permutation);

    ManifoldEdgeSetTopologyAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top) : EdgeSetTopologyAlgorithms<DataTypes>(top)
    {
    }

};

/**
 * A class that provides geometry information on an EdgeSet.
 */
template < class DataTypes >
class ManifoldEdgeSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;


    ManifoldEdgeSetGeometryAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top) : EdgeSetGeometryAlgorithms<DataTypes>(top)
    {
    }
};


/** Describes a topological object that only consists as a set of points and lines connecting these points */
template<class DataTypes>
class ManifoldEdgeSetTopology : public EdgeSetTopology <DataTypes>
{
public:

    ManifoldEdgeSetTopology(component::MechanicalObject<DataTypes> *obj);

    DataPtr< ManifoldEdgeSetTopologyContainer > *f_m_topologyContainer;

    virtual void init();


    /** \brief Returns the EdgeSetTopologyContainer object of this EdgeSetTopology.
     */
    virtual ManifoldEdgeSetTopologyContainer *getEdgeSetTopologyContainer() const
    {
        return (ManifoldEdgeSetTopologyContainer *)this->m_topologyContainer;
    }
    /** \brief Returns the EdgeSetTopologyAlgorithms object of this EdgeSetTopology.
     */
    virtual ManifoldEdgeSetTopologyAlgorithms<DataTypes> *getEdgeSetTopologyAlgorithms() const
    {
        return (ManifoldEdgeSetTopologyAlgorithms<DataTypes> *)this->m_topologyAlgorithms;
    }

    /** \brief Generic method returning the TopologyAlgorithms object
     */
    virtual core::componentmodel::topology::TopologyAlgorithms *getTopologyAlgorithms() const
    {
        return getEdgeSetTopologyAlgorithms();
    }

    /** \brief Returns the EdgeSetTopologyAlgorithms object of this EdgeSetTopology.
     */
    virtual ManifoldEdgeSetGeometryAlgorithms<DataTypes> *getEdgeSetGeometryAlgorithms() const
    {
        return (ManifoldEdgeSetGeometryAlgorithms<DataTypes> *)this->m_geometryAlgorithms;
    }

    /// BaseMeshTopology API
    /// @{

    virtual const SeqEdges& getEdges()         { return getEdgeSetTopologyContainer()->getEdgeArray(); }
    /// Returns the set of edges adjacent to a given vertex.
    virtual const VertexEdges& getEdgeVertexShell(PointID i) { return getEdgeSetTopologyContainer()->getEdgeVertexShell(i); }

    /// @}

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
