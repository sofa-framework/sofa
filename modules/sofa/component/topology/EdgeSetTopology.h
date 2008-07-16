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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGY_H

#include <sofa/component/topology/PointSetTopology.h>
#include <map>

namespace sofa
{

namespace component
{

namespace topology
{
template<class DataTypes>
class EdgeSetTopology;

class EdgeSetTopologyContainer;

template<class DataTypes>
class EdgeSetTopologyModifier;

template < class DataTypes >
class EdgeSetTopologyAlgorithms;

template < class DataTypes >
class EdgeSetGeometryAlgorithms;

template <class DataTypes>
class EdgeSetTopologyLoader;

class EdgesAdded;
class EdgesRemoved;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::EdgeID EdgeID;

typedef BaseMeshTopology::Edge Edge;
typedef BaseMeshTopology::SeqEdges SeqEdges;
typedef BaseMeshTopology::VertexEdges VertexEdges;


/////////////////////////////////////////////////////////
/// EdgeSetTopology objects
/////////////////////////////////////////////////////////


/** Describes a topological object that consists as a set of points and lines connecting these points */
template<class DataTypes>
class EdgeSetTopology : public PointSetTopology <DataTypes>
{
public:
    EdgeSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~EdgeSetTopology() {}

    virtual void init();

    /** \brief Returns the EdgeSetTopologyContainer object of this EdgeSetTopology.
    */
    EdgeSetTopologyContainer *getEdgeSetTopologyContainer() const
    {
        return static_cast<EdgeSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the EdgeSetTopologyModifier object of this EdgeSetTopology.
    */
    EdgeSetTopologyModifier<DataTypes> *getEdgeSetTopologyModifier() const
    {
        return static_cast<EdgeSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the EdgeSetTopologyAlgorithms object of this EdgeSetTopology.
    */
    EdgeSetTopologyAlgorithms<DataTypes> *getEdgeSetTopologyAlgorithms() const
    {
        return static_cast<EdgeSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Returns the EdgeSetTopologyAlgorithms object of this EdgeSetTopology.
    */
    EdgeSetGeometryAlgorithms<DataTypes> *getEdgeSetGeometryAlgorithms() const
    {
        return static_cast<EdgeSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
    }

    /// BaseMeshTopology API
    /// @{

    virtual const SeqEdges& getEdges()
    {
        return getEdgeSetTopologyContainer()->getEdgeArray();
    }

    /// Returns the set of edges adjacent to a given vertex.
    virtual const VertexEdges& getEdgeVertexShell(PointID i)
    {
        return getEdgeSetTopologyContainer()->getEdgeVertexShell(i);
    }

    /// @}

public:
    DataPtr< EdgeSetTopologyContainer > *f_m_topologyContainer;
};


/**
* A class that performs topology algorithms on an EdgeSet.
*/
template < class DataTypes >
class EdgeSetTopologyAlgorithms : public PointSetTopologyAlgorithms<DataTypes>
{
public:
    EdgeSetTopologyAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top)
        : PointSetTopologyAlgorithms<DataTypes>(top)
    {}

    virtual ~EdgeSetTopologyAlgorithms() {}

    EdgeSetTopology< DataTypes >* getEdgeSetTopology() const
    {
        return static_cast<EdgeSetTopology< DataTypes >* > (this->m_basicTopology);
    }

    /** \brief Remove a set  of edges
    @param edges an array of edge indices to be removed (note that the array is not const since it needs to be sorted)
    *
    */
    // side effect: edges are sorted in removeEdgesWarning
    virtual void removeEdges(/*const*/ sofa::helper::vector< unsigned int >& edges,
            const bool removeIsolatedPoints = true);

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(/*const*/ sofa::helper::vector< unsigned int >& items);

    /** \brief Generic method to write the current mesh into a msh file
    */
    virtual void writeMSH(const char * /*filename*/) {return;}

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> & index,
            const sofa::helper::vector<unsigned int> & inv_index);

    /** \brief add a set  of edges
    @param edges an array of pair of vertex indices describing the edge to be created
    @param ancestors for each edge to be created provides an array of edge ancestors (optional)
    @param baryCoefs for each edge provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addEdges(const sofa::helper::vector< Edge >& edges) ;

    /** \brief add a set  of edges
    @param edges an array of pair of vertex indices describing the edge to be created
    @param ancestors for each edge to be created provides an array of edge ancestors (optional)
    @param baryCoefs for each edge provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addEdges(const sofa::helper::vector< Edge >& edges,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs) ;

    /** \brief Swap a list of pair edges, replacing each edge pair ((p11, p12), (p21, p22)) by the edge pair ((p11, p21), (p12, p22))
    *
    */
    virtual void swapEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs);

    /** \brief Fuse a list of pair edges, replacing each edge pair ((p11, p12), (p21, p22)) by one edge (p11, p22)
    *
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void fuseEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs, const bool removeIsolatedPoints = true);

    /** \brief Split an array of edges, replacing each edge (p1, p2) by two edges (p1, p3) and (p3, p2) where p3 is the new vertex
    * On each edge, a vertex is created based on its barycentric coordinates
    *
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void splitEdges( sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedPoints = true);

    /** \brief Split an array of edges, replacing each edge (p1, p2) by two edges (p1, p3) and (p3, p2) where p3 is the new vertex
    * On each edge, a vertex is created based on its barycentric coordinates
    *
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void splitEdges( sofa::helper::vector<unsigned int> &indices,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
            const bool removeIsolatedPoints = true);

    /** \brief Gives the optimal vertex permutation according to the Reverse CuthillMckee algorithm (use BOOST GRAPH LIBRAIRY)
    */
    virtual void resortCuthillMckee(sofa::helper::vector<int>& inverse_permutation);
};


/** \brief A class used as an interface with an array : Useful to compute geometric information on each edge in an efficient way
*
*/
template < class T>
class BasicArrayInterface
{
public:
    // Access to i-th element.
    virtual T & operator[](int i)=0;
    virtual ~BasicArrayInterface() {}

};

/**
* A class that provides geometry information on an EdgeSet.
*/
template < class DataTypes >
class EdgeSetGeometryAlgorithms : public PointSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    EdgeSetGeometryAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top)
        : PointSetGeometryAlgorithms<DataTypes>(top)
    {}

    virtual ~EdgeSetGeometryAlgorithms() {}

    EdgeSetTopology< DataTypes >* getEdgeSetTopology() const
    {
        return static_cast<EdgeSetTopology< DataTypes >* > (this->m_basicTopology);
    }

    /// computes the length of edge no i and returns it
    virtual Real computeEdgeLength(const unsigned int i) const;

    /// computes the edge length of all edges are store in the array interface
    virtual void computeEdgeLength( BasicArrayInterface<Real> &ai) const;		// TODO: clarify, why not to use a vector here

    /// computes the initial length of edge no i and returns it
    virtual Real computeRestEdgeLength(const unsigned int i) const;

    /// computes the initial square length of edge no i and returns it
    virtual Real computeRestSquareEdgeLength(const unsigned int i) const;
};

/**
* A class that can apply basic transformations on a set of edges.
*/
template<class DataTypes>
class EdgeSetTopologyModifier : public PointSetTopologyModifier <DataTypes>
{
    friend class EdgeSetTopologyLoader<DataTypes>;

public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    EdgeSetTopologyModifier(core::componentmodel::topology::BaseTopology *top)
        : PointSetTopologyModifier<DataTypes>(top)
    {}

    virtual ~EdgeSetTopologyModifier() {}

    EdgeSetTopology<DataTypes> * getEdgeSetTopology() const
    {
        return static_cast<EdgeSetTopology<DataTypes> *> (this->m_basicTopology);
    }

    /** \brief Build an edge set topology from a file : also modifies the MechanicalObject
    *
    */
    virtual bool load(const char *filename);

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    virtual void addEdgesWarning(const unsigned int nEdges);

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    virtual void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList);

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    virtual void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors);

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    virtual void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs);


    /** \brief Add some edges to this topology.
    *
    * \sa addEdgesWarning
    */
    virtual void addEdgesProcess(const sofa::helper::vector< Edge > &edges);

    /** \brief Sends a message to warn that some edges are about to be deleted.
    *
    * \sa removeEdgesProcess
    */
    // side effect : edges are sorted first
    virtual void removeEdgesWarning(/*const*/ sofa::helper::vector<unsigned int> &edges);

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
    virtual void removeEdgesProcess(const sofa::helper::vector<unsigned int> &indices, const bool removeIsolatedItems = false);

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
    virtual void addPointsProcess(const unsigned int nPoints, const bool addDOF = true);

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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
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
    virtual void removePointsProcess(/*const*/ sofa::helper::vector<unsigned int> &indices,
            const bool removeDOF = true);

    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &/*inv_index*/,
            const bool renumberDOF = true);

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
    * @param removeIsolatedItems if true isolated vertices are also removed
    */
    virtual void splitEdgesProcess(/*const*/ sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedPoints = true);

    /** \brief Split the edges.
    *
    * @param removeIsolatedItems if true isolated vertices are also removed
    */
    virtual void splitEdgesProcess(/*const*/ sofa::helper::vector<unsigned int> &indices,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
            const bool removeIsolatedPoints = true);

    /** \brief Load an edge.
    */
    void addEdge(Edge e);
};

/** a class that stores a set of edges  and provides access to the adjacency between points and edges */
class EdgeSetTopologyContainer : public PointSetTopologyContainer
{
    template< typename DataTypes >
    friend class EdgeSetTopologyModifier;

public:
    EdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top = NULL);

    EdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top,
            const sofa::helper::vector< Edge > &edges);

    virtual ~EdgeSetTopologyContainer() {}

    template< typename DataTypes >
    EdgeSetTopology< DataTypes >* getEdgeSetTopology() const
    {
        return static_cast<EdgeSetTopology< DataTypes >* > (this->m_basicTopology);
    }

    /** \brief Returns the Edge array.
    *
    */
    virtual const sofa::helper::vector<Edge> &getEdgeArray(); // TODO: const;

    /** \brief Returns the ith Edge.
    *
    */
    virtual const Edge &getEdge(const unsigned int i); // TODO: const;

    /** \brief Returns the number of edges in this topology.
    *
    */
    virtual unsigned int getNumberOfEdges(); // TODO: const;

    /** \brief Returns the Edge Shell array.
    *
    */
    virtual const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getEdgeVertexShellArray(); // TODO: const;

    /** \brief Returns the edge shell of the ith DOF.
    *
    */
    virtual const sofa::helper::vector< unsigned int > &getEdgeVertexShell(const unsigned int i);

    /** \brief Returns the index of the edge joining vertex v1 and vertex v2; returns -1 if no edge exists
    *
    */
    virtual int getEdgeIndex(const unsigned int v1, const unsigned int v2);

    /** \brief Checks if the Edge Set Topology is coherent
    *
    * Check if the Edge and the Edhe Shell arrays are coherent
    */
    virtual bool checkTopology() const;

    /** \brief Returns the number of connected components from the graph containing all edges and give, for each vertex, which component it belongs to  (use BOOST GRAPH LIBRAIRY)
    @param components the array containing the optimal vertex permutation according to the Reverse CuthillMckee algorithm
    */
    virtual int getNumberConnectedComponents(sofa::helper::vector<unsigned int>& components); // TODO: const;

    inline friend std::ostream& operator<< (std::ostream& out, const EdgeSetTopologyContainer& t)
    {
        out << t.m_edge.size();
        for (unsigned int i=0; i<t.m_edge.size(); i++)
            out << " " << t.m_edge[i][0] << " " << t.m_edge[i][1] ;

        return out;
    }

    /// Needed to be compliant with Datas.
    inline friend std::istream& operator>>(std::istream& in, EdgeSetTopologyContainer& t)
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

protected:
    /** \brief Returns a non-const edge shell of the ith DOF for subsequent modification
    *
    */
    virtual sofa::helper::vector< unsigned int > &getEdgeVertexShellForModification(const unsigned int i);

    /** \brief Creates the EdgeSet array.
    *
    * This function must be implemented by derived classes to create a list of edges from a set of triangles or tetrahedra
    */
    virtual void createEdgeSetArray();

    /** \brief Creates the EdgeSetIndex.
    *
    * This function is only called if the EdgeShell member is required.
    * EdgeShell[i] contains the indices of all edges having the ith DOF as
    * one of their ends.
    */
    virtual void createEdgeVertexShellArray();

protected:
    /*** The array that stores the set of edges in the edge set */
    sofa::helper::vector<Edge> m_edge;

    /** the array that stores the set of edge-vertex shells, ie for each vertex gives the set of adjacent edges */
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_edgeVertexShell;
};

/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////


/** indicates that some edges were added */
class EdgesAdded : public core::componentmodel::topology::TopologyChange
{

public:
    EdgesAdded(const unsigned int nE)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::EDGESADDED),
          nEdges(nE)
    { }

    EdgesAdded(const unsigned int nE,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndex)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex)
    { }

    EdgesAdded(const unsigned int nE,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndex,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex),
          ancestorsList(ancestors)
    { }

    EdgesAdded(const unsigned int nE,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndex,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::EDGESADDED),
          nEdges(nE),
          edgeArray(edgesList),
          edgeIndexArray(edgesIndex),
          ancestorsList(ancestors),
          coefs(baryCoefs)
    { }

    virtual ~EdgesAdded() {}

    unsigned int getNbAddedEdges() const { return nEdges;}

public:
    unsigned int nEdges;
    sofa::helper::vector< Edge > edgeArray;
    sofa::helper::vector< unsigned int > edgeIndexArray;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double > > coefs;
};


/** indicates that some edges are about to be removed */
class EdgesRemoved : public core::componentmodel::topology::TopologyChange
{
public:
    EdgesRemoved(const sofa::helper::vector<unsigned int> _eArray)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::EDGESREMOVED),
          removedEdgesArray(_eArray)
    {}

    ~EdgesRemoved() {}

    virtual const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedEdgesArray;
    }

    virtual unsigned int getNbRemovedEdges() const
    {
        return removedEdgesArray.size();
    }

public:
    sofa::helper::vector<unsigned int> removedEdgesArray;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
