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
#include <map>

//#include <list> //BIBI ?

namespace sofa
{

namespace component
{

namespace topology
{
template<class DataTypes>
class ManifoldEdgeSetTopology;

class ManifoldEdgeSetTopologyContainer;

template<class DataTypes>
class ManifoldEdgeSetTopologyModifier;

template < class DataTypes >
class ManifoldEdgeSetTopologyAlgorithms;

template < class DataTypes >
class ManifoldEdgeSetGeometryAlgorithms;

template <class DataTypes>
class ManifoldEdgeSetTopologyLoader;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::EdgeID EdgeID;

typedef BaseMeshTopology::Edge Edge;
typedef BaseMeshTopology::SeqEdges SeqEdges;
typedef BaseMeshTopology::VertexEdges VertexEdges;

/////////////////////////////////////////////////////////
/// ManifoldEdgeSetTopology objects
/////////////////////////////////////////////////////////

/** Describes a topological object that only consists as a set of points and lines connecting these points.
    This topology is constraint by the manifold property : each vertex is adjacent either to one vertex or to two vertices. */
template<class DataTypes>
class ManifoldEdgeSetTopology : public EdgeSetTopology <DataTypes>
{
public:
    ManifoldEdgeSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~ManifoldEdgeSetTopology() {}

    virtual void init();

    /** \brief Returns the EdgeSetTopologyContainer object of this ManifoldEdgeSetTopology.
     */
    ManifoldEdgeSetTopologyContainer *getEdgeSetTopologyContainer() const
    {
        return static_cast<ManifoldEdgeSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the EdgeSetTopologyModifier object of this ManifoldEdgeSetTopology.
    */
    ManifoldEdgeSetTopologyModifier<DataTypes> *getEdgeSetTopologyModifier() const
    {
        return static_cast<ManifoldEdgeSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the EdgeSetTopologyAlgorithms object of this ManifoldEdgeSetTopology.
     */
    ManifoldEdgeSetTopologyAlgorithms<DataTypes> *getEdgeSetTopologyAlgorithms() const
    {
        return static_cast<ManifoldEdgeSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Generic method returning the TopologyAlgorithms object // BIBI
    */
    /*
    virtual core::componentmodel::topology::TopologyAlgorithms *getTopologyAlgorithms() const {
    return getEdgeSetTopologyAlgorithms();
    }
    */

    /** \brief Returns the EdgeSetGeometryAlgorithms object of this ManifoldEdgeSetTopology.
    */
    ManifoldEdgeSetGeometryAlgorithms<DataTypes> *getEdgeSetGeometryAlgorithms() const
    {
        return static_cast<ManifoldEdgeSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
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

    /// Returns the index of the edge joining vertex v1 and vertex v2; returns -1 if no edge exists
    virtual int getEdgeIndex(PointID v1, PointID v2)
    {
        return getEdgeSetTopologyContainer()->getEdgeIndex(v1, v2);
    }

    /// @}

protected:
    virtual void createComponents();
};

/**
* A class that performs topology algorithms on an ManifoldEdgeSet.
*/
template < class DataTypes >
class ManifoldEdgeSetTopologyAlgorithms : public EdgeSetTopologyAlgorithms<DataTypes>
{
public:
    ManifoldEdgeSetTopologyAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top)
        : EdgeSetTopologyAlgorithms<DataTypes>(top)
    {}

    virtual ~ManifoldEdgeSetTopologyAlgorithms() {}

    ManifoldEdgeSetTopology< DataTypes >* getEdgeSetTopology() const
    {
        return static_cast<ManifoldEdgeSetTopology< DataTypes >* > (this->m_basicTopology);
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

};

/**
* A class that provides geometry information on an ManifoldEdgeSet.
*/
template < class DataTypes >
class ManifoldEdgeSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    ManifoldEdgeSetGeometryAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top)
        : EdgeSetGeometryAlgorithms<DataTypes>(top)
    {}

    virtual ~ManifoldEdgeSetGeometryAlgorithms() {}

    ManifoldEdgeSetTopology< DataTypes >* getEdgeSetTopology() const
    {
        return static_cast<ManifoldEdgeSetTopology< DataTypes >* > (this->m_basicTopology);
    }
};

/**
* A class that can apply basic transformations on a set of points.
*/
template<class DataTypes>
class ManifoldEdgeSetTopologyModifier : public EdgeSetTopologyModifier <DataTypes>
{

    friend class ManifoldEdgeSetTopologyLoader<DataTypes>;

public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    ManifoldEdgeSetTopologyModifier(core::componentmodel::topology::BaseTopology *top)
        : EdgeSetTopologyModifier<DataTypes>(top)
    {}

    virtual ~ManifoldEdgeSetTopologyModifier() {}

    ManifoldEdgeSetTopology<DataTypes> * getEdgeSetTopology() const
    {
        return static_cast<ManifoldEdgeSetTopology<DataTypes> *> (this->m_basicTopology);
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

/** a class that stores a set of edges and provides access to the adjacency between points and edges.
  this topology is constraint by the manifold property : each vertex is adjacent either to one vertex or to two vertices. */
class ManifoldEdgeSetTopologyContainer : public EdgeSetTopologyContainer
{
    template< typename DataTypes >
    friend class ManifoldEdgeSetTopologyModifier;

public:
    ManifoldEdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top = NULL);

    ManifoldEdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top,
            const sofa::helper::vector< Edge > &edges);

    virtual ~ManifoldEdgeSetTopologyContainer() {}

    template< typename DataTypes >
    EdgeSetTopology< DataTypes >* getEdgeSetTopology() const
    {
        return static_cast<EdgeSetTopology< DataTypes >* > (this->m_basicTopology);
    }

    // Describe each connected component, which can be seen as an oriented line
    class ConnectedComponent
    {
    public:

        ConnectedComponent(unsigned int FirstVertexIndex=-1, unsigned int LastVertexIndex=-1, unsigned int size=0,unsigned int ccIndex=0)
            : FirstVertexIndex(FirstVertexIndex), LastVertexIndex(LastVertexIndex), size(size), ccIndex(ccIndex)
        {}

        virtual ~ConnectedComponent() {}

    public:
        unsigned int FirstVertexIndex; // index of the first vertex on the line
        unsigned int LastVertexIndex; // index of the last vertex on the line

        unsigned int size; // number of the vertices on the line

        unsigned int ccIndex; // index of the connected component stored in the m_ConnectedComponentArray
    };

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

    bool hasEdges() const;

    bool hasEdgeVertexShell() const;

    void clearEdges();

    void clearEdgeVertexShell();

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

protected:
    /*** The array that stores the set of edges in the edge set */
    sofa::helper::vector<Edge> m_edge;

    /** The array that stores the set of edge-vertex shells, ie for each vertex gives the set of adjacent edges */
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_edgeVertexShell;

    /** The array that stores for each vertex index, the connected component the vertex belongs to */
    sofa::helper::vector< unsigned int > m_ComponentVertexArray;

    /** The array that stores the connected components */
    sofa::helper::vector< ConnectedComponent > m_ConnectedComponentArray;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
