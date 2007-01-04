#ifndef SOFA_COMPONENTS_EDGESETTOPOLOGY_H
#define SOFA_COMPONENTS_EDGESETTOPOLOGY_H

#include "PointSetTopology.h"
#include <vector>
#include <map>

namespace Sofa
{

namespace Components
{


/// defining Edges as the pair of the DOFs indices
typedef std::pair<unsigned int, unsigned int> Edge;



/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////



/** indicates that some edges were added */
class EdgesAdded : public Core::TopologyChange
{

public:
    unsigned int nEdges;

    std::vector< Edge > edgeArray;

    std::vector< unsigned int > edgeIndexArray;

    std::vector< std::vector< unsigned int > > ancestorsList;

    std::vector< std::vector< double > > coefs;

    EdgesAdded(const unsigned int nE,
            const std::vector< Edge >& edgesList = (const std::vector< Edge >)0,
            const std::vector< unsigned int >& edgesIndex = (const std::vector< unsigned int >)0,
            const std::vector< std::vector< unsigned int > >& ancestors = (const std::vector< std::vector< unsigned int > >)0,
            const std::vector< std::vector< double > >& baryCoefs = (const std::vector< std::vector< double > >)0)
        : Core::TopologyChange(Core::EDGESADDED), nEdges(nE), edgeArray(edgesList), edgeIndexArray(edgesIndex),ancestorsList(ancestors), coefs(baryCoefs)
    {   }

    unsigned int getNbAddedEdges() const
    {
        return nEdges;
    }

};



/** indicates that some edges are about to be removed */
class EdgesRemoved : public Core::TopologyChange
{

public:
    std::vector<unsigned int> removedEdgesArray;

public:
    EdgesRemoved(const std::vector<unsigned int> _eArray) : Core::TopologyChange(Core::EDGESREMOVED), removedEdgesArray(_eArray)
    {
    }

    const std::vector<unsigned int> &getArray() const
    {
        return removedEdgesArray;
    }
    unsigned int getNbRemovedEdges() const
    {
        return removedEdgesArray.size();
    }

};



/////////////////////////////////////////////////////////
/// EdgeSetTopology objects
/////////////////////////////////////////////////////////


/** a class that stores a set of points and provides access
to each point */
class EdgeSetTopologyContainer : public PointSetTopologyContainer
{

private:
    /** \brief Creates the EdgeSetIndex.
     *
     * This function is only called if the EdgeShell member is required.
     * EdgeShell[i] contains the indices of all edges having the ith DOF as
     * one of their ends.
     */
    void createEdgeShellArray();

protected:
    std::vector<Edge> m_edge;
    std::vector< std::vector< unsigned int > > m_edgeShell;

    /** \brief Creates the EdgeSet array.
     *
     * This function is only called by derived classes to create a list of edges from a set of triangles or tetrahedra
    */
    virtual void createEdgeSetArray() {}

public:
    /** \brief Returns the Edge array.
     *
     */
    const std::vector<Edge> &getEdgeArray();



    /** \brief Returns the ith Edge.
     *
     */
    const Edge &getEdge(const unsigned int i);



    /** \brief Returns the number of edges in this topology.
     *
     */
    unsigned int getNumberOfEdges() ;



    /** \brief Returns the Edge Shells array.
     *
     */
    const std::vector< std::vector<unsigned int> > &getEdgeShellsArray() ;

    /** \brief Returns the edge shell of the ith DOF.
     *
     */
    const std::vector< unsigned int > &getEdgeShell(const unsigned int i) ;
    /** \brief Returns the index of the edge joining vertex v1 and vertex v2; returns -1 if no edge exists
     *
     */
    int getEdgeIndex(const unsigned int v1, const unsigned int v2);


    //EdgeSetTopologyContainer(Core::BasicTopology *top);

    EdgeSetTopologyContainer(Core::BasicTopology *top, const std::vector< unsigned int > &DOFIndex = (const std::vector< unsigned int >)0,
            const std::vector< Edge >         &edges    = (const std::vector< Edge >)        0 );

    template< typename DataTypes >
    friend class EdgeSetTopologyModifier;
protected:
    /** \brief Returns a non-const edge shell of the ith DOF for subsequent modification
     *
     */
    std::vector< unsigned int > &getEdgeShellForModification(const unsigned int i);

};



/**
 * A class that can apply basic transformations on a set of points.
 */
template<class DataTypes>
class EdgeSetTopologyModifier : public PointSetTopologyModifier <DataTypes>
{

public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    EdgeSetTopologyModifier(Core::BasicTopology *top) : PointSetTopologyModifier<DataTypes>(top)
    {
    }
    /** \brief Build an edge set topology from a file : also modifies the MechanicalObject
     *
     */
    virtual bool load(const char *filename);
    /** \brief Build a point set topology from a file : also modifies the MechanicalObject
     *
     */

    /** \brief Sends a message to warn that some edges were added in this topology.
     *
     * \sa addEdgesProcess
     */
    void addEdgesWarning(const unsigned int nEdges,
            const std::vector< Edge >& edgesList,
            const std::vector< unsigned int >& edgesIndexList,
            const std::vector< std::vector< unsigned int > > & ancestors= (const std::vector< std::vector<unsigned int > >) 0 ,
            const std::vector< std::vector< double > >& baryCoefs= (const std::vector< std::vector< double > >)0) ;



    /** \brief Add some edges to this topology.
     *
     * \sa addEdgesWarning
     */
    virtual void addEdgesProcess(const std::vector< Edge > &edges);




    /** \brief Sends a message to warn that some points are about to be deleted.
     *
     * \sa removeEdgesProcess
     */
    void removeEdgesWarning( std::vector<unsigned int> &edges);



    /** \brief Remove the points whose indices are given from this topology.
     *
     * Elements corresponding to these points are removed form the mechanical object's state vectors.
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
     * \sa removeEdgesWarning
     *
     * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
     */
    virtual void removeEdgesProcess(const unsigned int nEdges,  const std::vector<unsigned int> &indices);



    /** \brief Add some points to this topology.
     *
     * Use a list of ancestors to create the new points.
     * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
     * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
     * for the point being created.
     *
     * \sa addPointsWarning
     */
    virtual void addPointsProcess(const unsigned int nPoints,
            const std::vector< std::vector< unsigned int > >& ancestors = (const std::vector< std::vector< unsigned int > >)0,
            const std::vector< std::vector< double > >& baryCoefs = (const std::vector< std::vector< double > >)0 );



    /** \brief Remove the points whose indices are given from this topology.
     *
     * Elements corresponding to these points are removed form the mechanical object's state vectors.
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
     * \sa removePointsWarning
     */
    virtual void removePointsProcess(const unsigned int nPoints, std::vector<unsigned int> &indices);



    /** \brief Reorder this topology.
     *
     * \see MechanicalObject::renumberValues
     */
    virtual void renumberPointsProcess( const std::vector<unsigned int> &index );



    /** \brief Fuse the edges.
     *
     */
    virtual void fuseEdgesProcess(const std::vector< std::pair< unsigned int, unsigned int > >& edgesPair);



    /** \brief Split the edges.
     *
     */
    virtual void splitEdgesProcess( std::vector<unsigned int> &indices,
            const std::vector< std::vector< double > >& baryCoefs = (const std::vector< std::vector< double > >)0);

protected:
    void addEdge(Edge e);

    template< typename DataTypes >
    friend class EdgeSetTopologyLoader;

};



/**
 * A class that performs topology algorithms on an EdgeSet.
 */
template < class DataTypes >
class EdgeSetTopologyAlgorithms : public PointSetTopologyAlgorithms<DataTypes>
{

public:

    /** \brief Remove a set  of edges
    @param edges an array of edge indices to be removed (note that the array is not const since it needs to be sorted)
    *
    */
    virtual void removeEdges(std::vector< unsigned int >& edges);

    /** \brief add a set  of edges
    @param edges an array of pair of vertex indices describing the edge to be created
    @param ancestors for each edge to be created provides an array of edge ancestors (optional)
    @param baryCoefs for each edge provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addEdges(const std::vector< Edge >& edges,
            const std::vector< std::vector< unsigned int > > & ancestors= (const std::vector< std::vector<unsigned int > >) 0 ,
            const std::vector< std::vector< double > >& baryCoefs= (const std::vector< std::vector< double > >)0) ;


    /** \brief Fuse a list of pair edges.
     *
     */
    virtual void fuseEdges(const std::vector< std::pair< unsigned int, unsigned int > >& edgesPair);

    /** \brief Split an array of edges. On each edge, a vertex is created based on its barycentric coordinates
     *
     */
    virtual void splitEdges( std::vector<unsigned int> &indices,
            const std::vector< std::vector< double > >& baryCoefs = (const std::vector< std::vector< double > >)0 );


    EdgeSetTopologyAlgorithms(Sofa::Core::BasicTopology *top) : PointSetTopologyAlgorithms(top)
    {
    }
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


    EdgeSetGeometryAlgorithms(Sofa::Core::BasicTopology *top) : PointSetGeometryAlgorithms<DataTypes>(top)
    {
    }
    /// computes the length of edge no i and returns it
    Real computeEdgeLength(const unsigned int i) const;
    /// computes the edge length of all edges are store in the array interface
    void computeEdgeLength( BasicArrayInterface<Real> &ai) const;
    /// computes the initial length of edge no i and returns it
    Real computeRestEdgeLength(const unsigned int i) const;

};


/** Describes a topological object that only consists as a set of points :
it is a base class for all topological objects */
template<class DataTypes>
class EdgeSetTopology : public PointSetTopology <DataTypes>
{

public:
    EdgeSetTopology(Core::MechanicalObject<DataTypes> *obj);


    virtual void init();
    /** \brief Returns the EdgeSetTopologyContainer object of this EdgeSetTopology.
     */
    EdgeSetTopologyContainer *getEdgeSetTopologyContainer() const
    {
        return (EdgeSetTopologyContainer *)m_topologyContainer;
    }
    /** \brief Returns the EdgeSetTopologyAlgorithms object of this EdgeSetTopology.
     */
    EdgeSetTopologyAlgorithms<DataTypes> *getEdgeSetTopologyAlgorithms() const
    {
        return (EdgeSetTopologyAlgorithms<DataTypes> *)m_topologyAlgorithms;
    }
    /** \brief Returns the EdgeSetTopologyAlgorithms object of this EdgeSetTopology.
     */
    EdgeSetGeometryAlgorithms<DataTypes> *getEdgeSetGeometryAlgorithms() const
    {
        return (EdgeSetGeometryAlgorithms<DataTypes> *)m_geometryAlgorithms;
    }

};

} // namespace Components

} // namespace Sofa

#endif
