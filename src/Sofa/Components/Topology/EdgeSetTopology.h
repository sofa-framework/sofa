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


/*
** indicates that the indices of two edges are being swapped *
class EdgesIndicesSwap : public Core::TopologyChange  {

public:
    unsigned int index[2];

    EdgesIndicesSwap(const unsigned int i1,const unsigned int i2) : Core::TopologyChange(Core::EDGESINDICESSWAP) {
		index[0]=i1;
		index[1]=i2;
	}

};
*/


/** indicates that some edges were added */
class EdgesAdded : public Core::TopologyChange
{

public:
    unsigned int nEdges;

    std::vector< Edge > edge;

    std::vector< std::vector< Edge > > ancestorsList;

    std::vector< std::vector< double > > coefs;

    EdgesAdded(const unsigned int nE,
            const std::vector< Edge >& edgesList = (const std::vector< Edge >)0,
            const std::vector< std::vector< Edge > >& ancestors = (const std::vector< std::vector< Edge > >)0,
            const std::vector< std::vector< double > >& baryCoefs = (const std::vector< std::vector< double > >)0)
        : Core::TopologyChange(Core::EDGESADDED), nEdges(nE), edge(edgesList), ancestorsList(ancestors), coefs(baryCoefs)
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
    EdgesRemoved(const std::vector<unsigned int> _eArray) : Core::TopologyChange(Core::POINTSREMOVED), removedEdgesArray(_eArray)
    {
    }

    const std::vector<unsigned int> &getArray() const
    {
        return removedEdgesArray;
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
    void createEdgeShellsArray();

protected:
    //std::vector<unsigned int> m_DOFIndex;
    //std::vector<int> m_EdgeSetIndex;
    std::vector<Edge> m_edge;
    std::vector< std::vector< unsigned int > > m_edgeShell;



public:
    /** \brief Returns the Edge array.
     *
     */
    const std::vector<Edge> &getEdgeArray();



    /** \brief Returns the ith Edge.
     *
     */
    Edge &getEdge(const unsigned int i);



    /** \brief Returns the number of edges in this topology.
     *
     */
    unsigned int getNumberOfEdges() const;



    /** \brief Returns the Edge Shells array.
     *
     */
    const std::vector< std::vector<unsigned int> > &getEdgeShellsArray() const;



    /** \brief Returns the Edge Shells array.
     *
     */
    //std::vector < std::vector<unsigned int> > &getEdgeShellsArray();



    /** \brief Returns the edge shell of the ith DOF.
     *
     */
    const std::vector< unsigned int > &getEdgeShell(const unsigned int i) const;

    //EdgeSetTopologyContainer(Core::BasicTopology *top);

    EdgeSetTopologyContainer(Core::BasicTopology *top, const std::vector< unsigned int > &DOFIndex = (const std::vector< unsigned int >)0,
            const std::vector< Edge >         &edges    = (const std::vector< Edge >)        0 );

    template< typename DataTypes >
    friend class EdgeSetTopologyModifier;
    //template< typename DataTypes >
    //friend class EdgeSetTopologyContainer;

    //friend class EdgeSetTopologicalMapping;

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


    /** \brief Sends a message to warn that some edges were added in this topology.
     *
     * \sa addEdgesProcess
     */
    void addEdgesWarning(const unsigned int nEdges, const std::vector< std::vector< Edge > > &ancestors = (const std::vector< std::vector< Edge > > ) 0 );



    /** \brief Add some edges to this topology.
     *
     * \sa addEdgesWarning
     */
    virtual void addEdgesProcess(const std::vector< Edge > &edges);




    /** \brief Sends a message to warn that some points are about to be deleted.
     *
     * \sa removeEdgesProcess
     */
    void removeEdgesWarning(const std::vector<unsigned int> &edges);



    /** \brief Remove the points whose indices are given from this topology.
     *
     * Elements corresponding to these points are removed form the mechanical object's state vectors.
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
     * \sa removeEdgesWarning
     *
     * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
     */
    virtual void removeEdgesProcess(const unsigned int nEdges, std::vector<unsigned int> &indices);



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



};



/**
 * A class that performs complex algorithms on an EdgeSet.
 */
template < class DataTypes >
class EdgeSetTopologyAlgorithms : public PointSetTopologyAlgorithms
{

public:

    /** \brief Fuse the edges.
     *
     */
    virtual void fuseEdgesProcess(const std::vector< Edge >& edgesPair);



    /** \brief Split the edges.
     *
     */
    virtual void splitEdgesProcess(const std::vector<unsigned int> indices);

};


/**
 * A class that can perform some geometric computation on a set of points.
 */
template<class DataTypes>
class EdgeSetGeometryAlgorithms : public PointSetGeometryAlgorithms <DataTypes>
{

public:
    typedef typename DataTypes::Real Real;

    typedef typename DataTypes::Coord Coord;

    typedef typename DataTypes::VecCoord VecCoord;

    EdgeSetGeometryAlgorithms(Core::BasicTopology *top) : PointSetGeometryAlgorithms<DataTypes>(top)
    {
    }

};



/** Describes a topological object that only consists as a set of points :
it is a base class for all topological objects */
template<class DataTypes>
class EdgeSetTopology : public PointSetTopology <DataTypes>
{

public:
    EdgeSetTopology(Core::MechanicalObject<DataTypes> *obj) : PointSetTopology<DataTypes>( obj )
    {
    }

    virtual void init();

};

} // namespace Components

} // namespace Sofa

#endif
