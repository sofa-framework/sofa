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

    std::vector< std::vector< Edge > > ancestorsList;

    EdgesAdded(const unsigned int nE, std::vector< std::vector< Edge > > &ancestors = (std::vector< std::vector< Edge > >)0 )
        : Core::TopologyChange(Core::EDGESADDED), nEdges(nE), ancestorsList(ancestors)
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
    EdgesRemoved(std::vector<unsigned int> _eArray) : Core::TopologyChange(Core::POINTSREMOVED), removedEdgesArray(_eArray)
    {
    }

    const std::vector<unsigned int> &getArray() const
    {
        return removedEdgesArray;
    }

};



/** indicates that the indices of all edges have been reordered */
class EdgesRenumbering : public Core::TopologyChange
{

public:
    std::vector<unsigned int> indexArray;

    EdgesRenumbering() : Core::TopologyChange(Core::EDGESRENUMBERING) {}

    const std::vector<unsigned int> &getIndexArray() const
    {
        return indexArray;
    }

};


/*
** indicates that some edges are about to be fused. *
class EdgesFused : public Core::TopologyChange  {

public:
	std::vector<unsigned int> indexArray;

    EdgesFused() : Core::TopologyChange(Core::EDGESFUSED) {}

	const std::vector<unsigned int> &getIndexArray() const {
		return indexArray;
	}

};



** indicates that some edges were split. *
class EdgesSplit : public Core::TopologyChange  {

public:
	std::vector<unsigned int> indexArray;

    EdgesSplit() : Core::TopologyChange(Core::EDGESSPLIT) {}

	const std::vector<unsigned int> &getIndexArray() const {
		return indexArray;
	}

};
*/


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
    std::vector< std::vector<unsigned int> > m_edgeShell;



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

    EdgeSetTopologyContainer(Core::BasicTopology *top);

    EdgeSetTopologyContainer(Core::BasicTopology *top, std::vector<unsigned int> &DOFIndex, std::vector<Edge> &edges);

    template <typename DataTypes>
    friend class EdgeSetTopologyModifier;

    //friend class EdgeSetTopologicalMapping;

};



/**
 * A class that can apply basic transformations on a set of points.
 */
template<class TDataTypes>
class EdgeSetTopologyModifier : public PointSetTopologyModifier <TDataTypes>
{

public:
    typedef typename TDataTypes::VecCoord VecCoord;
    typedef typename TDataTypes::VecDeriv VecDeriv;


    /** \brief Sends a message to warn that some edges were added in this topology.
     *
     * \sa addEdgesProcess
     */
    void addEdgesWarning(const unsigned int nEdges, const std::vector< std::vector< Edge > > &ancestors = (std::vector< std::vector< Edge > > ) 0 );



    /** \brief Add some edges to this topology.
     *
     * \sa addEdgesWarning
     */
    virtual void addEdgesProcess(const std::vector< Edge > &edges);



    /** \brief Add some edges to this topology based on ancestors.
     *
     * \sa addEdgesWarning
     */
    virtual void addEdgesProcess(const std::vector< std::vector< Edge > > &ancestors);



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
     */
    virtual void removeEdgesProcess(const unsigned int nEdges, const std::vector<unsigned int> &indices);



    /** \brief Reorder this topology.
     *
     * \see MechanicalObject::renumberValues
     */
    virtual void renumberEdgesProcess( const std::vector<unsigned int> &index );



    // what about the warning? should it be called before? after? since there are both creations AND destruction in this...
    /** \brief Fuse the edges.
     *
     */
    virtual void fuseEdgesProcess(const unsigned int i1, const unsigned int i2);



    // what about the warning? should it be called before? after? since there are both creations AND destruction in this...
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

    EdgeSetGeometryAlgorithms(Core::BasicTopology *top) : GeometryAlgorithms(top)
    {
    }

    /** return the centroid of the set of points */
    Coord getEdgeSetCenter() const;

    /** return the centre and a radius of a sphere enclosing the  set of points (may not be the smalled one) */
    void getEnclosingSphere(Coord &center,Real &radius) const;

    /** return the axis aligned bounding box : index 0 = xmin, index 1=ymin,
    index 2 = zmin, index 3 = xmax, index 4 = ymax, index 5=zmax */
    void getAABB(Real bb[6]) const;

};



/** Describes a topological object that only consists as a set of points :
it is a base class for all topological objects */
template<class TDataTypes>
class EdgeSetTopology : public PointSetTopology <TDataTypes>
{

public:
    Core::MechanicalObject<TDataTypes> *object;

    //void createNewVertices() const;

    //void removeVertices() const;

public:
    EdgeSetTopology(Core::MechanicalObject<TDataTypes> *obj);

    Core::MechanicalObject<TDataTypes> *getDOF() const
    {
        return object;
    }

    virtual void propagateTopologicalChanges();

    virtual void init();



    /** \brief Return the number of DOF in the mechanicalObject this Topology deals with.
     *
     */
    virtual unsigned int getDOFNumber() { return object->getSize(); }



};

} // namespace Components

} // namespace Sofa

#endif
