#ifndef SOFA_COMPONENTS_EDGESETTOPOLOGY_H
#define SOFA_COMPONENTS_EDGESETTOPOLOGY_H

#include "Sofa/Component/Topology/PointSetTopology.h"

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Sofa::Core;

#define ADDEDGE_CODE 20
#define REMOVEDGE_CODE 21

/** indicates that the indices of two vertices are being swapped */
class AddEdge : public TopologyModification
{
public:
    typedef int index_type;
protected :
    unsigned int nbEdges;
public:
    AddEdge(const unsigned int n)
    {
        nbEdges=n;
    }
    unsigned int getNbEdges() const
    {
        return nbEdges;
    }
};


/** indicates that the indices of two vertices are being swapped */
class RemoveEdge : public TopologyModification
{
public:
    typedef int index_type;
protected :
    index_type edge;
public:
    RemoveEdge(const index_type e)
    {
        edge=e;
    }
};

/** a class that stores a set of edges and provides access
to the neighbors of each vertex */
class EdgeSetTopologyContainer : public TopologyContainer
{
public:
    typedef int index_type;
    typedef int component_index_type;
    typedef fixed_array<index_type, 2> Edge;

    typedef std::set<index_type> NeighborSet;
    typedef std::vector<Edge> EdgeArray;
    typedef std::vector<NeighborSet> NeighborSetArray;
    typedef std::vector<component_index_type> ComponentIndexArray;
    typedef std::vector<unsigned int> ComponentSizeArray;

protected:
    std::vector<Edge> edgeArray;
    NeighborSetArray neighborArray;
    bool validNeighborArray;
    void computeNeighborArray();
public:
    /// give a read-only access to the edge array
    const EdgeArray &getEdgeArray() const;
    /// give a read-only access to the neighbor array
    const NeighborSetArray &getNeighborSetArray() ;

    int getNumberOfEdges() const;

    const Edge& getEdge(const index_type i) const;

    const NeighborSet & getNeighbors(const index_type i);

    void getConnectedComponents(unsigned int &nbComponents,
            ComponentIndexArray &componentIndexArray, ComponentSizeArray &componentSizeArray);
    /*** fills in the array neighborArray the indices of the vertices whose topological
    distance from the vertex origin is exactly n. If n =1 then we get the same result
    than the getNeighbors() function. If n=2, then we get the set of neighbors of the
    neighbors, etc..*/
    void getNRing(const index_type origin,
            const unsigned int n, NeighborSet &neighborSet);
    /*** fills in the array neighborArray the indices of the vertices whose topological
    distance from the vertex origin is at most n. If n =1 then we get the same result
    than the getNeighbors() function. If n=2, then we get the set of vertices whose neighbors
    are the neighbors vertex origin., etc..*/
    void getNDisc(const index_type origin,
            const unsigned int n, NeighborSet &neighborSet);
};

template<class TDataTypes>
class EdgeSetTopologyModifier : public PointSetTopologyModifier<TDataTypes>
{
public:
    typedef unsigned int index_type;
    void removeEdge(const index_type e);
    /// add a number of edges
    void addEdge(const unsigned int nEdges);
    void renumberMesh(std::vector<index_type> *indexArray);
};

template<class TDataTypes>
class EdgeSetTopologyAlgorithms : public TopologyAlgorithms
{
public:
    typedef typename DataTypes::Real Real;
    void reverseCuthillMcKeeRenumbering(const index_type i1,
            const index_type i2);
    void reverseCuthillMcKeeRenumbering();
    /*** Split an edge into two edges. The middle point is inserted along
    the edge. Its  barycentric coordinate with respect to the first vertex of
    the edge is given by weight (it is (1-weight) for the second vertex) */
    void splitEdge(const Real weight=0.5);
};


template<class TDataTypes>
class EdgeSetGeometryAccessor : public PointSetGeometryAccessor<TDataTypes>
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    Coord getAverageNeighbors(const index_type i);
    Real getNRingAverageSize(const index_type i);
}

template<class TDataTypes>
class EdgeSetTopology : public PointSetTopology<TDataTypes>
{
public:
    EdgeSetTopology();
    virtual void init();
};

} // namespace Components

} // namespace Sofa

#endif
