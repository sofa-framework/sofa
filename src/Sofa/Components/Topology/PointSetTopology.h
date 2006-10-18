#ifndef SOFA_COMPONENTS_POINTSETTOPOLOGY_H
#define SOFA_COMPONENTS_POINTSETTOPOLOGY_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <Sofa/Core/BasicTopology.h>
#include <Sofa/Core/MechanicalObject.h>
#include <Sofa/Components/Common/fixed_array.h>

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Sofa::Core;


#define VERTEXSWAP_CODE 10
#define VERTEXADDED_CODE 11
#define VERTEXREMOVED_CODE 12
#define VERTEXRENUMBERING_CODE 13

/** indicates that the indices of two vertices are being swapped */
class VertexSwap : public TopologyChange
{
public:
    typedef  int index_type;
protected :
    index_type index[2];
public:
    VertexSwap(const index_type i1,const index_type i2)
    {
        index[0]=i1;
        index[1]=i2;
    }
};
/** indicates that the indices of two vertices are being swapped */
class VertexAdded : public TopologyChange
{
protected:
    unsigned int nVertices;
public:
    VertexAdded(const unsigned int nV) : nVertices(nV)
    {
    }
    unsigned int getNbAddedVertices() const
    {
        return nVertices;
    }

};
/** indicates that the indices of two vertices are being swapped */
class VertexRemoved : public TopologyChange
{
public:
    typedef  int index_type;
protected:
    unsigned int nVertices;
    std::vector<index_type> removedVertexArray;
public:
    VertexRemoved(const unsigned int nV,std::vector<index_type> _vArray)
        : nVertices(nV), removedVertexArray(_vArray)
    {
    }
    unsigned int getNbRemovedVertices() const
    {
        return nVertices;
    }
    const std::vector<index_type> &getArray() const
    {
        return removedVertexArray;
    }
};
/** indicates that the indices of two vertices are being swapped */
class VertexRenumbering : public TopologyChange
{
public:
    typedef  int index_type;
protected :
    std::vector<index_type> indexArray;
public:
    VertexRenumbering() {}
    std::vector<index_type> &getIndexArray()
    {
        return indexArray;
    }
};

template<class TDataTypes>
class PointSetTopologyModifier : public TopologyModifier
{
public:
    typedef int indexType;
    void swapVertices(const indexType i1,const indexType i2);
    void addVertices(const unsigned int nVertices);
    void removeVertices(const unsigned int nVertices);
};

/** a class that stores a set of edges and provides access
to the neighbors of each vertex */
class PointSetTopologyContainer : public TopologyContainer
{
public:
    typedef int indexType;
    typedef std::vector<indexType>  VertexArray;
    typedef std::vector<bool>  InSetArray;

protected:
    std::vector<indexType> vertexArray;
    std::vector<bool> vertexInSetArray;
public:
    /// give a read-only access to the edge array
    const VertexArray &getVertexArray() const;
    indexType getVertex(const indexType i) const;

    unsigned int getNumberOfVertices() const;

    const InSetArray &getVertexInSetArray() const;
    bool isVertexInSet(const indexType i) const;

    PointSetTopologyContainer(BasicTopology *top);
    PointSetTopologyContainer(BasicTopology *top,std::vector<int> &_vertexArray);

    template<class DT> friend class PointSetTopologyModifier;
};


template<class TDataTypes>
class PointSetGeometryAlgorithms : public GeometryAlgorithms
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    PointSetGeometryAlgorithms(Core::BasicTopology *top) : GeometryAlgorithms(top)
    {
    }
    /** return the centroid of the set of points */
    Coord getPointSetCenter() const;

    /** return the centre and a radius of a sphere enclosing the  set of points (may not be the smalled one) */
    void getEnclosingSphere(Coord &center,Real &radius) const;
    /** return the axis aligned bounding box : index 0 = xmin, index 1=ymin,
    index 2 = zmin, index 3 = xmax, index 4 = ymax, index 5=zmax */
    void getAABB(Real bb[6]) const;
};

/** Describes a topological object that only consists as a set of points :
it is a base class for all topological objects */
template<class TDataTypes>
class PointSetTopology : public Core::BasicTopology
{
protected:
    MechanicalObject<TDataTypes> *object;
    void createNewVertices() const;
    void removeVertices() const;
public:
    PointSetTopology(MechanicalObject<TDataTypes> *obj);
    MechanicalObject<TDataTypes> *getDOF() const
    {
        return object;
    }
    virtual void propagateTopologicalChanges();
    virtual void init();
};

} // namespace Components

} // namespace Sofa

#endif
