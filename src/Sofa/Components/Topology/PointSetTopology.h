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



/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////



/** indicates that the indices of two points are being swapped */
class PointsIndicesSwap : public TopologyChange
{

public:
    int index[2];

    PointsIndicesSwap(const int i1,const int i2)
    {
        index[0]=i1;
        index[1]=i2;
    }

};



/** indicates that some points were added */
class PointsAdded : public TopologyChange
{

public:
    unsigned int nVertices;

    PointsAdded(const unsigned int nV) : nVertices(nV)
    {
    }

    unsigned int getNbAddedVertices() const
    {
        return nVertices;
    }

};



/** indicates that some points are about to be removed */
class PointsRemoved : public TopologyChange
{

public:
    std::vector<int> removedVertexArray;

public:
    PointsRemoved(std::vector<int> _vArray) : removedVertexArray(_vArray)
    {
    }

    const std::vector<int> &getArray() const
    {
        return removedVertexArray;
    }

};



/** indicates that the indices all points have been reordered */
class VertexRenumbering : public TopologyChange
{

public:
    std::vector<int> indexArray;

    VertexRenumbering() {}
    std::vector<int> &getIndexArray()
    {
        return indexArray;
    }

};



/////////////////////////////////////////////////////////
/// PointSetTopology objects
/////////////////////////////////////////////////////////


/** a class that stores a set of pointss and provides access
to each point */
class PointSetTopologyContainer : public TopologyContainer
{

protected:
    std::vector<int> vertexArray;

    std::vector<bool> vertexInSetArray;

public:
    /// give a read-only access to the edge array
    const std::vector<int> &getVertexArray() const;

    int getVertex(const int i) const;

    unsigned int getNumberOfVertices() const;

    const std::vector<bool> &getVertexInSetArray() const;

    bool isVertexInSet(const int i) const;

    PointSetTopologyContainer(BasicTopology *top);

    PointSetTopologyContainer(BasicTopology *top, std::vector<int> &_vertexArray);

    template<class DT> friend class PointSetTopologyModifier;

};



/**
 * A class that can apply basic transformations on a set of points.
 */
template<class TDataTypes>
class PointSetTopologyModifier : public TopologyModifier
{

public:
    typedef typename TDataTypes::VecCoord VecCoord;

    void swapVertices(const int i1,const int i2);

    void addVertices(const unsigned int nVertices, VecCoord &X = (VecCoord)0 );

    void removeVertices(const unsigned int nVertices, std::vector<int> &indices);

};



/**
 * A class that can perform some geometric computation on a set of points.
 */
template<class DataTypes>
class PointSetGeometryAlgorithms : public GeometryAlgorithms
{

public:
    typedef typename DataTypes::Real Real;

    typedef typename DataTypes::Coord Coord;

    typedef typename DataTypes::VecCoord VecCoord;

    PointSetGeometryAlgorithms(BasicTopology *top) : GeometryAlgorithms(top)
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

public:
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

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGY_H
