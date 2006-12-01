#ifndef SOFA_COMPONENTS_POINTSETTOPOLOGY_H
#define SOFA_COMPONENTS_POINTSETTOPOLOGY_H

//#include <stdlib.h>
#include <vector>
//#include <string>
#include <Sofa/Core/BasicTopology.h>
#include <Sofa/Core/MechanicalObject.h>
#include <Sofa/Components/Common/fixed_array.h>

namespace Sofa
{

namespace Components
{


/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////



/** indicates that the indices of two points are being swapped */
class PointsIndicesSwap : public Core::TopologyChange
{

public:
    unsigned int index[2];

    PointsIndicesSwap(const unsigned int i1,const unsigned int i2) : Core::TopologyChange(Core::POINTSINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

};



/** indicates that some points were added */
class PointsAdded : public Core::TopologyChange
{

public:
    unsigned int nVertices;

    std::vector< std::vector< unsigned int > > ancestorsList;

    PointsAdded(const unsigned int nV, const std::vector< std::vector< unsigned int > > &ancestors = (std::vector< std::vector< unsigned int > >)0)
        : Core::TopologyChange(Core::POINTSADDED), nVertices(nV), ancestorsList(ancestors)
    { }

    unsigned int getNbAddedVertices() const
    {
        return nVertices;
    }

};



/** indicates that some points are about to be removed */
class PointsRemoved : public Core::TopologyChange
{

public:
    std::vector<unsigned int> removedVertexArray;

public:
    PointsRemoved(const std::vector<unsigned int> _vArray) : Core::TopologyChange(Core::POINTSREMOVED), removedVertexArray(_vArray)
    {
    }

    const std::vector<unsigned int> &getArray() const
    {
        return removedVertexArray;
    }

};



/** indicates that the indices of all points have been reordered */
class PointsRenumbering : public Core::TopologyChange
{

public:
    std::vector<unsigned int> indexArray;

    PointsRenumbering(const std::vector< unsigned int > &indices = (std::vector< unsigned int >)0) : Core::TopologyChange(Core::POINTSRENUMBERING), indexArray(indices) {}

    const std::vector<unsigned int> &getIndexArray() const
    {
        return indexArray;
    }

};



/////////////////////////////////////////////////////////
/// PointSetTopology objects
/////////////////////////////////////////////////////////


/** a class that stores a set of points and provides access
to each point */
class PointSetTopologyContainer : public Core::TopologyContainer
{

private:
    /** \brief Creates the PointSetIndex.
     *
     * This function is only called if the PointSetIndex member is required.
     * PointSetIndex[i] contains -1 if the ith DOF is not part of this topology,
     * and its index in this topology otherwise.
     */
    void createPointSetIndex();

protected:
    std::vector<unsigned int> m_DOFIndex;
    std::vector<int> m_PointSetIndex;


public:
    /** \brief Returns the PointSetIndex.
     *
     * See getPointSetIndex(const unsigned int i) for more explanation.
     */
    const std::vector<int> &getPointSetIndexArray();



    /** \brief Returns the index in this topology of the point corresponding to the ith DOF of the mechanical object, or -1 if the ith DOF is not in this topology.
     *
     */
    int getPointSetIndex(const unsigned int i);



    /** \brief Returns the number of vertices in this topology.
     *
     */
    unsigned int getNumberOfVertices() const;



    /** \brief Returns the DOFIndex.
     *
     * See getDOFIndex(const int i) for more explanation.
     */
    const std::vector<unsigned int> &getDOFIndexArray() const;



    /** \brief Returns the DOFIndex.
     *
     * See getDOFIndex(const int i) for more explanation.
     */
    std::vector<unsigned int> &getDOFIndexArray();



    /** \brief Returns the index in the mechanical object of the DOF corresponding to the ith point of this topology.
     *
     */
    unsigned int getDOFIndex(const int i) const;

    PointSetTopologyContainer(Core::BasicTopology *top);

    PointSetTopologyContainer(Core::BasicTopology *top, std::vector<unsigned int> &DOFIndex);

    template <typename DataTypes>
    friend class PointSetTopologyModifier;

    //friend class PointSetTopologicalMapping;

};



/**
 * A class that can apply basic transformations on a set of points.
 */
template<class TDataTypes>
class PointSetTopologyModifier : public Core::TopologyModifier
{

public:
    typedef typename TDataTypes::VecCoord VecCoord;
    typedef typename TDataTypes::VecDeriv VecDeriv;

    /** \brief Swap points i1 and i2.
     *
     */
    virtual void swapPoints(const int i1,const int i2);



    /** \brief Sends a message to warn that some points were added in this topology.
     *
     * \sa addPointsProcess
     */
    void addPointsWarning(const unsigned int nPoints, const std::vector< std::vector< unsigned int > > &ancestors = (std::vector< std::vector< unsigned int > >) 0);



    /** \brief Add some points to this topology.
     *
     * State vectors may be defined for these new points.
     *
     * \sa addPointsWarning
     */
    virtual void addPointsProcess(const unsigned int nPoints, const VecCoord &X = (VecCoord &)0, const VecCoord &X0 = (VecCoord &)0,
            const VecDeriv &V = (VecDeriv &)0, const VecDeriv &V0 = (VecDeriv &)0,
            const VecDeriv &F = (VecDeriv &)0, const VecDeriv &DX = (VecDeriv &)0 );


    /** \brief Add some points to this topology.
     *
     * Use a list of ancestors to create the new points.
     *
     * \sa addPointsWarning
     */
    virtual void addPointsProcess(const unsigned int nPoints, const std::vector< std::vector< unsigned int > > &ancestors );



    /** \brief Sends a message to warn that some points are about to be deleted.
     *
     * \sa removePointsProcess
     */
    void removePointsWarning(const unsigned int nPoints, const std::vector<unsigned int> &indices);



    /** \brief Remove the points whose indices are given from this topology.
     *
     * Elements corresponding to these points are removed form the mechanical object's state vectors.
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
     * \sa removePointsWarning
     */
    virtual void removePointsProcess(const unsigned int nPoints, const std::vector<unsigned int> &indices);



    /** \brief Reorder this topology.
     *
     * \see MechanicalObject::renumberValues
     */
    virtual void renumberPointsProcess( const std::vector<unsigned int> &index );

};



/**
 * A class that can perform some geometric computation on a set of points.
 */
template<class DataTypes>
class PointSetGeometryAlgorithms : public Core::GeometryAlgorithms
{

public:
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

public:
    Core::MechanicalObject<TDataTypes> *object;

    //void createNewVertices() const;

    //void removeVertices() const;

public:
    PointSetTopology(Core::MechanicalObject<TDataTypes> *obj);

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

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGY_H
