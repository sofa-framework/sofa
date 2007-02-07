#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_H

//#include <stdlib.h>
#include <vector>
//#include <string>
#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/helper/fixed_array.h>

namespace sofa
{

namespace component
{

namespace topology
{

/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////



/** indicates that the indices of two points are being swapped */
class PointsIndicesSwap : public core::componentmodel::topology::TopologyChange
{

public:
    unsigned int index[2];

    PointsIndicesSwap(const unsigned int i1,const unsigned int i2) : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

};



/** indicates that some points were added */
class PointsAdded : public core::componentmodel::topology::TopologyChange
{

public:
    unsigned int nVertices;

    std::vector< std::vector< unsigned int > > ancestorsList;

    std::vector< std::vector< double       > > coefs;

    PointsAdded(const unsigned int nV,
            const std::vector< std::vector< unsigned int > >& ancestors = (const std::vector< std::vector< unsigned int > >)0,
            const std::vector< std::vector< double       > >& baryCoefs = (const std::vector< std::vector< double       > >)0)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSADDED), nVertices(nV), ancestorsList(ancestors), coefs(baryCoefs)
    { }

    unsigned int getNbAddedVertices() const
    {
        return nVertices;
    }

};



/** indicates that some points are about to be removed */
class PointsRemoved : public core::componentmodel::topology::TopologyChange
{

public:
    std::vector<unsigned int> removedVertexArray;

public:
    PointsRemoved(const std::vector<unsigned int>& _vArray) : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSREMOVED), removedVertexArray(_vArray)
    {
    }

    const std::vector<unsigned int> &getArray() const
    {
        return removedVertexArray;
    }

};



/** indicates that the indices of all points have been reordered */
class PointsRenumbering : public core::componentmodel::topology::TopologyChange
{

public:
    std::vector<unsigned int> indexArray;

    PointsRenumbering(const std::vector< unsigned int >& indices = (const std::vector< unsigned int >)0)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSRENUMBERING), indexArray(indices)
    { }

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
class PointSetTopologyContainer : public core::componentmodel::topology::TopologyContainer
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
    const std::vector<int>& getPointSetIndexArray();



    /** \brief Returns the index in this topology of the point corresponding to the ith DOF of the mechanical object, or -1 if the ith DOF is not in this topology.
     *
     */
    int getPointSetIndex(const unsigned int i);

    /** \brief Returns the number of vertices in this index array
    *
    */
    unsigned int getPointSetIndexSize() const;

    /** \brief Returns the number of vertices in this topology.
     *
     */
    unsigned int getNumberOfVertices() const;

    /** \brief Returns the DOFIndex.
     *
     * See getDOFIndex(const int i) for more explanation.
     */
    const std::vector<unsigned int>& getDOFIndexArray() const;

    /** \brief Returns the index in the mechanical object of the DOF corresponding to the ith point of this topology.
     *
     */
    unsigned int getDOFIndex(const int i) const;

    PointSetTopologyContainer(core::componentmodel::topology::BaseTopology *top);

    PointSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, const std::vector<unsigned int>& DOFIndex);

    template <typename DataTypes>
    friend class PointSetTopologyModifier;
protected:
    /** \brief Returns the DOFIndex.
    *
    * See getDOFIndex(const int i) for more explanation.
    */
    std::vector<unsigned int>& getDOFIndexArrayForModification();
    /** \brief Returns the PointSetIndex array for modification.
     */
    std::vector<int>& getPointSetIndexArrayForModification();

    //friend class PointSetTopologicalMapping;

};

// forward declaration
template< typename DataTypes > class PointSetTopologyLoader;

/**
 * A class that can apply basic transformations on a set of points.
 */

template<class DataTypes>
class PointSetTopologyModifier : public core::componentmodel::topology::TopologyModifier
{

public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    PointSetTopologyModifier(core::componentmodel::topology::BaseTopology *top) : TopologyModifier(top)
    {
    }

    /** \brief Swap points i1 and i2.
     *
     */
    virtual void swapPoints(const int i1,const int i2);


    /** \brief Build a point set topology from a file : also modifies the MechanicalObject
     *
     */
    virtual bool load(const char *filename);


    /** \brief Sends a message to warn that some points were added in this topology.
     *
     * \sa addPointsProcess
     */
    void addPointsWarning(const unsigned int nPoints,
            const std::vector< std::vector< unsigned int > >& ancestors = (const std::vector< std::vector< unsigned int > >) 0,
            const std::vector< std::vector< double       > >& coefs     = (const std::vector< std::vector< double       > >) 0);



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
     *
     * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
     */
    virtual void removePointsProcess(const unsigned int nPoints, std::vector<unsigned int> &indices);



    /** \brief Reorder this topology.
     *
     * \see MechanicalObject::renumberValues
     */
    virtual void renumberPointsProcess( const std::vector<unsigned int> &index );

protected:
    /// modifies the mechanical object and creates the point set container
    void loadPointSet(PointSetTopologyLoader<DataTypes> *);


};



/** A class that performs complex algorithms on a PointSet.
 *
 */
template<class DataTypes>
class PointSetTopologyAlgorithms : public core::componentmodel::topology::TopologyAlgorithms
{
    // no methods implemented yet
public:
    PointSetTopologyAlgorithms(core::componentmodel::topology::BaseTopology *top) : TopologyAlgorithms(top)
    {
    }
};


/**
 * A class that can perform some geometric computation on a set of points.
 */
template<class DataTypes>
class PointSetGeometryAlgorithms : public core::componentmodel::topology::GeometryAlgorithms
{

public:
    typedef typename DataTypes::Real Real;

    typedef typename DataTypes::Coord Coord;

    typedef typename DataTypes::VecCoord VecCoord;

    PointSetGeometryAlgorithms(core::componentmodel::topology::BaseTopology *top) : GeometryAlgorithms(top)
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
template<class DataTypes>
class PointSetTopology : public core::componentmodel::topology::BaseTopology
{

public:
    component::MechanicalObject<DataTypes> *object;

    //void createNewVertices() const;

    //void removeVertices() const;

public:
    PointSetTopology(component::MechanicalObject<DataTypes> *obj);

    component::MechanicalObject<DataTypes> *getDOF() const
    {
        return object;
    }

    virtual void propagateTopologicalChanges();

    virtual void init();

    /** \brief Build a topology from a file : call the load member function in the modifier object
    *
    */
    virtual bool load(const char *filename);


    /** \brief Return the number of DOF in the mechanicalObject this Topology deals with.
     *
     */
    virtual unsigned int getDOFNumber() const { return object->getSize(); }

protected:
    PointSetTopology(component::MechanicalObject<DataTypes> *obj,const PointSetTopology *);


};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGY_H
