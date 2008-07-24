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
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_H

#include <sofa/helper/vector.h>
#include <sofa/core/componentmodel/topology/Topology.h>		// TopologyChange
#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/component/MechanicalObject.h>

namespace sofa
{

namespace component
{

namespace topology
{
template<class DataTypes>
class PointSetTopology;

template<class DataTypes>
class PointSetTopologyAlgorithms;

template<class DataTypes>
class PointSetGeometryAlgorithms;

template< typename DataTypes >
class PointSetTopologyLoader;

template<class DataTypes>
class PointSetTopologyModifier;

class PointSetTopologyContainer;

class PointsIndicesSwap;
class PointsAdded;
class PointsRemoved;
class PointsRenumbering;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::PointID PointID;

/////////////////////////////////////////////////////////
/// PointSetTopology objects
/////////////////////////////////////////////////////////

/** Describes a topological object that only consists as a set of points :
it is a base class for all topological objects */
template<class DataTypes>
class PointSetTopology : public core::componentmodel::topology::BaseTopology
{
public:
    PointSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~PointSetTopology() {}

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    virtual void parse(sofa::core::objectmodel::BaseObjectDescription* arg);

    virtual void init();

    /** \brief Returns the PointSetTopologyContainer object of this PointSetTopologyContainer.
    */
    PointSetTopologyContainer *getPointSetTopologyContainer() const
    {
        return static_cast<PointSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the PointSetTopologyModifier object of this PointSetTopology.
    */
    PointSetTopologyModifier<DataTypes> *getPointSetTopologyModifier() const
    {
        return static_cast<PointSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the PointSetTopologyAlgorithms object of this PointSetTopology.
    */
    PointSetTopologyAlgorithms<DataTypes> *getPointSetTopologyAlgorithms() const
    {
        return static_cast<PointSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Returns the PointSetGeometryAlgorithms object of this PointSetTopology.
    */
    PointSetGeometryAlgorithms<DataTypes> *getPointSetGeometryAlgorithms() const
    {
        return static_cast<PointSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
    }

    /** \brief Called by a topology to warn specific topologies linked to it that TopologyChange objects happened.
    *
    * Member m_changeList should contain all TopologyChange objects corresponding to changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @see BaseTopology::m_changeList
    * @sa firstChange()
    * @sa lastChange()
    */
    virtual void propagateTopologicalChanges();

    /** \brief Called by a topology to warn the Mechanical Object component that points have been added or will be removed.
    *
    * Member m_StateChangeList should contain all TopologyChange objects corresponding to vertex changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @see BaseTopology::m_changeList
    * @sa firstChange()
    * @sa lastChange()
    */
    virtual void propagateStateChanges();

    /** return the latest revision number */
    virtual int getRevision() const { return revisionCounter; }

    /** \brief Returns the object where the mechanical DOFs are stored */
    component::MechanicalObject<DataTypes> *getDOF() const { return object;	}

    /** \brief Build a topology from a file : call the load member function in the modifier object
    *
    */
    virtual bool load(const char *filename);

    /** \brief Translates the DOF : call the applyTranslation member function in the modifier object
    *
    */
    virtual void applyTranslation (const double dx,const double dy,const double dz);

    /** \brief Scales the DOF : call the applyScale member function in the modifier object
    *
    */
    virtual void applyScale (const double s);

    /** \brief Return the number of DOF in the mechanicalObject this Topology deals with.
    *
    */
    virtual unsigned int getDOFNumber() const { return object->getSize(); }


    /// BaseMeshTopology API
    /// @{
    virtual void clear()                       { }
    virtual const SeqEdges& getEdges()         { static SeqEdges     empty; return empty; }
    virtual const SeqTriangles& getTriangles() { static SeqTriangles empty; return empty; }
    virtual const SeqQuads& getQuads()         { static SeqQuads     empty; return empty; }
    virtual const SeqTetras& getTetras()       { static SeqTetras    empty; return empty; }
    virtual const SeqHexas& getHexas()         { static SeqHexas     empty; return empty; }
    /// @}

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalObject.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<MechanicalObject<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return core::componentmodel::topology::BaseTopology::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    ///
    /// Get the MechanicalObject.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        obj = new T(
            (context?dynamic_cast<MechanicalObject<DataTypes>*>(context->getMechanicalState()):NULL));
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const PointSetTopology<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

public:
    /** the object where the mechanical DOFs are stored */
    component::MechanicalObject<DataTypes> *object;					// TODO: clarify, should not this be in the container?

    // TODO: clarify, do these members have to be public?
    DataPtr< PointSetTopologyContainer > *f_m_topologyContainer;	// TODO: clarify, what is this needed for

protected:
    virtual void createComponents();

private:
    int revisionCounter;
};

/** A class that performs complex algorithms on a PointSet.
*
*/
template<class DataTypes>
class PointSetTopologyAlgorithms : public core::componentmodel::topology::TopologyAlgorithms
{
    // no methods implemented yet
public:
    PointSetTopologyAlgorithms(core::componentmodel::topology::BaseTopology *top)
        : TopologyAlgorithms(top)
    {}

    virtual ~PointSetTopologyAlgorithms() {}

    PointSetTopology<DataTypes>* getPointSetTopology() const
    {
        return static_cast<PointSetTopology<DataTypes>*> (this->m_basicTopology);
    }

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(sofa::helper::vector< unsigned int >& /*items*/)
    { }

    /** \brief Generic method to write the current mesh into a msh file
    */
    virtual void writeMSH(const char * /*filename*/)
    { }

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> &/*index*/,
            const sofa::helper::vector<unsigned int> &/*inv_index*/)
    { }
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

    PointSetGeometryAlgorithms(core::componentmodel::topology::BaseTopology *top)
        : GeometryAlgorithms(top)
    {}

    virtual ~PointSetGeometryAlgorithms() {}

    PointSetTopology<DataTypes>* getPointSetTopology() const
    {
        return static_cast<PointSetTopology<DataTypes>*> (this->m_basicTopology);
    }

    /** return the centroid of the set of points */
    Coord getPointSetCenter() const;

    /** return the centre and a radius of a sphere enclosing the  set of points (may not be the smalled one) */
    void getEnclosingSphere(Coord &center, Real &radius) const;

    /** return the axis aligned bounding box : index 0 = xmin, index 1=ymin,
    index 2 = zmin, index 3 = xmax, index 4 = ymax, index 5=zmax */
    void getAABB(Real bb[6]) const;
};


/**
* A class that can apply basic topology transformations on a set of points.
*/
template<class DataTypes>
class PointSetTopologyModifier : public core::componentmodel::topology::TopologyModifier
{

public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    PointSetTopologyModifier(core::componentmodel::topology::BaseTopology *top)
        : TopologyModifier(top)
    {}

    virtual ~PointSetTopologyModifier() {}

    PointSetTopology<DataTypes>* getPointSetTopology() const
    {
        return static_cast<PointSetTopology<DataTypes>*> (this->m_basicTopology);
    }

    /** \brief Build a point set topology from a file : also modifies the MechanicalObject
    *
    */
    virtual bool load(const char *filename);

    /** \brief Swap points i1 and i2.
    *
    */
    virtual void swapPoints(const int i1,const int i2);

    /** \brief Translates the DOF : call the applyTranslation member function in the MechanicalObject
    *
    */
    virtual void applyTranslation (const double dx,const double dy,const double dz);

    /** \brief Scales the DOF : call the applyScale member function in the MechanicalObject object
    *
    */
    virtual void applyScale (const double s);

    /** \brief Sends a message to warn that some points were added in this topology.
    *
    * \sa addPointsProcess
    */
    void addPointsWarning(const unsigned int nPoints, const bool addDOF = true);

    /** \brief Sends a message to warn that some points were added in this topology.
    *
    * \sa addPointsProcess
    */
    void addPointsWarning(const unsigned int nPoints,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double       > >& coefs,
            const bool addDOF = true);


    /** \brief Add some points to this topology.
    *
    * Use a list of ancestors to create the new points.
    * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
    * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
    * for the point being created.
    *
    * @param addDOF if true the points are actually added from the mechanical object's state vectors
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints, const bool addDOF = true);

    /** \brief Add some points to this topology.
    *
    * Use a list of ancestors to create the new points.
    * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
    * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
    * for the point being created.
    *
    * @param addDOF if true the points are actually added from the mechanical object's state vectors
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
    virtual void addNewPoint(unsigned int i,  const sofa::helper::vector< double >& x);

    /** \brief Sends a message to warn that some points are about to be deleted.
    *
    * \sa removePointsProcess
    */
    // side effect: indices are sorted first
    void removePointsWarning(/*const*/ sofa::helper::vector<unsigned int> &indices,
            const bool removeDOF = true);


    /** \brief Remove a subset of points
    *
    * Elements corresponding to these points are removed from the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    *
    * @param indices is not const because it is actually sorted from the highest index to the lowest one.
    * @param removeDOF if true the points are actually deleted from the mechanical object's state vectors
    */
    virtual void removePointsProcess(const sofa::helper::vector<unsigned int> &indices,
            const bool removeDOF = true);


    /** \brief Sends a message to warn that points are about to be reordered.
    *
    * \sa renumberPointsProcess
    */
    void renumberPointsWarning( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &inv_index,
            const bool renumberDOF = true);

    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &/*inv_index*/,
            const bool renumberDOF = true);

protected:
    /// modifies the mechanical object and creates the point set container
    void loadPointSet(PointSetTopologyLoader<DataTypes> *);
};

/** The container class that stores a set of points and provides access
to each point. This set of point may be a subset of the DOF of the mechanical model */
class PointSetTopologyContainer : public core::componentmodel::topology::TopologyContainer
{
    template <typename DataTypes>
    friend class PointSetTopologyModifier;

public:
    /** \brief Constructor from a a Base Topology.
    */
    PointSetTopologyContainer(core::componentmodel::topology::BaseTopology *top=NULL);

    virtual ~PointSetTopologyContainer() {}

    template <typename DataTypes>
    PointSetTopology<DataTypes>* getPointSetTopology() const
    {
        return static_cast<PointSetTopology<DataTypes>*> (this->m_basicTopology);
    }

    /** \brief Returns the number of vertices in this topology.
    *
    */
    unsigned int getNumberOfVertices() const;

    /** \brief Checks if the Topology is coherent
    *
    */
    virtual bool checkTopology() const;

    inline friend std::ostream& operator<< (std::ostream& out, const PointSetTopologyContainer& /*t*/)
    {
        return out;
    }

    /// Needed to be compliant with Datas.
    inline friend std::istream& operator>>(std::istream& in, PointSetTopologyContainer& /*t*/)
    {
        return in;
    }
};

/////////////////////////////////////////////////////////
/// TopologyChange subclasses
/////////////////////////////////////////////////////////


/** indicates that the indices of two points are being swapped */
class PointsIndicesSwap : public core::componentmodel::topology::TopologyChange
{
public:
    PointsIndicesSwap(const unsigned int i1,const unsigned int i2)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSINDICESSWAP)
    {
        index[0]=i1;
        index[1]=i2;
    }

public:
    unsigned int index[2];
};

/** indicates that some points were added */
class PointsAdded : public core::componentmodel::topology::TopologyChange
{
public:

    PointsAdded(const unsigned int nV)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSADDED)
        , nVertices(nV)
    { }

    PointsAdded(const unsigned int nV,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double       > >& baryCoefs)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSADDED)
        , nVertices(nV), ancestorsList(ancestors), coefs(baryCoefs)
    { }

    unsigned int getNbAddedVertices() const {return nVertices;}

public:
    unsigned int nVertices;
    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;
    sofa::helper::vector< sofa::helper::vector< double       > > coefs;
};

/** indicates that some points are about to be removed */
class PointsRemoved : public core::componentmodel::topology::TopologyChange
{
public:
    PointsRemoved(const sofa::helper::vector<unsigned int>& _vArray)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSREMOVED),
          removedVertexArray(_vArray)
    { }

    const sofa::helper::vector<unsigned int> &getArray() const { return removedVertexArray;	}

public:
    sofa::helper::vector<unsigned int> removedVertexArray;
};


/** indicates that the indices of all points have been renumbered */
class PointsRenumbering : public core::componentmodel::topology::TopologyChange
{
public:

    PointsRenumbering()
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSRENUMBERING)
    { }

    PointsRenumbering(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< unsigned int >& inv_indices)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSRENUMBERING),
          indexArray(indices), inv_indexArray(inv_indices)
    { }

    const sofa::helper::vector<unsigned int> &getIndexArray() const { return indexArray; }

    const sofa::helper::vector<unsigned int> &getinv_IndexArray() const { return inv_indexArray; }

public:
    sofa::helper::vector<unsigned int> indexArray;
    sofa::helper::vector<unsigned int> inv_indexArray;
};


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGY_H
