#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_H

//#include <stdlib.h>
#include <vector>
//#include <string>
#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/helper/fixed_array.h>

namespace sofa
{

namespace component
{

namespace topology
{

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::PointID PointID;

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

    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestorsList;

    sofa::helper::vector< sofa::helper::vector< double       > > coefs;

    PointsAdded(const unsigned int nV,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0,
            const sofa::helper::vector< sofa::helper::vector< double       > >& baryCoefs = (const sofa::helper::vector< sofa::helper::vector< double       > >)0)
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
    sofa::helper::vector<unsigned int> removedVertexArray;

public:
    PointsRemoved(const sofa::helper::vector<unsigned int>& _vArray) : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSREMOVED), removedVertexArray(_vArray)
    {
    }

    const sofa::helper::vector<unsigned int> &getArray() const
    {
        return removedVertexArray;
    }

};



/** indicates that the indices of all points have been renumbered */
class PointsRenumbering : public core::componentmodel::topology::TopologyChange
{

public:
    sofa::helper::vector<unsigned int> indexArray;

    PointsRenumbering(const sofa::helper::vector< unsigned int >& indices = (const sofa::helper::vector< unsigned int >)0)
        : core::componentmodel::topology::TopologyChange(core::componentmodel::topology::POINTSRENUMBERING), indexArray(indices)
    { }

    const sofa::helper::vector<unsigned int> &getIndexArray() const
    {
        return indexArray;
    }

};



/////////////////////////////////////////////////////////
/// PointSetTopology objects
/////////////////////////////////////////////////////////


/** The container class that stores a set of points and provides access
to each point. This set of point may be a subset of the DOF of the mechanical model */
class PointSetTopologyContainer : public core::componentmodel::topology::TopologyContainer
{

public:

    inline friend std::ostream& operator<< (std::ostream& out, const PointSetTopologyContainer& /*t*/)
    {
        return out;
    }

    /// Needed to be compliant with Datas.
    inline friend std::istream& operator>>(std::istream& in, PointSetTopologyContainer& /*t*/)
    {
        return in;
    }

    /** \brief Returns the number of vertices in this topology.
     *
     */
    unsigned int getNumberOfVertices() const;

    /** \brief Constructor from a a Base Topology.
     */
    PointSetTopologyContainer(core::componentmodel::topology::BaseTopology *top=NULL);


    /** \brief Checks if the Topology is coherent
     *
     */
    virtual bool checkTopology() const;

    template <typename DataTypes>
    friend class PointSetTopologyModifier;

};

// forward declaration
template< typename DataTypes > class PointSetTopologyLoader;

/**
 * A class that can apply basic topology transformations on a set of points.
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

    /*
    template< typename DataTypes >
      friend class PointSetTopologyAlgorithms;

    friend class sofa::core::componentmodel::topology::TopologicalMapping;

    template< typename In, typename Out >
    friend class Tetra2TriangleTopologicalMapping;
    */

    //protected:
    /** \brief Sends a message to warn that some points were added in this topology.
     *
     * \sa addPointsProcess
     */
    void addPointsWarning(const unsigned int nPoints,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = (const sofa::helper::vector< sofa::helper::vector< unsigned int > >) 0,
            const sofa::helper::vector< sofa::helper::vector< double       > >& coefs     = (const sofa::helper::vector< sofa::helper::vector< double       > >) 0);



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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs = (const sofa::helper::vector< sofa::helper::vector< double > >)0 );


    /** \brief Add a new point (who has no ancestors) to this topology.
     *
     * \sa addPointsWarning
     */
    virtual void addNewPoint( const sofa::helper::vector< double >& x);

    /** \brief Sends a message to warn that some points are about to be deleted.
     *
     * \sa removePointsProcess
     */
    void removePointsWarning(sofa::helper::vector<unsigned int> &indices);


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
    virtual void removePointsProcess( sofa::helper::vector<unsigned int> &indices, const bool removeDOF = true);



    /** \brief Reorder this topology.
     *
     * \see MechanicalObject::renumberValues
     */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index );

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

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(sofa::helper::vector< unsigned int >& /*items*/) {return;}

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
class PointSetTopology : public core::componentmodel::topology::BaseTopology, public core::componentmodel::topology::BaseMeshTopology
{

public:
    /** the object where the mechanical DOFs are stored */
    component::MechanicalObject<DataTypes> *object;

    DataPtr< PointSetTopologyContainer > *f_m_topologyContainer;
public:
    PointSetTopology(component::MechanicalObject<DataTypes> *obj);

    /** \brief Returns the PointSetTopologyAlgorithms object of this PointSetTopology.
     */
    PointSetTopologyAlgorithms<DataTypes> *getPointSetTopologyAlgorithms() const
    {
        return (PointSetTopologyAlgorithms<DataTypes> *)this->m_topologyAlgorithms;
    }

    /** \brief Generic method returning the TopologyAlgorithms object
     */
    virtual core::componentmodel::topology::TopologyAlgorithms *getTopologyAlgorithms() const
    {
        return getPointSetTopologyAlgorithms();
    }

    /** \brief Returns the object where the mechanical DOFs are stored */
    component::MechanicalObject<DataTypes> *getDOF() const
    {
        return object;
    }
    /** creates a TopologyChangeVisitor and therefore warns all components that
        some topological changes have occured */
    virtual void propagateTopologicalChanges();

    virtual void init();
    /** \brief Returns the PointSetTopologyContainer object of this PointSetTopologyContainer.
     */
    PointSetTopologyContainer *getPointSetTopologyContainer() const
    {
        return (PointSetTopologyContainer *)this->m_topologyContainer;
    }

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


    /** Parse the XML attributes : allows to load a topology from a file */
    void parse(core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("filename"))
            this->load(arg->getAttribute("filename"));
        if (arg->getAttribute("scale")!=NULL)
        {
            this->applyScale(atof(arg->getAttribute("scale")));
        }
        if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
        {
            this->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
        }
        this->core::componentmodel::topology::BaseTopology::parse(arg);
    }

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

protected:
    PointSetTopology(component::MechanicalObject<DataTypes> *obj,const PointSetTopology *);

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGY_H
