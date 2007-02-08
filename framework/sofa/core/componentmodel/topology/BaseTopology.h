#ifndef SOFA_CORE_COMPONENTMODEL_TOPOLOGY_BASETOPOLOGY_H
#define SOFA_CORE_COMPONENTMODEL_TOPOLOGY_BASETOPOLOGY_H

#include <stdlib.h>
#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace topology
{

// forward definitions :
/// Contains the actual topology data and give acces to it.
class TopologyContainer;
/// Provides low-level topology methods (e.g. AddPoint, RemoveEdge, etc).
class TopologyModifier;
/// Provides high-level topology algorithms (e.g. CutAlongPlane, DecimateTopology, etc).
class TopologyAlgorithms;
/// Provides some geometric functions (e.g. ComputeTriangleNormal, ComputeShell, etc).
class GeometryAlgorithms;
/// Translates topology events (TopologyChange objects) from a topology so that they apply on another one.
class TopologicalMapping;


/// The enumeration used to give unique identifiers to TopologyChange objects.
enum TopologyChangeType
{
    BASE,               ///< For TopologyChange class, should never be used.
    POINTSINDICESSWAP,  ///< For PointsIndicesSwap class.
    POINTSADDED,        ///< For PointsAdded class.
    POINTSREMOVED,      ///< For PointsRemoved class.
    POINTSRENUMBERING,  ///< For PointsRenumbering class.
    EDGESADDED,         ///< For EdgesAdded class.
    EDGESREMOVED,       ///< For EdgesRemoved class.
    EDGESRENUMBERING    ///< For EdgesRenumbering class.

};


/** \brief Base class to indicate a topology change occurred.
 *
 * All topological changes taking place in a given BaseTopology will issue a TopologyChange in the
 * BaseTopology's changeList, so that BasicTopologies mapped to it can know what happened and decide how to
 * react.
 * Classes inheriting from this one describe a given topolopy change (e.g. RemovedPoint, AddedEdge, etc).
 * The exact type of topology change is given by member changeType.
 */
class TopologyChange
{

protected:
    TopologyChangeType m_changeType; ///< A code that tells the nature of the Topology modification event (could be an enum).

    TopologyChange( TopologyChangeType changeType = BASE ):m_changeType(changeType)
    {
    }

public:
    /** \brief Returns the code of this TopologyChange. */
    TopologyChangeType getChangeType() const
    {
        return m_changeType;
    }

    /** \ brief Destructor.
     *
     * Must be virtual for TopologyChange to be a Polymorphic type.
     */
    virtual ~TopologyChange() { };
};



/** \brief Base class that gives access to the 4 topology related objects and an array of topology modifications.
 *
 * This class containss everything that is needed to manage a particular topology. That is :
 * - a TopologyContainer object to hold actual data like DOF indices or neighborhood information,
 * - a TopologyModifier object providing simple low-level methods on these data (like RemoveEdge,
 * DuplicatePoint, etc),
 * - a TopologyAlgorithms object providing high-level methods (like CutAlongPlane or RefineLocally) which
 * mostly rely on calls to methods of TopologyModifier,
 * - a GeometryAlgorithms object providing geometric functions (like ComputeTriangleNormals).
 *
 * The class also holds an array of TopologyChange objects needed by Topologies linked to this one to know
 * what happened and how to take it into account (or ignore it).
 */
class BaseTopology : public objectmodel::BaseObject
{

public :
    /** \brief Provides an iterator on the first element in the list of TopologyChange objects.
     */
    std::list<const TopologyChange *>::const_iterator firstChange() const;



    /** \brief Provides an iterator on the last element in the list of TopologyChange objects.
     */
    std::list<const TopologyChange *>::const_iterator lastChange() const;



    /** \brief Returns the TopologyContainer object of this Topology.
     */
    TopologyContainer *getTopologyContainer() const
    {
        return m_topologyContainer;
    }



    /** \brief Returns the TopologyModifier object of this Topology.
     */
    TopologyModifier *getTopologyModifier() const
    {
        return m_topologyModifier;
    }



    /** \brief Returns the TopologyAlgorithms object of this Topology if it is a main topology, 0 otherwise.
     *
     * Specific topologies cannot be allowed to be directly modified, since this might invalidate their
     * mapping from the main topology.
     */
    TopologyAlgorithms *getTopologyAlgorithms() const
    {
        if (m_mainTopology)
            return m_topologyAlgorithms;
        else
            return (TopologyAlgorithms *) 0;
    }



    /** \brief Returns the GeometryAlgorithms object of this Topology.
     */
    GeometryAlgorithms *getGeometryAlgorithms() const
    {
        return m_geometryAlgorithms;
    }



    /** \brief Constructor.
     *
     * Optionnial parameter isMainTopology may be used to specify wether this is a main or a specific topology
     * (defaults to true).
     *
     * All topology related objects (TopologyContainer, TopologyModifier, TopologyAlgorithms,
     * GeometryAlgorithms) are initialized to 0.
     */
    BaseTopology(bool isMainTopology = true)
        : m_topologyContainer(0),
          m_topologyModifier(0),
          m_topologyAlgorithms(0),
          m_geometryAlgorithms(0),
          m_mainTopology(isMainTopology)
    {
    }



    /// Destructor.
    ~BaseTopology();



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
    virtual void propagateTopologicalChanges()
    {
    }
    /** \brief Returns whether this topology is a main one or a specific one.
     *
     * @see BaseTopology::m_mainTopology
     */
    bool isMainTopology() const
    {
        return m_mainTopology;
    }

    /** \brief Return the number of DOF in the mechanicalObject this Topology deals with.
     *
     */
    virtual unsigned int getDOFNumber() const { return 0; }

protected :
    /// Contains the actual topology data and give acces to it (nature of these data heavily depends on the kind of topology).
    TopologyContainer *m_topologyContainer;

    /// Provides low-level topology methods (e.g. AddPoint, RemoveEdge, etc).
    TopologyModifier *m_topologyModifier;

    /// Provides high-level topology algorithms (e.g. CutAlongPlane, DecimateTopology, etc).
    TopologyAlgorithms *m_topologyAlgorithms;

    /// Provides some geometric functions (e.g. ComputeTriangleNormal, ComputeShell, etc).
    GeometryAlgorithms *m_geometryAlgorithms;

    /** \brief Defines wehther this topology is the main one for its mechanical object.
     *
     * If true, then this topology is the main topology of the MechanicalObject, meaning this is the one
     * obtained by default when asking the MechanicalObject for its topology.
     * Otherwise this topology is a specific one, relying on the main one of the MechanicalObject through a
     * TopologicalMapping. For example, a specific topology might be a subset of the main topology of the
     * MechanicalObject, on which a Constraint or a ForceField applies.
     */
    bool m_mainTopology;

protected:
    /** \brief Free each Topology changes in the list and remove them from the list
      *
      */
    void resetTopologyChangeList() const;

};



/** A class that contains a description of the topology (set of edges, triangles, adjacency information, ...) */
class TopologyContainer
{

public:
    /** \brief Constructor.
     *
     * @param basicTopology the topology this object describes.
     */
    TopologyContainer(BaseTopology *basicTopology) : m_basicTopology(basicTopology)
    {
    }
    /// Destructor
    virtual ~TopologyContainer()
    {
    }

    const std::list<const TopologyChange *> &getChangeList() const
    {
        return m_changeList;
    }
protected:
    /// The topology this object describes.
    BaseTopology *m_basicTopology;

    /// Array of topology modifications that have already occured (addition) or will occur next (deletion).
    std::list<const TopologyChange *> m_changeList; // shouldn't this be private?


    /** \brief Adds a TopologyChange to the list.
    *
    * Needed by topologies linked to this one to know what happened and what to do to take it into account.
    *
    * Only TopologyModifier and TopologyAlgorithms objects of this topology should have access to this method.
    *
    * Question : Is this wrapper really needed since member m_changeTopology is protected and TopologyModifier
    * and TopologyAlgorithms are friend classes?
    */
    void addTopologyChange(const TopologyChange *topologyChange)
    {
        m_changeList.push_back(topologyChange);
    }

    /** \brief Free each Topology changes in the list and remove them from the list
     *
     */
    void resetTopologyChangeList();

    // Friend classes declaration.
    // Needed so that TopologyModifier and TopologyAlgorithms and BaseTopology can access private
    // method addTopologyChange and resetTopologyList
    friend class TopologyModifier;
    friend class TopologyAlgorithms;
    friend class BaseTopology;
};



/** A class that contains a set of low-level methods that perform topological changes */
class TopologyModifier
{

public:
    /** \brief Constructor.
     *
     * @param basicTopology the topology this object applies to.
     */
    TopologyModifier(BaseTopology *basicTopology) : m_basicTopology(basicTopology)
    {
    }

    /// Destructor
    virtual ~TopologyModifier()
    {
    }

protected:
    /// The topology this object applies to.
    BaseTopology *m_basicTopology;



    /** \brief Adds a TopologyChange object to the list of the topology this object describes.
     */
    void addTopologyChange(const TopologyChange *topologyChange)
    {
        m_basicTopology->getTopologyContainer()->addTopologyChange(topologyChange);
    }

};



/** A class that contains a set of high-level (user friendly) methods that perform topological changes */
class TopologyAlgorithms
{

public:
    /** \brief Constructor.
     *
     * @param basicTopology the topology this object applies to.
     */
    TopologyAlgorithms(BaseTopology *basicTopology) : m_basicTopology(basicTopology)
    {
    }
    /// Destructor
    virtual ~TopologyAlgorithms()
    {
    }
protected:
    /// The topology this object applies to.
    BaseTopology *m_basicTopology;



    /** \brief Adds a TopologyChange object to the list of the topology this object describes.
     */
    void addTopologyChange(const TopologyChange *topologyChange)
    {
        m_basicTopology->getTopologyContainer()->addTopologyChange(topologyChange);
    }
};



/** A class that contains a set of methods that describes the geometry of the object */
class GeometryAlgorithms
{

public:
    /** \brief Constructor.
     *
     * @param basicTopology the topology this object applies to.
     */
    GeometryAlgorithms(BaseTopology *basicTopology) : m_basicTopology(basicTopology)
    {
    }
    /// Destructor
    virtual ~GeometryAlgorithms()
    {
    }
protected:
    /// The topology this object applies to.
    BaseTopology *m_basicTopology;

};



} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BASICTOPOLOGY_H
