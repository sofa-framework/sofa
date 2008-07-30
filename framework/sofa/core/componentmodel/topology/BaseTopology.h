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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_TOPOLOGY_BASETOPOLOGY_H
#define SOFA_CORE_COMPONENTMODEL_TOPOLOGY_BASETOPOLOGY_H

#include <list>
#include <string>

#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/core/componentmodel/topology/BaseTopologyObject.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace topology
{
using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::PointID PointID;

// forward declarations:

class BaseTopology;

/// Provides high-level topology algorithms (e.g. CutAlongPlane, DecimateTopology, etc).
class TopologyAlgorithms;

/// Provides some geometric functions (e.g. ComputeTriangleNormal, ComputeShell, etc).
class GeometryAlgorithms;

/// Provides low-level topology methods (e.g. AddPoint, RemoveEdge, etc).
class TopologyModifier;

/// Contains the actual topology data and give acces to it.
class TopologyContainer;

/// Translates topology events (TopologyChange objects) from a topology so that they apply on another one.
class TopologicalMapping;

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
class BaseTopology :  public core::componentmodel::topology::BaseMeshTopology
{
public :
    /** \brief Constructor.
    *
    * Optional parameter isMainTopology may be used to specify whether this is a main or a specific topology
    * (defaults to true).
    *
    * All topology related objects (TopologyContainer, TopologyModifier, TopologyAlgorithms,
    * GeometryAlgorithms) are initialized to 0.
    */
    BaseTopology(bool isMainTopology = true);

    /// Destructor.
    virtual ~BaseTopology();

    /** \brief Returns the TopologyContainer object of this Topology.
    */
    TopologyContainer *getTopologyContainer() const { return m_topologyContainer;}


    /** \brief Returns the TopologyModifier object of this Topology.
    */
    TopologyModifier *getTopologyModifier() const {	return m_topologyModifier;	}

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
            return NULL;
    }

    /** \brief Returns the GeometryAlgorithms object of this Topology.
    */
    GeometryAlgorithms *getGeometryAlgorithms() const {	return m_geometryAlgorithms; }


    // TODO: remove these methods (the implementation has been moved into container)
    std::list<const TopologyChange *>::const_iterator firstChange() const;
    std::list<const TopologyChange *>::const_iterator lastChange() const;
    std::list<const TopologyChange *>::const_iterator firstStateChange() const;
    std::list<const TopologyChange *>::const_iterator lastStateChange() const;
    void propagateTopologicalChanges();
    void propagateStateChanges();
    void resetTopologyChangeList() const;
    void resetStateChangeList() const;

    /** \brief Returns whether this topology is a main one or a specific one.
    *
    * @see BaseTopology::m_mainTopology
    */
    bool isMainTopology() const { return m_mainTopology; }

    /** return the latest revision number */
    int getRevision() const { return revisionCounter; }

protected :
    /// Contains the actual topology data and give acces to it (nature of these data heavily depends on the kind of topology).
    TopologyContainer *m_topologyContainer;

    /// Provides low-level topology methods (e.g. AddPoint, RemoveEdge, etc).
    TopologyModifier *m_topologyModifier;

    /// Provides high-level topology algorithms (e.g. CutAlongPlane, DecimateTopology, etc).
    TopologyAlgorithms *m_topologyAlgorithms;

    /// Provides some geometric functions (e.g. ComputeTriangleNormal, ComputeShell, etc).
    GeometryAlgorithms *m_geometryAlgorithms;

private:
    /** \brief Defines whether this topology is the main one for its mechanical object.
    *
    * If true, then this topology is the main topology of the MechanicalObject, meaning this is the one
    * obtained by default when asking the MechanicalObject for its topology.
    * Otherwise this topology is a specific one, relying on the main one of the MechanicalObject through a
    * TopologicalMapping. For example, a specific topology might be a subset of the main topology of the
    * MechanicalObject, on which a Constraint or a ForceField applies.
    */
    bool m_mainTopology;

    int revisionCounter;
};


/** A class that contains a set of high-level (user friendly) methods that perform topological changes */
class TopologyAlgorithms : public virtual sofa::core::componentmodel::topology::BaseTopologyObject
{
public:
    /** \brief Constructor.
    *
    * @param basicTopology the topology this object applies to.
    */
    TopologyAlgorithms(BaseTopology *basicTopology = NULL)
        : m_basicTopology(basicTopology)
    {}

    /// Destructor
    virtual ~TopologyAlgorithms()
    {}

    /** \notify the end for the current sequence of topological change events.
    */
    void notifyEndingEvent()
    {
        EndingEvent *e=new EndingEvent();
        addTopologyChange(e);
    }

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(sofa::helper::vector< unsigned int >& /*items*/) {	}

    /** \brief Generic method to write the current mesh into a msh file
    */
    virtual void writeMSH(const char * /*filename*/) {return;}

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> &/*index*/, const sofa::helper::vector<unsigned int> &/*inv_index*/) { }

protected:
    /** \brief Adds a TopologyChange object to the list of the topology this object describes.
    */
    void addTopologyChange(const TopologyChange *topologyChange);

protected:

    /// The topology this object applies to.
    BaseTopology *m_basicTopology;
};

/** A class that contains a set of methods that describes the geometry of the object */
class GeometryAlgorithms : public virtual sofa::core::componentmodel::topology::BaseTopologyObject
{
public:
    /** \brief Constructor.
    *
    * @param basicTopology the topology this object applies to.
    */
    GeometryAlgorithms(BaseTopology *basicTopology = NULL)
        : m_basicTopology(basicTopology)
    {}

    /// Destructor
    virtual ~GeometryAlgorithms()
    {}

protected:
    /// The topology this object applies to.
    BaseTopology *m_basicTopology;
};

/** A class that contains a set of low-level methods that perform topological changes */
class TopologyModifier : public virtual sofa::core::componentmodel::topology::BaseTopologyObject
{
public:
    /** \brief Constructor.
    *
    * @param basicTopology the topology this object applies to.
    */
    TopologyModifier(TopologyContainer *container=NULL)
        : m_topologyContainer(container)
    { }

    /// Destructor
    virtual ~TopologyModifier()
    { }

protected:
    /** \brief Adds a TopologyChange object to the list of the topology this object describes.
    */
    void addTopologyChange(const TopologyChange *topologyChange);

    /** \brief Adds a StateChange object to the list of the topology this object describes.
    */
    void addStateChange(const TopologyChange *topologyChange);

protected:
    /// Contains the actual topology data and give acces to it (nature of these data heavily depends on the kind of topology).
    TopologyContainer *m_topologyContainer;
};

/** A class that contains a description of the topology (set of edges, triangles, adjacency information, ...) */
class TopologyContainer : public virtual sofa::core::componentmodel::topology::BaseTopologyObject
{
public:
    /** \brief Constructor.
    *
    * @param basicTopology the topology this object describes.
    */
    TopologyContainer()
    {}

    /// Destructor
    virtual ~TopologyContainer()
    {}

    const std::list<const TopologyChange *> &getChangeList() const { return m_changeList; }

    const std::list<const TopologyChange *> &getStateChangeList() const { return m_stateChangeList; }

    /** \brief Adds a TopologyChange to the list.
    *
    * Needed by topologies linked to this one to know what happened and what to do to take it into account.
    *
    */
    void addTopologyChange(const TopologyChange *topologyChange)
    {
        m_changeList.push_back(topologyChange);
    }

    /** \brief Adds a StateChange to the list.
    *
    * Needed by topologies linked to this one to know what happened and what to do to take it into account.
    *
    */
    void addStateChange(const TopologyChange *topologyChange)
    {
        m_stateChangeList.push_back(topologyChange);
    }

    /** \brief Provides an iterator on the first element in the list of TopologyChange objects.
     */
    std::list<const TopologyChange *>::const_iterator firstChange() const;

    /** \brief Provides an iterator on the last element in the list of TopologyChange objects.
     */
    std::list<const TopologyChange *>::const_iterator lastChange() const;

    /** \brief Provides an iterator on the first element in the list of StateChange objects.
     */
    std::list<const TopologyChange *>::const_iterator firstStateChange() const;

    /** \brief Provides an iterator on the last element in the list of StateChange objects.
     */
    std::list<const TopologyChange *>::const_iterator lastStateChange() const;

    /** \brief Called by a topology to warn specific topologies linked to it that TopologyChange objects happened.
    *
    * ChangeList should contain all TopologyChange objects corresponding to changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @sa firstChange()
    * @sa lastChange()
    */
    virtual void propagateTopologicalChanges() {}

    /** \brief Called by a topology to warn the Mechanical Object component that points have been added or will be removed.
    *
    * StateChangeList should contain all TopologyChange objects corresponding to vertex changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @sa firstChange()
    * @sa lastChange()
    */
    virtual void propagateStateChanges() {}

    /** \brief Free each Topology changes in the list and remove them from the list
    *
    */
    void resetTopologyChangeList();

    /** \brief Free each State changes in the list and remove them from the list
    *
    */
    void resetStateChangeList();

private:
    /// Array of topology modifications that have already occured (addition) or will occur next (deletion).
    std::list<const TopologyChange *> m_changeList;

    /// Array of state modifications that have already occured (addition) or will occur next (deletion).
    std::list<const TopologyChange *> m_stateChangeList;
};

} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BASICTOPOLOGY_H
