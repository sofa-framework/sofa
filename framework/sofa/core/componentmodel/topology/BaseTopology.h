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

#include <stdlib.h>
#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <sofa/core/objectmodel/BaseObject.h>

//#include <sofa/core/componentmodel/topology/Topology.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>

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
class BaseTopology : public virtual core::objectmodel::BaseObject, public core::componentmodel::topology::BaseMeshTopology
{

public :

    /** \brief Provides an iterator on the first element in the list of TopologyChange objects.
     */
    virtual std::list<const TopologyChange *>::const_iterator firstChange() const;

    /** \brief Provides an iterator on the last element in the list of TopologyChange objects.
     */
    virtual std::list<const TopologyChange *>::const_iterator lastChange() const;

    /** \brief Provides an iterator on the first element in the list of StateChange objects.
     */
    std::list<const TopologyChange *>::const_iterator firstStateChange() const;


    /** \brief Provides an iterator on the last element in the list of StateChange objects.
     */
    std::list<const TopologyChange *>::const_iterator lastStateChange() const;



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
    virtual TopologyAlgorithms *getTopologyAlgorithms() const
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
          m_mainTopology(isMainTopology),
          filename(initData(&filename,"filename","Filename of the object"))
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
    virtual void propagateStateChanges()
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

    /** \brief Free each Topology changes in the list and remove them from the list
     *
     */
    void resetTopologyChangeList() const;

    virtual std::string getFilename() const {return filename.getValue();}

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

    /** \brief Free each State changes in the list and remove them from the list
     *
     */
    void resetStateChangeList() const;

    Data< std::string > filename;

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

    const std::list<const TopologyChange *> &getStateChangeList() const
    {
        return m_StateChangeList;
    }
    /*
            inline friend std::ostream& operator<< (std::ostream& out, const TopologyContainer& )
            {
    	  return out;
            }

            /// Needed to be compliant with Datas.
            inline friend std::istream& operator>>(std::istream& in, TopologyContainer& )
            {
    	  return in;
      	}*/
    void setTopology(BaseTopology *b) { m_basicTopology = b;}
protected:
    /// The topology this object describes.
    BaseTopology *m_basicTopology;

    /// Array of topology modifications that have already occured (addition) or will occur next (deletion).
    std::list<const TopologyChange *> m_changeList; // shouldn't this be private?

    /// Array of state modifications that have already occured (addition) or will occur next (deletion).
    std::list<const TopologyChange *> m_StateChangeList; // shouldn't this be private?


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

    /** \brief Adds a StateChange to the list.
     *
     * Needed by topologies linked to this one to know what happened and what to do to take it into account.
     *
     * Only TopologyModifier and TopologyAlgorithms objects of this topology should have access to this method.
     *
     * Question : Is this wrapper really needed since member m_StateChangeList is protected and TopologyModifier
     * and TopologyAlgorithms are friend classes?
     */
    void addStateChange(const TopologyChange *topologyChange)
    {
        m_StateChangeList.push_back(topologyChange);
    }

    /** \brief Free each Topology changes in the list and remove them from the list
     *
     */
    void resetTopologyChangeList();

    /** \brief Free each State changes in the list and remove them from the list
     *
     */
    void resetStateChangeList();

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

    friend class TopologyMapping;

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

    /** \brief Adds a StateChange object to the list of the topology this object describes.
     */
    void addStateChange(const TopologyChange *topologyChange)
    {
        m_basicTopology->getTopologyContainer()->addStateChange(topologyChange);
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

public:
    /** \notify the end for the current sequence of topological change events.
     */
    void notifyEndingEvent()
    {
        EndingEvent *e=new EndingEvent();
        addTopologyChange(e);
    }

    /** \brief Generic method to remove a list of items.
     */
    virtual void removeItems(sofa::helper::vector< unsigned int >& /*items*/)
    {
    }

    /** \brief Generic method to write the current mesh into a msh file
     */
    virtual void writeMSH(const char * /*filename*/) {return;}

    /** \brief Generic method for points renumbering
     */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> &/*index*/, const sofa::helper::vector<unsigned int> &/*inv_index*/)
    {
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
