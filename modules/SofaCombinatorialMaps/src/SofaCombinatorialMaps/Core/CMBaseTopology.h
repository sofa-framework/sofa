/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFACOMBINATORIALMAPS_CORE_CMBASETOPOLOGY_H_
#define SOFACOMBINATORIALMAPS_CORE_CMBASETOPOLOGY_H_
#include <SofaCombinatorialMaps/config.h>

#include <sofa/core/topology/BaseTopologyObject.h>
#include <SofaCombinatorialMaps/Core/CMapTopology.h>
//#include <SofaCombinatorialMaps/Core/CMTopologyChange.h>
#include <sofa/core/VecId.h>

#include <sofa/helper/list.h>
#include <sofa/core/objectmodel/BaseNode.h>


namespace sofa
{

namespace core
{

namespace cm_topology
{
using core::topology::BaseMeshTopology;

// forward declarations:

/// Provides high-level topology algorithms (e.g. CutAlongPlane, DecimateTopology, etc).
class TopologyAlgorithms;

/// Provides some geometric functions (e.g. ComputeTriangleNormal, ComputeShell, etc).
class GeometryAlgorithms;

/// Provides low-level topology methods (e.g. AddPoint, RemoveEdge, etc).
class TopologyModifier;

/// Contains the actual topology data and give acces to it.
//class TopologyContainer;

/// Translates topology events (TopologyChange objects) from a topology so that they apply on another one.
class TopologicalMapping;

/// Allow topological handle events
class TopologyEngine;

struct PointAncestorElem;

/** A class that contains a set of high-level (user frisendly) methods that perform topological changes */
class SOFA_COMBINATORIALMAPS_API TopologyAlgorithms : public sofa::core::topology::BaseTopologyObject
{
public:
    SOFA_CLASS(TopologyAlgorithms, BaseTopologyObject);

protected:
    /** \brief Constructor.
    *
    */
    TopologyAlgorithms()
    {}


    /// Destructor
    virtual ~TopologyAlgorithms()
    {}
public:
	virtual void init() override;

protected:
    /** \brief Adds a TopologyChange object to the list of the topology this object describes.
    */
    void addTopologyChange(const TopologyChange *topologyChange);

protected:
    /// Contains the actual topology data and give acces to it (nature of these data heavily depends on the kind of topology).
	topology::CMapTopology *m_topology;
};

/** A class that contains a set of methods that describes the geometry of the object */
class SOFA_COMBINATORIALMAPS_API GeometryAlgorithms : public sofa::core::topology::BaseTopologyObject
{
public:
    SOFA_CLASS(GeometryAlgorithms, BaseTopologyObject);

protected:
    /** \brief Constructor.
    *
    */
    GeometryAlgorithms()
    {}


    /// Destructor
    virtual ~GeometryAlgorithms()
    {}
public:
	virtual void init() override;

    /** \brief Called by the MechanicalObject state change callback to initialize added
    * points according to the topology (topology element & local coordinates)
    *
    * \param ancestorElems are the ancestors topology info used in the points modifications
    */
    virtual void initPointsAdded(const helper::vector< unsigned int > &indices, const helper::vector< PointAncestorElem > &ancestorElems
        , const helper::vector< core::VecCoordId >& coordVecs, const helper::vector< core::VecDerivId >& derivVecs );
};

/** A class that contains a set of low-level methods that perform topological changes */
class SOFA_COMBINATORIALMAPS_API TopologyModifier : public sofa::core::topology::BaseTopologyObject
{
public:
    SOFA_CLASS(TopologyModifier, BaseTopologyObject);

protected:
    /** \brief Constructor.
    *
    */
    TopologyModifier()
        : m_topology(NULL)
    { }


    /// Destructor
    virtual ~TopologyModifier()
    { }
public:
	virtual void init() override;

    /** \brief Called by a topology to warn the Mechanical Object component that points have been added or will be removed.
    *
    * StateChangeList should contain all TopologyChange objects corresponding to vertex changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @sa firstChange()
    * @sa lastChange()
    */
    virtual void propagateStateChanges();

    /** \brief Called by a topology to warn specific topologies linked to it that TopologyChange objects happened.
    *
    * ChangeList should contain all TopologyChange objects corresponding to changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @sa firstChange()
    * @sa lastChange()
    */
    virtual void propagateTopologicalChanges();

    /** \brief notify the end for the current sequence of topological change events.
    */
    virtual void notifyEndingEvent();

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(const sofa::helper::vector<unsigned int> & /*items*/);

protected:
    /** \brief Adds a TopologyChange object to the list of the topology this object describes.
    */
    void addTopologyChange(const TopologyChange *topologyChange);

    /** \brief Adds a StateChange object to the list of the topology this object describes.
    */
    void addStateChange(const TopologyChange *topologyChange);

protected:
    /// Contains the actual topology data and give acces to it (nature of these data heavily depends on the kind of topology).
	topology::CMapTopology *m_topology;
};

} // namespace cm_topology

} // namespace core

} // namespace sofa

#endif // SOFACOMBINATORIALMAPS_CORE_CMBASETOPOLOGY_H_
