/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/BaseTopologyObject.h>
#include <sofa/core/VecId.h>
#include <array>

namespace sofa::core::topology
{
using core::topology::BaseMeshTopology;

// forward declarations:

/// Provides some geometric functions (e.g. ComputeTriangleNormal, ComputeShell, etc) and high-level topology algorithms (e.g. CutAlongPlane, DecimateTopology, etc).
class GeometryAlgorithms;

/// Provides low-level topology methods (e.g. AddPoint, RemoveEdge, etc).
class TopologyModifier;

/// Contains the actual topology data and give access to it.
class TopologyContainer;

/// Translates topology events (TopologyChange objects) from a topology so that they apply on another one.
class TopologicalMapping;

/// Allow topological handle events
class TopologyHandler;


/** A class that contains a set of methods that describes the geometry of the object */
class SOFA_CORE_API GeometryAlgorithms : public sofa::core::topology::BaseTopologyObject
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
    ~GeometryAlgorithms() override
    {}
public:
    void init() override;

    /** \brief Called by the MechanicalObject state change callback to initialize added
    * points according to the topology (topology element & local coordinates)
    *
    * \param ancestorElems are the ancestors topology info used in the points modifications
    */
    virtual void initPointsAdded(const type::vector< sofa::Index > &indices, const type::vector< PointAncestorElem > &ancestorElems
        , const type::vector< core::VecCoordId >& coordVecs, const type::vector< core::VecDerivId >& derivVecs );
};

/** A class that contains a set of low-level methods that perform topological changes */
class SOFA_CORE_API TopologyModifier : public sofa::core::topology::BaseTopologyObject
{
public:
    typedef sofa::Index Index;

    SOFA_CLASS(TopologyModifier, BaseTopologyObject);

protected:
    /** \brief Constructor.
    *
    */
    TopologyModifier()
        : m_topologyContainer(nullptr)
    { }


    /// Destructor
    ~TopologyModifier() override
    { }
public:
    void init() override;

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
    virtual void removeItems(const sofa::type::vector<Index> & /*items*/);

protected:
    /** \brief Adds a TopologyChange object to the list of the topology this object describes.
    */
    void addTopologyChange(const TopologyChange *topologyChange);

    /** \brief Adds a StateChange object to the list of the topology this object describes.
    */
    void addStateChange(const TopologyChange *topologyChange);

    /// Contains the actual topology data and give access to it (nature of these data heavily depends on the kind of topology).
    TopologyContainer *m_topologyContainer;
};



/** A class that contains a description of the topology (set of edges, triangles, adjacency information, ...) */
class SOFA_CORE_API TopologyContainer : public sofa::core::topology::BaseTopologyObject,
    public core::topology::BaseMeshTopology
{
public:
    SOFA_CLASS2(TopologyContainer, BaseTopologyObject, BaseMeshTopology);

protected:

    /** \brief Constructor.
    *
    */
    TopologyContainer()
    {}


    /// Destructor
    ~TopologyContainer() override;
public:
    void init() override;

    /// BaseMeshTopology API
    /// @{
    const SeqEdges& getEdges()         override { static SeqEdges     empty; return empty; }
    const SeqTriangles& getTriangles() override { static SeqTriangles empty; return empty; }
    const SeqQuads& getQuads()         override { static SeqQuads     empty; return empty; }
    const SeqTetrahedra& getTetrahedra()       override { static SeqTetrahedra    empty; return empty; }
    const SeqHexahedra& getHexahedra()         override { static SeqHexahedra     empty; return empty; }
    const SeqPrisms& getPrisms() override { static SeqPrisms empty; return empty; }
    const SeqPyramids& getPyramids() override { static SeqPyramids empty; return empty; }

    /** \brief Get the current revision of this mesh.
    *
    * This can be used to detect changes, however topological changes event should be used whenever possible.
    */
    int getRevision() const override { return m_changeList.getCounter(); }

    /// @}

    /// TopologyChange interactions
    /// @{
    const std::list<const TopologyChange *> &getChangeList() const { return m_changeList.getValue(); }

    const std::list<const TopologyChange *> &getStateChangeList() const { return m_stateChangeList.getValue(); }

    const Data <std::list<const TopologyChange *> > &getDataChangeList() const { return m_changeList; }

    const Data <std::list<const TopologyChange *> > &getDataStateChangeList() const { return m_stateChangeList; }

    /** \brief Adds a TopologyChange to the list.
    *
    * Needed by topologies linked to this one to know what happened and what to do to take it into account.
    *
    */
    virtual void addTopologyChange(const TopologyChange *topologyChange);

    /** \brief Adds a StateChange to the list.
    *
    * Needed by topologies linked to this one to know what happened and what to do to take it into account.
    *
    */
    virtual void addStateChange(const TopologyChange *topologyChange);

    /** \brief Provides an iterator on the first element in the list of TopologyChange objects.
     */
    std::list<const TopologyChange *>::const_iterator beginChange() const override;

    /** \brief Provides an iterator on the last element in the list of TopologyChange objects.
     */
    std::list<const TopologyChange *>::const_iterator endChange() const override;

    /** \brief Provides an iterator on the first element in the list of StateChange objects.
     */
    std::list<const TopologyChange *>::const_iterator beginStateChange() const override;

    /** \brief Provides an iterator on the last element in the list of StateChange objects.
     */
    std::list<const TopologyChange *>::const_iterator endStateChange() const override;


    /** \brief Free each Topology changes in the list and remove them from the list
    *
    */
    virtual void resetTopologyChangeList();

    /** \brief Free each State changes in the list and remove them from the list
    *
    */
    virtual void resetStateChangeList();

    ///@}

    /// TopologyHandler interactions
    ///@{
    const std::set<TopologyHandler*>& getTopologyHandlerList(sofa::geometry::ElementType elementType) const;

    /** \brief Adds a TopologyHandler, linked to a certain type of Element.
    */
    [[nodiscard]] bool addTopologyHandler(TopologyHandler* _TopologyHandler, sofa::geometry::ElementType elementType);

    /** \brief Remove a TopologyHandler, linked to a certain type of Element.
    */
    void removeTopologyHandler(TopologyHandler* _TopologyHandler, sofa::geometry::ElementType elementType);


    /** \brief Free each Topology changes in the list and remove them from the list
    *
    */
    void resetTopologyHandlerList();


    /** \ brief Generic function to link potential data (related to a type of element) with a topologyHandler
    *
    */
    virtual bool linkTopologyHandlerToData(TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType);

    /** \ brief Generic function to link potential data (related to a type of element) with a topologyHandler
    *
    */
    virtual bool unlinkTopologyHandlerToData(TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType);

    /// Array of topology modifications that have already occurred (addition) or will occur next (deletion).
    Data <std::list<const TopologyChange *> >m_changeList;

    /// Array of state modifications that have already occurred (addition) or will occur next (deletion).
    Data <std::list<const TopologyChange *> >m_stateChangeList;

    /// List of topology engines which will interact on all topological Data.
    std::array< std::set<TopologyHandler*>, sofa::geometry::NumberOfElementType> m_topologyHandlerListPerElement{};

    bool insertInNode( objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    bool removeInNode( objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }
};

} // namespace sofa::core::topology
