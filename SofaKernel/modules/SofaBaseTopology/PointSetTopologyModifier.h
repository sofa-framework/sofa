/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYMODIFIER_H
#include "config.h"

#include <sofa/helper/vector.h>
#include <sofa/core/topology/BaseTopology.h>

namespace sofa
{

namespace component
{

namespace topology
{
class PointSetTopologyContainer;


/**
* A class that can apply basic topology transformations on a set of points.
*/
class SOFA_BASE_TOPOLOGY_API PointSetTopologyModifier : public core::topology::TopologyModifier
{
public:
    SOFA_CLASS(PointSetTopologyModifier,core::topology::TopologyModifier);
protected:
    PointSetTopologyModifier()
        : TopologyModifier()
    {}

    virtual ~PointSetTopologyModifier() override {}
public:
    virtual void init() override;

    /** \brief Swap points i1 and i2.
    *
    */
    virtual void swapPoints(const int i1,const int i2);

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

    /** \brief Sends a message to warn that some points were added in this topology.
    *
    * \sa addPointsProcess
    */
    void addPointsWarning(const unsigned int nPoints,
            const sofa::helper::vector< core::topology::PointAncestorElem >& ancestorElems,
            const bool addDOF = true);


    /** \brief Extend the point container storage by nPoints.
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints);

    /** \brief Add a set of points
    * 
    * \sa addPoints
    */
    virtual void addPoints(const unsigned int nPoints, const bool addDOF = true);
 
    /** \brief Add a set of points
    * 
    * \sa addPoints
    */
    virtual void addPoints(const unsigned int nPoints,
                           const sofa::helper::vector< sofa::helper::vector< unsigned int> >& ancestors,
                           const sofa::helper::vector< sofa::helper::vector< double      > >& coefs,
                           const bool addDOF = true);

    /** \brief Add a set of points according to their ancestors topology elements
     *
     * \sa addPoints
     */
    void addPoints( const unsigned int nPoints,
                    const sofa::helper::vector< core::topology::PointAncestorElem >& ancestorElems,
                    const bool addDOF = true);


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

    /** \brief move input points indices to input new coords. Also propagate event
     *
     * @param id : list of indices to move
     * @param : ancestors list of ancestors to define relative new position
     * @param coefs : barycoef to locate new coord relatively to ancestors.
     * @moveDOF bool allowing the move (default true)
     */
    virtual void movePointsProcess (const sofa::helper::vector <unsigned int>& id,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs,
            const bool moveDOF = true);

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

    /** \brief Called by a topology to warn specific topologies linked to it that TopologyChange objects happened.
    *
    * ChangeList should contain all TopologyChange objects corresponding to changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @sa beginChange()
    * @sa endChange()
    */
    virtual void propagateTopologicalChanges() override;  // DEPRECATED

    /// TODO: doc ??
    void propagateTopologicalChangesWithoutReset();

    /// \brief function to propagate topological change events by parsing the list of topologyEngines linked to this topology.
    /// TODO: temporary duplication of topological events (commented by default)
    virtual void propagateTopologicalEngineChanges();


    /** \brief Called by a topology to warn the Mechanical Object component that points have been added or will be removed.
    *
    * StateChangeList should contain all TopologyChange objects corresponding to vertex changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @sa beginChange()
    * @sa endChange()
    */
    void propagateStateChanges() override;
    /// @}

    /** \notify the end for the current sequence of topological change events.
    */
    void notifyEndingEvent() override;

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(const sofa::helper::vector< unsigned int >& /*items*/) override
    { }

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> &/*index*/,
            const sofa::helper::vector<unsigned int> &/*inv_index*/)
    { }

private:
    PointSetTopologyContainer* 	m_container;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGYMODIFIER_H
