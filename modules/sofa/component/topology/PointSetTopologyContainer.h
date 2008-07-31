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
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYCONTAINER_H

#include <sofa/helper/vector.h>
#include <sofa/core/componentmodel/topology/BaseTopology.h>

namespace sofa
{

namespace component
{

namespace topology
{
class PointSetTopologyModifier;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::PointID PointID;

/** The container class that stores a set of points and provides access
to each point. This set of point may be a subset of the DOF of the mechanical model */
class PointSetTopologyContainer : public core::componentmodel::topology::TopologyContainer
{
public:
    PointSetTopologyContainer();

    PointSetTopologyContainer(const int nPoints);

    virtual ~PointSetTopologyContainer() {}

    virtual void init();

    virtual void clear();

    /// BaseMeshTopology API
    /// @{
    /** \brief Returns the number of vertices in this topology.
    *
    */
    int getNbPoints() const {return nbPoints;}

    /** \brief Called by a topology to warn specific topologies linked to it that TopologyChange objects happened.
    *
    * ChangeList should contain all TopologyChange objects corresponding to changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @sa firstChange()
    * @sa lastChange()
    */
    void propagateTopologicalChanges();

    /** \brief Called by a topology to warn the Mechanical Object component that points have been added or will be removed.
    *
    * StateChangeList should contain all TopologyChange objects corresponding to vertex changes in this topology
    * that just happened (in the case of creation) or are about to happen (in the case of destruction) since
    * last call to propagateTopologicalChanges.
    *
    * @sa firstChange()
    * @sa lastChange()
    */
    void propagateStateChanges();
    /// @}

    /** \brief Checks if the Topology is coherent
    *
    */
    virtual bool checkTopology() const;

    void addPoint();

    void addPoints(const unsigned int nPoints);

    void removePoint();

    void removePoints(const unsigned int nPoints);

    inline friend std::ostream& operator<< (std::ostream& out, const PointSetTopologyContainer& /*t*/)
    {
        return out;
    }

    inline friend std::istream& operator>>(std::istream& in, PointSetTopologyContainer& /*t*/)
    {
        return in;
    }

private:
    unsigned int	nbPoints;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGYCONTAINER_H
