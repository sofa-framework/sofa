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
template<class DataTypes>
class PointSetTopology;

template<class DataTypes>
class PointSetTopologyModifier;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::PointID PointID;

/** The container class that stores a set of points and provides access
to each point. This set of point may be a subset of the DOF of the mechanical model */
class PointSetTopologyContainer : public core::componentmodel::topology::TopologyContainer
{
public:
    template <typename DataTypes>
    friend class PointSetTopologyModifier;

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

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGYCONTAINER_H
