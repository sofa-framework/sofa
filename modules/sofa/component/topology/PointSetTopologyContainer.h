/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

class MeshLoader;

namespace topology
{
using core::componentmodel::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID			PointID;

/** The container class that stores a set of points and provides access
to each point. This set of point may be a subset of the DOF of the mechanical model */
class SOFA_COMPONENT_CONTAINER_API PointSetTopologyContainer : public core::componentmodel::topology::TopologyContainer
{
public:

    PointSetTopologyContainer(int nPoints = 0);

    virtual ~PointSetTopologyContainer() {}

    /// Procedural creation methods
    /// @{
    virtual void clear();
    virtual void addPoint(double px, double py, double pz);
    /// @}

    virtual void init();


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

    /// BaseMeshTopology API
    /// @{
    /** \brief Returns the number of vertices in this topology.
    *
    */
    int getNbPoints() const { return (int)nbPoints.getValue(); }
    void setNbPoints(int n);
    virtual bool hasPos() const;
    virtual double getPX(int i) const;
    virtual double getPY(int i) const;
    virtual double getPZ(int i) const;
    /// @}

protected:
    virtual void loadFromMeshLoader(sofa::component::MeshLoader* loader);

protected:
    Data<unsigned int> nbPoints;

private:
    typedef defaulttype::Vec3Types InitTypes;
    InitTypes::VecCoord initPoints;
    DataPtr<InitTypes::VecCoord> d_initPoints;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGYCONTAINER_H
