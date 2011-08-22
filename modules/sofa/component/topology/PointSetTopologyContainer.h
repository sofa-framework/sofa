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
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace topology
{
class PointSetTopologyModifier;
using core::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID			PointID;

/** The container class that stores a set of points and provides access
to each point. This set of point may be a subset of the DOF of the mechanical model */
class SOFA_COMPONENT_CONTAINER_API PointSetTopologyContainer : public core::topology::TopologyContainer
{
public:
    SOFA_CLASS(PointSetTopologyContainer,core::topology::TopologyContainer);

    friend class PointSetTopologyModifier;
    typedef defaulttype::Vec3Types InitTypes;


    PointSetTopologyContainer(int nPoints = 0);

    virtual ~PointSetTopologyContainer() {}


    virtual void init();



    /// Procedural creation methods
    /// @{
    virtual void clear();
    virtual void addPoint(double px, double py, double pz);
    /// @}



    /// BaseMeshTopology API
    /// @{

    /** \brief Returns the number of vertices in this topology. */
    int getNbPoints() const { return (int)nbPoints.getValue(); }

    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    virtual unsigned int getNumberOfElements() const;

    /** \brief Returns a reference to the Data of points array container. */
    Data<InitTypes::VecCoord>& getPointDataArray() {return d_initPoints;}

    /** \brief Set the number of vertices in this topology. */
    void setNbPoints(int n);


    /** \brief check if vertices in this topology have positions. */
    virtual bool hasPos() const;

    /** \brief Returns the X coordinate of the ith DOF. */
    virtual double getPX(int i) const;

    /** \brief Returns the Y coordinate of the ith DOF. */
    virtual double getPY(int i) const;

    /** \brief Returns the Z coordinate of the ith DOF. */
    virtual double getPZ(int i) const;

    /// @}



    /// Dynamic Topology API
    /// @{

    /** \brief Checks if the Topology is coherent
     *
     */
    virtual bool checkTopology() const;

    /** \brief add one DOF in this topology (simply increment the number of DOF)
     *
     */
    void addPoint();


    /** \brief add a number of DOFs in this topology (simply increase the number of DOF according to this parameter)
     *
     * @param The number of point to add.
     */
    void addPoints(const unsigned int nPoints);


    /** \brief remove one DOF in this topology (simply decrement the number of DOF)
     *
     */
    void removePoint();


    /** \brief remove a number of DOFs in this topology (simply decrease the number of DOF according to this parameter)
     *
     * @param The number of point to remove.
     */
    void removePoints(const unsigned int nPoints);

    /// @}

    inline friend std::ostream& operator<< (std::ostream& out, const PointSetTopologyContainer& /*t*/)
    {
        return out;
    }

    inline friend std::istream& operator>>(std::istream& in, PointSetTopologyContainer& /*t*/)
    {
        return in;
    }




protected:
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    virtual void updateTopologyEngineGraph();

    virtual void displayDataGraph();

    /// List of Topological Data link to this Data. TODO: check if necessary or doublon with engine list
    sofa::helper::list<sofa::core::objectmodel::BaseData*> m_topologyDataDependencies;

    /// graph map
    sofa::helper::vector < sofa::helper::vector <std::string> > m_dataGraph;
    sofa::helper::vector < sofa::helper::vector <std::string> > m_enginesGraph;

    sofa::helper::list <sofa::core::topology::TopologyEngine *> m_enginesList;

#endif

    Data<unsigned int> nbPoints;

    Data<InitTypes::VecCoord> d_initPoints;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGYCONTAINER_H
