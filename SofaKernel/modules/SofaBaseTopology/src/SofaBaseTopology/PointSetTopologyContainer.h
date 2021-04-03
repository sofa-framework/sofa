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
#include <SofaBaseTopology/config.h>

#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/topology/BaseTopology.h>

namespace sofa::component::topology
{
class PointSetTopologyModifier;

/** The container class that stores a set of points and provides access
to each point. This set of point may be a subset of the DOF of the mechanical model */
class SOFA_SOFABASETOPOLOGY_API PointSetTopologyContainer : public core::topology::TopologyContainer
{
public:
    SOFA_CLASS(PointSetTopologyContainer,core::topology::TopologyContainer);

    friend class PointSetTopologyModifier;
    typedef defaulttype::Vec3Types InitTypes;

protected:
    PointSetTopologyContainer(Size nPoints = 0);

    ~PointSetTopologyContainer() override {}
public:

    void init() override;

    /// Procedural creation methods
    /// @{
    void clear() override;
    void addPoint(double px, double py, double pz) override;
    /// @}



    /// BaseMeshTopology API
    /// @{

    /** \brief Returns the number of vertices in this topology. */
    Size getNbPoints() const override { return nbPoints.getValue(); }

    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    virtual Size getNumberOfElements() const;

    /** \brief Returns a reference to the Data of points array container. */
    Data<InitTypes::VecCoord>& getPointDataArray() {return d_initPoints;}

    /** \brief Set the number of vertices in this topology. */
    void setNbPoints(Size n) override;


    /** \brief check if vertices in this topology have positions. */
    bool hasPos() const override;

    /** \brief Returns the X coordinate of the ith DOF. */
    SReal getPX(Index i) const override;

    /** \brief Returns the Y coordinate of the ith DOF. */
    SReal getPY(Index i) const override;

    /** \brief Returns the Z coordinate of the ith DOF. */
    SReal getPZ(Index i) const override;

    /** \brief Returns the type of the topology */
    sofa::core::topology::TopologyElementType getTopologyType() const override {return sofa::core::topology::TopologyElementType::POINT;}
    
    /// @}



    /// Dynamic Topology API
    /// @{

    /** \brief Checks if the Topology is coherent
     *
     */
    bool checkTopology() const override;

    /** \brief add one DOF in this topology (simply increment the number of DOF)
     *
     */
    void addPoint();


    /** \brief add a number of DOFs in this topology (simply increase the number of DOF according to this parameter)
     *
     * @param The number of point to add.
     */
    void addPoints(const Size nPoints);


    /** \brief remove one DOF in this topology (simply decrement the number of DOF)
     *
     */
    void removePoint();


    /** \brief remove a number of DOFs in this topology (simply decrease the number of DOF according to this parameter)
     *
     * @param The number of point to remove.
     */
    void removePoints(const Size nPoints);

    /// @}

    inline friend std::ostream& operator<< (std::ostream& out, const PointSetTopologyContainer& /*t*/)
    {
        return out;
    }

    inline friend std::istream& operator>>(std::istream& in, PointSetTopologyContainer& /*t*/)
    {
        return in;
    }

    const sofa::helper::vector<PointID>& getPoints() const;

protected:
    /// \brief Function creating the data graph linked to d_point
    void updateTopologyHandlerGraph() override;

    /// \brief functions to really update the graph of Data/DataEngines linked to the different Data array, using member variable.
    virtual void updateDataEngineGraph(sofa::core::objectmodel::BaseData& my_Data, std::list<sofa::core::topology::TopologyHandler *>& my_enginesList);


    /// Use a specific boolean @see m_pointTopologyDirty in order to know if topology Data is dirty or not.
    /// Set/Get function access to this boolean
    void setPointTopologyToDirty();
    void cleanPointTopologyFromDirty();
    const bool& isPointTopologyDirty() {return m_pointTopologyDirty;}

    /// \brief function to add a TopologyHandler to the current list of engines.
    void addEngineToList(sofa::core::topology::TopologyHandler * _engine);

    /// \brief functions to display the graph of Data/DataEngines linked to the different Data array, using member variable.
    virtual void displayDataGraph(sofa::core::objectmodel::BaseData& my_Data);

public:
    Data<InitTypes::VecCoord> d_initPoints; ///< Initial position of points    

    Data<bool> d_checkTopology; ///< Bool parameter to activate internal topology checks in several methods 

protected:


    /// Boolean used to know if the topology Data of this container is dirty
    bool m_pointTopologyDirty;

    /// List of engines related to this specific container
    std::list<sofa::core::topology::TopologyHandler *> m_enginesList;

    /// \brief variables used to display the graph of Data/DataEngines linked to this Data array.
    sofa::helper::vector < sofa::helper::vector <std::string> > m_dataGraph;
    sofa::helper::vector < sofa::helper::vector <std::string> > m_enginesGraph;

private:
    
    Data<Size> nbPoints; ///< Number of points

    Data<sofa::helper::vector<PointID> > points; ///< List of point indices
};

} //namespace sofa::component::topology
