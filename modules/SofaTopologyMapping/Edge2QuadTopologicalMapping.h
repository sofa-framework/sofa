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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGE2QUADTOPOLOGICALMAPPING_H
#define SOFA_COMPONENT_TOPOLOGY_EDGE2QUADTOPOLOGICALMAPPING_H
#include "config.h"

#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <map>

#include <sofa/core/BaseMapping.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <SofaGeneralSimpleFem/RadiusContainer.h>


namespace sofa
{

namespace component
{

namespace topology
{

/**
* This class, called Edge2QuadTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
*
* INPUT TOPOLOGY = EdgeSetTopology
* OUTPUT TOPOLOGY = QuadSetTopology based on new DOFs, as the tubular skinning of INPUT TOPOLOGY.
*
* Edge2QuadTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
*
*/

class SOFA_TOPOLOGY_MAPPING_API Edge2QuadTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(Edge2QuadTopologicalMapping,sofa::core::topology::TopologicalMapping);

    typedef sofa::core::State<defaulttype::Rigid3Types>::VecCoord VecCoord;
    typedef sofa::core::State<defaulttype::Rigid3Types>::Coord Coord;
    typedef Coord::value_type Real;
    enum { M=Coord::spatial_dimensions };
    typedef defaulttype::Mat<M,M,Real> Mat;
    typedef defaulttype::Vec<M,Real> Vec;
    typedef helper::vector<unsigned int> VecIndex;

    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;


protected:

    /** \brief Constructor.
    *
    * @param from the topology issuing TopologyChange objects (the "source").
    * @param to   the topology for which the TopologyChange objects must be translated (the "target").
    */
    Edge2QuadTopologicalMapping()
        :  TopologicalMapping(),
           m_nbPointsOnEachCircle( initData(&m_nbPointsOnEachCircle, "nbPointsOnEachCircle", "Discretization of created circles")),
           m_radius( initData(&m_radius, "radius", "Radius of created circles")),
           edgeList(initData(&edgeList, "edgeList", "list of input edges for the topological mapping: by default, all considered")),
           flipNormals(initData(&flipNormals, bool(false), "flipNormals", "Flip Normal ? (Inverse point order when creating quad)")),
        m_radiusContainer(NULL)
    {
    }

    /** \brief Destructor.
    *
    * Does nothing.
    */
    virtual ~Edge2QuadTopologicalMapping()
    {}
public:
    /** \brief Initializes the target BaseTopology from the source BaseTopology.
    */
    virtual void init() override;


    /** \brief Translates the TopologyChange objects from the source to the target.
    *
    * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
    * reflect the effects of the first topology changes on the second topology.
    *
    */
    virtual void updateTopologicalMappingTopDown() override;

    virtual unsigned int getFromIndex(unsigned int ind) override;

protected:
    Data<unsigned int> m_nbPointsOnEachCircle; ///< number of points to create along the circles around each point of the input topology (10 by default)
    Data<double> m_radius;	///< radius of the circles around each point of the input topology (1 by default)

    Data<VecIndex> edgeList; ///< list of input edges for the topological mapping: by default, all considered
    Data<bool> flipNormals; ///< Flip Normal ? (Inverse point order when creating quad)

    
    container::RadiusContainer* m_radiusContainer;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_EDGE2QUADTOPOLOGICALMAPPING_H
