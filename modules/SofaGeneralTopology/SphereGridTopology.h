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
#ifndef SOFA_COMPONENT_TOPOLOGY_SPHEREGRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_SPHEREGRIDTOPOLOGY_H
#include "config.h"

#include <SofaBaseTopology/GridTopology.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace topology
{

/** \brief Define a sphere grid topology
 * Paramenters are its @sa d_radius and discretisation .
 * Position and direction are set by @sa d_center and @sa d_axis
 * nz discretisation is along the sphere axis
  */
class SOFA_GENERAL_TOPOLOGY_API SphereGridTopology : public GridTopology
{
public:
    SOFA_CLASS(SphereGridTopology,GridTopology);
    typedef sofa::defaulttype::Vector3 Vector3;
protected:
    /// Default constructor
    SphereGridTopology();
    /// Constructor with grid size by int
    SphereGridTopology(int nx, int ny, int nz);

public:
    /** \brief Overload method of @sa GridTopology::getPoint.
     * Get Point in grid @return Vector3 given its @param id i. Will call @sa getPointInGrid.
     * */
    Vector3 getPoint(int i) const override;

    /** \brief Overload method of @sa GridTopology::getPointInGrid.
     * Get Point in grid @return Vector3 given its position in grid @param i, @param j, @param k
     * */
    Vector3 getPointInGrid(int i, int j, int k) const override;

    /// Set Sphere grid center by @param 3 SReal
    void setCenter(SReal x, SReal y, SReal z);
    /// Set Sphere axis center by @param 3 SReal
    void setAxis(SReal x, SReal y, SReal z);
    /// Set Sphere radius from @param SReal
    void setRadius(SReal radius);

public:
    /// Data storing the center position
    Data< Vector3 > d_center;
    /// Data storing the axis direction
    Data< Vector3 > d_axis;
    /// Data storing the radius value
    Data< SReal > d_radius;

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_SPHEREGRIDTOPOLOGY_H
