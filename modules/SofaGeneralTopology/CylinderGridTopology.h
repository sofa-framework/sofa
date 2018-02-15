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
#ifndef SOFA_COMPONENT_TOPOLOGY_CYLINDERGRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_CYLINDERGRIDTOPOLOGY_H
#include "config.h"

#include <SofaBaseTopology/GridTopology.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace topology
{

/** \brief Define a cylinder grid topology
 * Paramenters are its @sa d_radius and @sa d_length .
 * Position and direction are set by @sa d_center and @sa d_axis
 * nz discretisation is along the cylinder axis
  */
class SOFA_GENERAL_TOPOLOGY_API CylinderGridTopology : public GridTopology
{
public:
    SOFA_CLASS(CylinderGridTopology,GridTopology);
    typedef sofa::defaulttype::Vector3 Vector3;
protected:
    /// Default constructor
    CylinderGridTopology();
    /// Constructor with grid size by int
    CylinderGridTopology(int nx, int ny, int nz);

public:
    /** \brief Overload method of @sa GridTopology::getPointInGrid.
     * Get Point in grid @return Vector3 given its position in grid @param i, @param j, @param k
     * */
    Vector3 getPointInGrid(int i, int j, int k) const override;

    /// Set Cylinder grid center by @param 3 SReal
    void setCenter(SReal x, SReal y, SReal z);
    /// Set Cylinder axis center by @param 3 SReal
    void setAxis(SReal x, SReal y, SReal z);
    /// Set Cylinder radius from @param SReal
    void setRadius(SReal radius);
    /// Set Cylinder length from @param SReal
    void setLength(SReal length);

public:
    /// Data storing the center position
    Data< Vector3 > d_center;
    /// Data storing the axis direction
    Data< Vector3 > d_axis;
    /// Data storing the radius value
    Data< SReal > d_radius;
    /// Data storing the length value
    Data< SReal > d_length;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
