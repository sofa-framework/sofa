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
#include <sofa/component/visual/config.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::visual
{

namespace
{
    using sofa::type::Vec3;
}

class SOFA_COMPONENT_VISUAL_API VisualGrid : public core::visual::VisualModel
{
public:
    SOFA_CLASS(VisualGrid, VisualModel);

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);

    enum PLANE
    {
        PLANE_X = 0,
        PLANE_Y = 1,
        PLANE_Z = 2
    };

    Data<std::string> d_plane; ///< Plane of the grid


    Data<float> d_size; ///< Size of the squared grid
    Data<int> d_nbSubdiv; ///< Number of subdivisions

    Data<sofa::type::RGBAColor> d_color; ///< Color of the lines in the grid. default=(0.34,0.34,0.34,1.0)
    Data<float> d_thickness; ///< Thickness of the lines in the grid
    core::objectmodel::lifecycle::RemovedData d_draw {this, "v23.06", "23.12", "draw", "Use the 'enable' data field instead of 'draw'"};


    VisualGrid();
    ~VisualGrid() override = default;

    void init() override;
    void reinit() override;
    void doDrawVisual(const core::visual::VisualParams*) override;
    void updateVisual() override;
    void buildGrid();

protected:

    PLANE internalPlane;

    ///< Pre-computed points used to draw the grid
    sofa::type::vector<Vec3> m_drawnPoints;

};

} // namespace sofa::component::visual
