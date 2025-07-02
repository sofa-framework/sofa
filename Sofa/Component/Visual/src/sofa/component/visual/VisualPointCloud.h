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
#include <sofa/core/State.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/type/trait/Rebind.h>

namespace sofa::component::visual
{

inline constexpr sofa::helper::Item pointDrawMode{
    .key = "Point", .description = "Coordinates are displayed with points"};
inline constexpr sofa::helper::Item sphereDrawMode{
    .key = "Sphere", .description = "Coordinates are displayed using spheres"};
inline constexpr sofa::helper::Item frameDrawMode{
    .key = "Frame", .description = "Coordinates are displayed using oriented frames"};

template<class DataTypes>
concept hasWriteOpenGlMatrix = requires(typename DataTypes::Coord& c, float glTransform[16])
{
    c.writeOpenGlMatrix(glTransform);
};

MAKE_SELECTABLE_ITEMS(RegularCoordDrawMode,
    pointDrawMode, sphereDrawMode
);

MAKE_SELECTABLE_ITEMS(OridentedCoordDrawMode,
    pointDrawMode, sphereDrawMode, frameDrawMode
);

template<class DataTypes>
using CoordDrawMode = std::conditional_t<
    hasWriteOpenGlMatrix<DataTypes>,
        OridentedCoordDrawMode,
        RegularCoordDrawMode>;

template <class DataTypes>
class VisualPointCloud : public core::visual::VisualModel
{
public:
    SOFA_CLASS(VisualPointCloud, core::visual::VisualModel);

private:
    using VecCoord = VecCoord_t<DataTypes>;
    using Real = Real_t<DataTypes>;
    using VecFloat = sofa::type::vector<float>;
    using VecColor = sofa::type::vector<type::RGBAColor>;
    using DrawMode = CoordDrawMode<DataTypes>;

    static DrawMode defaultDrawMode();

public:
    Data<VecCoord> d_position;
    Data<DrawMode> d_drawMode;
    Data<float> d_pointSize;
    Data<VecFloat> d_sphereRadius;
    Data<type::RGBAColor> d_color;

    Data<bool> d_showIndices;
    Data<float> d_indicesScale;
    Data<type::RGBAColor> d_indicesColor;

    void computeBBox(const core::ExecParams*, bool) override;

private:
    VisualPointCloud();

    void doDrawVisual(const core::visual::VisualParams* vparams) override;

    void drawFrames(
        const core::visual::VisualParams* vparams,
        type::RGBAColor color) requires hasWriteOpenGlMatrix<DataTypes>;

    void drawIndices(const core::visual::VisualParams* vparams) const;

    type::vector<type::Vec3> convertCoord() const;
};


}  // namespace sofa::component::visual
