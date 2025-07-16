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

namespace sofa::component::visual
{

MAKE_SELECTABLE_ITEMS(VectorFieldDrawMode,
    sofa::helper::Item{.key = "Line", .description = "Coordinates are displayed using lines"},
    sofa::helper::Item{.key = "Cylinder", .description = "Coordinates are displayed using cylinders"},
    sofa::helper::Item{.key = "Arrow", .description = "Coordinates are displayed using arrows"},
);

template <class DataTypes>
class VisualVectorField : public core::visual::VisualModel
{
public:
    SOFA_CLASS(VisualVectorField, core::visual::VisualModel);

private:
    using VecCoord = VecCoord_t<DataTypes>;
    using VecDeriv = VecDeriv_t<DataTypes>;

public:
    Data<VecCoord> d_position;
    Data<VecDeriv> d_vector;
    Data<SReal> d_vectorScale;
    Data<VectorFieldDrawMode> d_drawMode;
    Data<type::RGBAColor> d_color;

    void computeBBox(const core::ExecParams*, bool) override;

private:
    VisualVectorField();

    void doDrawVisual(const core::visual::VisualParams* vparams) override;
};


}  // namespace sofa::component::visual
