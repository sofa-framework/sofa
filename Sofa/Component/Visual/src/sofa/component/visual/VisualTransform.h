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
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::visual
{

/// Visually apply a (translation,rotation) transformation to visual elements rendering within a node or a sub-graph.
/// This can be used to change where elements are rendered, but has no effect on the actual simulation.
/// It can be used for example to correctly render forcefields applied to a mesh that is then transformed by a rigid DOF using DeformableOnRigidFrameMapping.

class SOFA_COMPONENT_VISUAL_API VisualTransform : public sofa::core::visual::VisualModel
{
public:
    SOFA_CLASS(VisualTransform,sofa::core::visual::VisualModel);

    typedef defaulttype::Rigid3Types::Coord Coord;

protected:
    VisualTransform();
    ~VisualTransform() override;
public:
    void fwdDraw(sofa::core::visual::VisualParams* vparams) override;
    void bwdDraw(sofa::core::visual::VisualParams* vparams) override;

    void draw(const sofa::core::visual::VisualParams* vparams) override;
    void doDrawVisual(const sofa::core::visual::VisualParams* vparams) override;
    void drawTransparent(const sofa::core::visual::VisualParams* vparams) override;

    Data<Coord> transform; ///< Transformation to apply
    Data<bool> recursive; ///< True to apply transform to all nodes below

    void push(const sofa::core::visual::VisualParams* vparams);
    void pop(const sofa::core::visual::VisualParams* vparams);

protected:
    int nbpush;
};


} // namespace sofa::component::visual
