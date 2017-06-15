/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_VISUALMODEL_VISUALTRANSFORM_H
#define SOFA_COMPONENT_VISUALMODEL_VISUALTRANSFORM_H
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

/// Visually apply a (translation,rotation) transformation to visual elements rendering within a node or a sub-graph.
/// This can be used to change where elements are rendered, but has no effect on the actual simulation.
/// It can be used for example to correctly render forcefields applied to a mesh that is then transformed by a rigid DOF using DeformableOnRigidFrameMapping.

class SOFA_GENERAL_VISUAL_API VisualTransform : public sofa::core::visual::VisualModel
{
public:
    SOFA_CLASS(VisualTransform,sofa::core::visual::VisualModel);

    typedef defaulttype::Rigid3Types::Coord Coord;

protected:
    VisualTransform();
    virtual ~VisualTransform();
public:
    void fwdDraw(sofa::core::visual::VisualParams* vparams);
    void bwdDraw(sofa::core::visual::VisualParams* vparams);

    void draw(const sofa::core::visual::VisualParams* vparams);
    void drawVisual(const sofa::core::visual::VisualParams* vparams);
    void drawTransparent(const sofa::core::visual::VisualParams* vparams);

    Data<Coord> transform;
    Data<bool> recursive;

    void push(const sofa::core::visual::VisualParams* vparams);
    void pop(const sofa::core::visual::VisualParams* vparams);

protected:
    int nbpush;
};


} // visual

} // component

} // sofa

#endif // SOFA_COMPONENT_VISUALMODEL_VISUALTRANSFORM_H
