/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include "Parameters.h"
#include <sofa/core/ObjectFactory.h>


namespace plugins
{

namespace pim
{

SOFA_DECL_CLASS(Parameters)

int ParametersClass = sofa::core::RegisterObject("")
        .add< Parameters >()
        ;

Parameters::Parameters():
    d_transformMatrix(initData(&d_transformMatrix, "matrix", "") )
    , d_translation(initData(&d_translation, "translation", "") )
    , d_rotation(initData(&d_rotation, "rotation", "") )
    , d_scale(initData(&d_scale, "scale", "") )
    , d_scale3d(initData(&d_scale3d, "scale3d", "") )
    , d_rotationMat(initData(&d_rotationMat, "rotationMat", "") )
    , d_rotationQuat(initData(&d_rotationQuat, "rotationQuat", "") )
    , d_center(initData(&d_center, "center", "") )
    , d_axis(initData(&d_axis, "axis", "") )
    , d_angle(initData(&d_angle, "angle", "") )
    , d_uterus(initData(&d_uterus, "uterus", "") )
    , d_output(initData(&d_output, "output", "") )
    , d_muscleLayer(initData(&d_muscleLayer, "muscleLayer", "") )
    , d_fatLayer(initData(&d_fatLayer, "fatLayer", "") )
    , d_intersectionLayer(initData(&d_intersectionLayer, "intersectionLayer", "") )
{
}

void Parameters::init()
{
    addInput(&d_transformMatrix);
    addOutput(&d_translation);
    addOutput(&d_rotation);
    addInput(&d_scale);
    addOutput(&d_scale3d);
    addOutput(&d_rotationMat);
    addOutput(&d_rotationQuat);
    addOutput(&d_center);
    addOutput(&d_axis);
    addOutput(&d_angle);
    addOutput(&d_uterus);
    addOutput(&d_output);
    addOutput(&d_muscleLayer);
    addOutput(&d_fatLayer);
    addOutput(&d_intersectionLayer);

    setDirtyValue();
}

void Parameters::update()
{
    cleanDirty();

    const Vec16& transformMatrix = d_transformMatrix.getValue();
    const double& scale = d_scale.getValue();

    Vec3d& translation = (*d_translation.beginEdit());
    translation = Vec3d(transformMatrix[12], transformMatrix[13], transformMatrix[14])*scale;


    Vec3d& rotation = (*d_rotation.beginEdit());
    sofa::helper::Quater<double>& quat = (*d_rotationQuat.beginEdit());

    Mat3x3d& rotationMat = (*d_rotationMat.beginEdit());

    rotationMat = Mat3x3d(Vec3d(transformMatrix[0], transformMatrix[4], transformMatrix[8]),
            Vec3d(transformMatrix[1], transformMatrix[5], transformMatrix[9]),
            Vec3d(transformMatrix[2], transformMatrix[6], transformMatrix[10]));

    quat.fromMatrix(rotationMat);

    rotation = quat.toEulerVector()*180.0/M_PI;


    Vec3d& scale3d = (*d_scale3d.beginEdit());
    scale3d = Vec3d(scale, scale, scale);

    d_translation.endEdit();
    d_rotation.endEdit();
    d_scale3d.endEdit();
    d_rotationMat.endEdit();
    d_rotationQuat.endEdit();
}

} // namespace pim

} // namespace plugins

