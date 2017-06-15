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
#include "TransformPlaneConstraint.h"
#include <sofa/core/ObjectFactory.h>

namespace plugins
{

namespace pim
{

SOFA_DECL_CLASS(TransformPlaneConstraint)

int TransformPlaneConstraintClass = sofa::core::RegisterObject("")
        .add< TransformPlaneConstraint >()
        ;

TransformPlaneConstraint::TransformPlaneConstraint():
    d_planes(initData(&d_planes, "plane", "") )
    , d_outPlanes(initData(&d_outPlanes, "outPlane", "") )
    , d_rotation(initData(&d_rotation, "rotation", "") )
    , d_translation(initData(&d_translation, "translation", "") )
    , d_scale(initData(&d_scale, "scale", "") )
{
}

void TransformPlaneConstraint::init()
{
    addInput(&d_planes);
    addInput(&d_rotation);
    addInput(&d_translation);
    addInput(&d_scale);
    addOutput(&d_outPlanes);

    setDirtyValue();
}

void TransformPlaneConstraint::update()
{
    cleanDirty();

    const sofa::helper::vector<Vec10>& planes = d_planes.getValue();
    const sofa::helper::Quater<double>& rotation = d_rotation.getValue();
    const Vec3d& translation = d_translation.getValue();
    const double scale = d_scale.getValue();

    sofa::helper::vector<Vec10>& outPlanes = (*d_outPlanes.beginEdit());

    outPlanes.resize(planes.size());
    Vec3d result, p;

    for (unsigned int i=0; i<planes.size(); ++i)
    {
        for (unsigned int j=0; j<3; ++j)
        {
            p = Vec3d(planes[i][(j*3)], planes[i][(j*3)+1], planes[i][(j*3)+2]);
            result = p*scale;
            result = rotation.rotate(result);
            result += translation;
            outPlanes[i][(j*3)] = result[0];
            outPlanes[i][(j*3)+1] = result[1];
            outPlanes[i][(j*3)+2] = result[2];
        }
        outPlanes[i][9] = planes[i][9]*scale;
    }
    d_outPlanes.endEdit();
}

} // namespace pim

} // namespace plugins

