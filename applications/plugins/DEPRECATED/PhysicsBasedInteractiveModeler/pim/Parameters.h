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
#ifndef PLUGINS_PIM_PARAMETERS_H
#define PLUGINS_PIM_PARAMETERS_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/Quater.h>

namespace plugins
{

namespace pim
{

using namespace sofa::defaulttype;

/**
 *
 */
class Parameters : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(Parameters,sofa::core::objectmodel::BaseObject);
    typedef Vec<16, double> Vec16;

    Parameters();

    ~Parameters() {}

    void init();

    void update();

    Data<Vec16> d_transformMatrix;
    Data<Vec3d> d_translation;
    Data<Vec3d> d_rotation;
    Data<double> d_scale;
    Data<Vec3d> d_scale3d;
    Data<Mat3x3d> d_rotationMat;
    Data<sofa::helper::Quater<double> > d_rotationQuat;
    Data<Vec3d> d_center;
    Data<Vec3d> d_axis;
    Data<double> d_angle;
    Data<std::string> d_uterus;
    Data<std::string> d_output;
    Data<std::string> d_muscleLayer;
    Data<std::string> d_fatLayer;
    Data<std::string> d_intersectionLayer;
};

} // namespace pim

} // namespace plugins

#endif
