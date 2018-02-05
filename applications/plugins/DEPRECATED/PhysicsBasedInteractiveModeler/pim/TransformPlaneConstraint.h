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
#ifndef PLUGINS_PIM_TRANSFORMPLANECONSTRAINT_H
#define PLUGINS_PIM_TRANSFORMPLANECONSTRAINT_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/Quater.h>

namespace plugins
{

namespace pim
{

using namespace sofa::defaulttype;

/**
 *
 */
class TransformPlaneConstraint : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(TransformPlaneConstraint,sofa::core::objectmodel::BaseObject);
    typedef Vec<10, double> Vec10;

    TransformPlaneConstraint();

    ~TransformPlaneConstraint() {}

    void init();

    void update();

    Data< sofa::helper::vector<Vec10> > d_planes, d_outPlanes;
    Data<sofa::helper::Quater<double> > d_rotation;
    Data<Vec3d> d_translation;
    Data<double> d_scale;
};

} // namespace pim

} // namespace plugins

#endif
