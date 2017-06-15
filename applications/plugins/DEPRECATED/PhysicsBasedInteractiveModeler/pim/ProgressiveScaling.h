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
#ifndef PLUGINS_PIM_PROGRESSIVESCALING_H
#define PLUGINS_PIM_PROGRESSIVESCALING_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace plugins
{

namespace pim
{

using namespace sofa::core::behavior;
using namespace sofa::core::topology;
using namespace sofa::core::objectmodel;

/**
 * This class apply a progresive scaling all over the points of a mechanical object.
 */
template <class DataTypes>
class ProgressiveScaling : public sofa::core::DataEngine
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef sofa::defaulttype::Vec<9,Real> Vec6;
public:
    SOFA_CLASS(SOFA_TEMPLATE(ProgressiveScaling,DataTypes),sofa::core::DataEngine);
    ProgressiveScaling();

    ~ProgressiveScaling() {}

    void init();

    void reinit();

    void reset();

    void update();

    void handleEvent(sofa::core::objectmodel::Event *event);

    void draw();

    Data<VecCoord> f_X0;
    Data<VecCoord> f_X;
    VecCoord local_X0;
    Data<double> from_scale, to_scale;

    Data<double> d_scale;
    Data<Vec3d> d_axis;
    Data<Vec3d> d_center;
    Data<double> d_angle;
    Data<double> step;
    Data<std::string> d_file_in;

    Vector3 cm0;
    double progressiveScale;

    Data<Vector3> right_femoral_head;
    Data<Vector3> left_femoral_head;
    Vector3 scaling_center;
    VecCoord uterus_position_from_scaling_center;
    Vector3 rotation_center;
    Data<double> rotation;
    Quat quat;
    Data< sofa::helper::vector<Vec6> > boxes;
    Vector3 new_center;


    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const ProgressiveScaling<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_PROGRESSIVESCALING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_ENGINE_API ProgressiveScaling<defaulttype::Vec3dTypes>;
extern template class SOFA_COMPONENT_ENGINE_API ProgressiveScaling<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_ENGINE_API ProgressiveScaling<defaulttype::Vec3fTypes>;
extern template class SOFA_COMPONENT_ENGINE_API ProgressiveScaling<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace pim

} // namespace plugins

#endif
