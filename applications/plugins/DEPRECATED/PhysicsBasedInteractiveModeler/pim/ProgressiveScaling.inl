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
#ifndef PLUGINS_PIM_PROGRESSIVESCALING_INL
#define PLUGINS_PIM_PROGRESSIVESCALING_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "ProgressiveScaling.h"
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/gui/GUIManager.h>
#include <flowvr/render/mesh.h>

namespace plugins
{

namespace pim
{

using namespace sofa::helper;
using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;

template <class DataTypes>
ProgressiveScaling<DataTypes>::ProgressiveScaling():
    f_X0(initData(&f_X0, "rest_position", "Rest position coordinates of the degrees of freedom") )
    , f_X(initData(&f_X, "position","scaled position coordiates") )
    , d_scale(initData(&d_scale, "scale", "initial scale applied to the degrees of freedom") )
    , d_axis(initData(&d_axis, Vector3(0, 0, 0), "axis", "") )
    , d_center(initData(&d_center,  Vector3(0, 0, 0), "center", "") )
    , d_angle(initData(&d_angle,  20.0, "angle", "") )
    , step(initData(&step,  0.001, "step", "") )
    , d_file_in(initData(&d_file_in, "filename", "") )
    , cm0(Vector3(0,0,0)), progressiveScale(1.0)
{
    f_listening.setValue(true);
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::init()
{
    // set the middle point between the femoral heads as the scaling center
    const double& scale = d_scale.getValue();
    const VecCoord& X0 = f_X0.getValue();

    const Vec3d& center = d_center.getValue()*scale;

    scaling_center = X0[0]*scale;
    double min = ((X0[0]*scale) - center).norm();

    for(unsigned int i=1; i<X0.size(); ++i)
    {
        if ( ((X0[i]*scale) - center).norm() < min )
        {
            scaling_center = X0[i]*scale;
            min = ((X0[i]*scale) - center).norm();
        }
    }

    progressiveScale = 0.1;
    VecCoord& X = *(f_X.beginEdit());
    X.resize(X0.size());
    f_X.endEdit();

    uterus_position_from_scaling_center.resize(X0.size());
    const Vec3d& axis = d_axis.getValue();
    const double& angle = d_angle.getValue();

    quat.axisToQuat(axis, (angle*M_PI)/180.0);

    for (unsigned int i=0; i<X0.size(); ++i)
    {
        uterus_position_from_scaling_center[i] = quat.rotate((X0[i]*scale) - scaling_center) ;
    }

    addInput(&f_X0);
    addOutput(&f_X);
    setDirtyValue();
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::reinit()
{
    const Vec3d& axis = d_axis.getValue();
    const double& angle = d_angle.getValue();

    quat.axisToQuat(axis, (angle*M_PI)/180.0);
    setDirtyValue();
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::reset()
{
    progressiveScale = 0.1;
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::update()
{
    cleanDirty();

    VecCoord& X = *(f_X.beginEdit());
    for( unsigned i=0; i<uterus_position_from_scaling_center.size(); ++i )
    {
        Real x=0.0,y=0.0,z=0.0;
        DataTypes::get(x,y,z,uterus_position_from_scaling_center[i]);
        Vector3 result = scaling_center + (Vector3(x, y, z)*progressiveScale);
        DataTypes::set(X[i], result.x(), result.y(), result.z());
    }

    f_X.endEdit();
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateEndEvent *>(event))
    {
        if (progressiveScale < 1)
        {
            progressiveScale += step.getValue();
            setDirtyValue();
        }
        else
        {
            sofa::simulation::Node* groot = dynamic_cast<sofa::simulation::Node*>( sofa::gui::GUIManager::CurrentSimulation() );
            if (groot!=NULL)
            {
                const std::string file_in = d_file_in.getValue();
                sofa::simulation::getSimulation()->exportOBJ(groot, file_in.c_str());

                flowvr::render::Mesh obj;

                if (!obj.load(file_in.c_str()))
                {
                    std::cerr << "Failed to read "<<file_in<<std::endl;
                    return;
                }

                const Vec3d& axis = d_axis.getValue();
                const double& angle = d_angle.getValue();
                const double& scale = d_scale.getValue();

                quat.axisToQuat(axis, (-angle*M_PI)/180.0);
                Quat quat2;
                quat2.axisToQuat(Vec3d(1,0,0), (-90*M_PI)/180.0);

                for (int i=0; i<obj.nbp(); i++)
                {
                    Vec3d p = Vec3d(obj.PP(i)[0] - scaling_center[0], obj.PP(i)[1] - scaling_center[1], obj.PP(i)[2] - scaling_center[2]);
                    Vec3d r = (scaling_center + quat.rotate(p))/scale;
                    r = quat2.rotate(r);
                    obj.PP(i)[0] = r[0]; obj.PP(i)[1] = r[1]; obj.PP(i)[2] = r[2];
                }

                std::cout << "Saving result..."<<std::endl;
                obj.save(file_in.c_str());

                sofa::simulation::getSimulation()->unload(groot);
            }
        }
    }
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::draw()
{
    /*    std::vector<  Vector3 > points;
        points.push_back(scaling_center);
        sofa::simulation::getSimulation()->DrawUtility.drawPoints(points, 20, Vec<4,float>(1,1,1,1));*/
}

} // namespace pim

} // namespace plugins

#endif
