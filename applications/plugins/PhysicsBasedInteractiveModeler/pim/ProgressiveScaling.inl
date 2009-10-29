/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
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
#include <sofa/simulation/common/AnimateEndEvent.h>

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
    , from_scale(initData(&from_scale,  1.0, "from_scale", "initial scale applied to the degrees of freedom") )
    , to_scale(initData(&to_scale,  1.0, "to_scale", "final scale applied to the degrees of freedom") )
    , step(initData(&step, 0.1, "step", "progressive step for scaling. Range: 0 - 1") )
    , cm0(Vector3(0,0,0)), progressiveScale(1.0)

{
    f_listening.setValue(true);
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::init()
{
    progressiveScale = to_scale.getValue() - from_scale.getValue();
    const VecCoord& X0 = f_X0.getValue();
    VecCoord& X = *(f_X.beginEdit());
    X.resize(X0.size());
    f_X.endEdit();

    for (unsigned int i=0; i<X0.size(); ++i)
    {
        cm0 += Vector3(X0[i][0], X0[i][1], X0[i][2])*to_scale.getValue();
    }
    cm0 /= X0.size();

    local_X0.resize(X0.size());
    for (unsigned int i=0; i<X0.size(); ++i)
    {
        local_X0[i][0] = (X0[i][0]*to_scale.getValue()) - cm0.x();
        local_X0[i][1] = (X0[i][1]*to_scale.getValue()) - cm0.y();
        local_X0[i][2] = (X0[i][2]*to_scale.getValue()) - cm0.z();
    }
    addInput(&f_X0);
    addOutput(&f_X);
    setDirtyValue();
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::reset()
{
    progressiveScale = to_scale.getValue() - from_scale.getValue();
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::reinit()
{
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::update()
{
    cleanDirty();

    VecCoord& X = *(f_X.beginEdit());
    for( unsigned i=0; i<local_X0.size(); ++i )
    {
        Real x=0.0,y=0.0,z=0.0;
        DataTypes::get(x,y,z,local_X0[i]);
        Vector3 result = cm0 + (Vector3(x, y, z)*progressiveScale);
        DataTypes::set(X[i], result.x(), result.y(), result.z());
    }

    f_X.endEdit();
}

template <class DataTypes>
void ProgressiveScaling<DataTypes>::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateEndEvent *>(event) && progressiveScale < 1.0)
    {
        progressiveScale += step.getValue();
        setDirtyValue();
    }
}

} // namespace pim

} // namespace plugins

#endif
