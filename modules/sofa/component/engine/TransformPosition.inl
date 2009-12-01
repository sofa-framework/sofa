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
#ifndef SOFA_COMPONENT_ENGINE_TRANSFORMPOSITION_INL
#define SOFA_COMPONENT_ENGINE_TRANSFORMPOSITION_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/engine/TransformPosition.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/Quater.h>
#include <math.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::helper;
using namespace sofa::defaulttype;
using namespace core::objectmodel;

template <class DataTypes>
TransformPosition<DataTypes>::TransformPosition()
    : f_origin( initData(&f_origin, "origin", "a 3d point on the plane") )
    , f_inputX( initData (&f_inputX, "input_position", "input array of 3d points") )
    , f_outputX( initData (&f_outputX, "output_position", "output array of 3d points projected on a plane") )
    , f_normal(initData(&f_normal, "normal", "plane normal") )
    , f_translation(initData(&f_translation, "translation", "translation vector ") )
    , f_rotation(initData(&f_rotation, "rotation", "rotation vector ") )
    , f_scale(initData(&f_scale, (Real) 1.0, "scale", "scale factor") )
    , method(initData(&method, "method", "transformation method either translation or scale or rotation or projectOnPlane") )
{
}

template <class DataTypes>
void TransformPosition<DataTypes>::init()
{
    if (method=="projectOnPlane")
    {
        transformationMethod=PROJECT_ON_PLANE;
    }
    else if (method=="translation")
    {
        transformationMethod=TRANSLATION;
    }
    else if (method=="rotation")
    {
        transformationMethod=ROTATION;
    }
    else if (method=="scale")
    {
        transformationMethod=SCALE;
    }
    else if (method=="scaleTranslation")
    {
        transformationMethod=SCALE_TRANSLATION;
    }
    else if (method=="scaleRotationTranslation")
    {
        transformationMethod=SCALE_ROTATION_TRANSLATION;
    }
    else
    {
        transformationMethod=TRANSLATION;
        serr << "Error : Method " << method << "is unknown" <<sendl;
    }

    Coord& normal = *(f_normal.beginEdit());

    /// check if the normal is of norm 1
    if (fabs((normal.norm2()-1.0))>1e-10)
    {
        normal/=normal.norm();
    }

    f_normal.endEdit();

    addInput(&f_inputX);
    addOutput(&f_outputX);

    setDirtyValue();
}

template <class DataTypes>
void TransformPosition<DataTypes>::reinit()
{
    update();
}



template <class DataTypes>
void TransformPosition<DataTypes>::update()
{
    cleanDirty();

    helper::ReadAccessor< Data<VecCoord> > in = f_inputX;
    helper::WriteAccessor< Data<VecCoord> > out = f_outputX;
    helper::ReadAccessor< Data<Coord> > normal = f_normal;
    helper::ReadAccessor< Data<Coord> > origin = f_origin;
    helper::ReadAccessor< Data<Coord> > translation = f_translation;
    helper::ReadAccessor< Data<Real> > scale = f_scale;
    helper::ReadAccessor< Data<Coord> > rotation = f_rotation;

    out.resize(in.size());
    unsigned int i;
    switch(transformationMethod)
    {
    case PROJECT_ON_PLANE :
        for (i=0; i< in.size(); ++i)
        {
            out[i]=in[i]+normal.ref()*dot((origin.ref()-in[i]),normal.ref());
        }
        break;
    case TRANSLATION :
        for (i=0; i< in.size(); ++i)
        {
            out[i]=in[i]+translation.ref();
        }
        break;
    case SCALE :
        for (i=0; i< in.size(); ++i)
        {
            out[i]=in[i]*scale;
        }
        break;
    case SCALE_TRANSLATION :
        for (i=0; i< in.size(); ++i)
        {
            out[i]=in[i]*scale.ref() +translation.ref();
        }
        break;
    case ROTATION :
    {
        Quaternion q=helper::Quater<Real>::createQuaterFromEuler( rotation.ref()*M_PI/180.0);

        for (i=0; i< in.size(); ++i)
        {
            out[i]=q.rotate(in[i]);
        }
    }
    break;
    case SCALE_ROTATION_TRANSLATION :
    {
        Quaternion q=helper::Quater<Real>::createQuaterFromEuler( rotation.ref()*M_PI/180.0);

        for (i=0; i< in.size(); ++i)
        {
            out[i]=q.rotate(in[i]*scale) +translation.ref();
        }
        break;
    }
    }

}

template <class DataTypes>
void TransformPosition<DataTypes>::draw()
{

}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
