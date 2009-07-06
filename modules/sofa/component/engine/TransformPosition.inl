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
    : originPtr( initDataPtr(&originPtr,&origin, "origin", "a 3d point on the plane") )
    , f_inputX( initData (&f_inputX, "input_position", "input array of 3d points") )
    , f_outputX( initData (&f_outputX, "output_position", "output array of 3d points projected on a plane") )
    , normalPtr(initDataPtr(&normalPtr,&normal, "normal", "plane normal") )
    , translationPtr(initDataPtr(&translationPtr,&translation, "translation", "translation vector ") )
    , rotationPtr(initDataPtr(&rotationPtr,&rotation, "rotation", "rotation vector ") )
    , scalePtr(initDataPtr(&scalePtr,&scale, "scale", "scale factor") )
    , method(initData(&method, "method", "transformation method either translation or scale or rotation or projectOnPlane") )
    ,scale ((Real) 1.0)
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
        std::cerr << "Error : Method " << method << "is unknown" <<std::endl;
    }

    addInput(&f_inputX);
    addOutput(&f_outputX);

    setDirty();
    /// check if the normal is of norm 1
    if (fabs((normal.norm2()-1.0))>1e-10)
    {
        normal/=normal.norm();
    }
}

template <class DataTypes>
void TransformPosition<DataTypes>::reinit()
{
    update();
}



template <class DataTypes>
void TransformPosition<DataTypes>::update()
{
    dirty = false;


    const helper::vector<Coord>& in = f_inputX.getValue();
    helper::vector<Coord>& out = *(f_outputX.beginEdit());

    out.resize(in.size());
    unsigned int i;
    switch(transformationMethod)
    {
    case PROJECT_ON_PLANE :
        for (i=0; i< in.size(); ++i)
        {
            out[i]=in[i]+normal*dot((origin-in[i]),normal);
        }
        break;
    case TRANSLATION :
        for (i=0; i< in.size(); ++i)
        {
            out[i]=in[i]+translation;
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
            out[i]=in[i]*scale +translation;
        }
        break;
    case ROTATION :
    {
        Quaternion q=helper::Quater<Real>::createQuaterFromEuler( rotation*M_PI/180.0);

        for (i=0; i< in.size(); ++i)
        {
            out[i]=q.rotate(in[i]);
        }
    }
    break;
    case SCALE_ROTATION_TRANSLATION :
    {
        Quaternion q=helper::Quater<Real>::createQuaterFromEuler( rotation*M_PI/180.0);

        for (i=0; i< in.size(); ++i)
        {
            out[i]=q.rotate(in[i]*scale) +translation;
        }
        break;
    }
    }

    f_outputX.endEdit();

}

template <class DataTypes>
void TransformPosition<DataTypes>::draw()
{

}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
