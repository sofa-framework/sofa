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
#ifndef SOFA_COMPONENT_ENGINE_TRANSFORMENGINE_INL
#define SOFA_COMPONENT_ENGINE_TRANSFORMENGINE_INL

#include <sofa/component/engine/TransformEngine.h>
#include <sofa/helper/rmath.h> //M_PI

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
TransformEngine<DataTypes>::TransformEngine()
    : f_inputX ( initData (&f_inputX, "input_position", "input array of 3d points") )
    , f_outputX( initData (&f_outputX, "output_position", "output array of 3d points projected on a plane") )
    , translation(initData(&translation, defaulttype::Vector3(0,0,0),"translation", "translation vector ") )
    , rotation(initData(&rotation, defaulttype::Vector3(0,0,0), "rotation", "rotation vector ") )
    , scale(initData(&scale, defaulttype::Vector3(1,1,1),"scale", "scale factor") )
{
}


template <class DataTypes>
void TransformEngine<DataTypes>::init()
{
    addInput(&f_inputX);
    addOutput(&f_outputX);
    setDirtyValue();
}

template <class DataTypes>
void TransformEngine<DataTypes>::reinit()
{
    update();
}


template <class DataTypes>
void TransformEngine<DataTypes>::applyScale(Coord &p, const Real sx,const Real sy,const Real sz) const
{
    Real x,y,z;
    DataTypes::get(x,y,z,p);
    DataTypes::set(p,x*sx,y*sy,z*sz);
}


#ifndef SOFA_FLOAT
template<>
void TransformEngine<defaulttype::Rigid3dTypes>::applyRotation (Coord &p, const defaulttype::Quat q) const;

#endif
#ifndef SOFA_DOUBLE
template<>
void TransformEngine<defaulttype::Rigid3fTypes>::applyRotation (Coord &p, const defaulttype::Quat q) const;
#endif


template <class DataTypes>
void TransformEngine<DataTypes>::applyRotation(Coord &p, const defaulttype::Quat q) const
{
    defaulttype::Vector3 pos;
    DataTypes::get(pos[0],pos[1],pos[2],p);
    pos=q.rotate(pos);
    DataTypes::set(p,pos[0],pos[1],pos[2]);
}

template <class DataTypes>
void TransformEngine<DataTypes>::applyTranslation(Coord &p, const Real tx,const Real ty,const Real tz) const
{
    Real x,y,z;
    DataTypes::get(x,y,z,p);
    DataTypes::set(p,x+tx,y+ty,z+tz);
}



template <class DataTypes>
void TransformEngine<DataTypes>::update()
{
    cleanDirty();

    const VecCoord& in = f_inputX.getValue();
    VecCoord& out = *(f_outputX.beginEdit());
    out.resize(in.size());

    const defaulttype::Vector3 &s=scale.getValue();
    const defaulttype::Vector3 &r=rotation.getValue();
    const defaulttype::Vector3 &t=translation.getValue();

    const defaulttype::Quaternion q=helper::Quater<Real>::createQuaterFromEuler( r*(M_PI/180.0));
    for (unsigned int i=0; i< in.size(); ++i)
    {
        out[i] = in[i];
        if (s != defaulttype::Vector3(1,1,1))
        {
            applyScale(out[i], s[0],s[1],s[2]);
        }
        if (r != defaulttype::Vector3(0,0,0))
        {
            applyRotation(out[i], q);
        }
        if (t != defaulttype::Vector3(0,0,0))
        {
            applyTranslation( out[i], t[0],t[1],t[2]);
        }
    }


    translation.setValue(defaulttype::Vector3(0,0,0));
    rotation.setValue(defaulttype::Vector3(0,0,0));
    scale.setValue(defaulttype::Vector3(1,1,1));

    f_outputX.endEdit();
}



} // namespace engine

} // namespace component

} // namespace sofa

#endif
