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
#ifndef SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_INL
#define SOFA_COMPONENT_MAPPING_LAPAROSCOPICRIGIDMAPPING_INL

#include <sofa/component/mapping/LaparoscopicRigidMapping.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/gl/template.h>
#include <string>


namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::init()
{
    this->BasicMapping::init();
}

template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize(1);
    out[0].getOrientation() = in[0].getOrientation(); // * rotation.getValue();
    out[0].getCenter() = pivot.getValue() + in[0].getOrientation().rotate(sofa::defaulttype::Vector3(0,0,in[0].getTranslation()));
    currentRotation = in[0].getOrientation();
}

template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize(1);
    out[0].getVOrientation() = in[0].getVOrientation(); //rotation * in[0].getVOrientation();
    out[0].getVCenter() = currentRotation.rotate(sofa::defaulttype::Vector3(0,0,in[0].getVTranslation()));
}

template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    out[0].getVOrientation() += in[0].getVOrientation(); //rotation * in[0].getVOrientation();
    out[0].getVTranslation() += dot(currentRotation.rotate(sofa::defaulttype::Vector3(0,0,1)), in[0].getVCenter());
}

template <class BasicMapping>
void LaparoscopicRigidMapping<BasicMapping>::draw()
{
    if (!this->getShow()) return;
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
