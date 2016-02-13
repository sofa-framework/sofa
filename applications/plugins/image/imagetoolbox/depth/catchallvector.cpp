/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_IMAGE_CATCHALLVECTOR_CPP

#include "catchallvector.h"
#include <sofa/core/ObjectFactory.h>
#include <image/ImageTypes.h>

namespace sofa
{
namespace component
{
namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(CatchAllVector)

int CatchAllVectorClass = core::RegisterObject("CatchAllVector")
        .add<CatchAllVector<float > >(true)
        //.add<CatchAllVector<unsigned float> >()
        .add<CatchAllVector<short > >()
        .add<CatchAllVector<unsigned short > >()
        .add<CatchAllVector<int > >()
        .add<CatchAllVector<unsigned int > >()
        .add<CatchAllVector<double > >()
        //.add<CatchAllVector<unsigned double> >()
        .add<CatchAllVector<long > >()
        .add<CatchAllVector<unsigned long > >()
        .add<CatchAllVector<bool > >()
        .add<CatchAllVector<sofa::defaulttype::Vec3f> >()
        .add<CatchAllVector<sofa::defaulttype::Vec3d> >()
        ;

template class SOFA_IMAGE_GUI_API CatchAllVector<float >;
//template class SOFA_IMAGE_GUI_API CatchAllVector<unsigned float >;
template class SOFA_IMAGE_GUI_API CatchAllVector<short >;
template class SOFA_IMAGE_GUI_API CatchAllVector<unsigned short >;
template class SOFA_IMAGE_GUI_API CatchAllVector<int >;
template class SOFA_IMAGE_GUI_API CatchAllVector<unsigned int >;
template class SOFA_IMAGE_GUI_API CatchAllVector<double >;
//template class SOFA_IMAGE_GUI_API CatchAllVector<unsigned double >;
template class SOFA_IMAGE_GUI_API CatchAllVector<long >;
template class SOFA_IMAGE_GUI_API CatchAllVector<unsigned long >;
template class SOFA_IMAGE_GUI_API CatchAllVector<bool >;
template class SOFA_IMAGE_GUI_API CatchAllVector<sofa::defaulttype::Vec3f >;
template class SOFA_IMAGE_GUI_API CatchAllVector<sofa::defaulttype::Vec3d >;




} //
} // namespace component
} // namespace sofa

