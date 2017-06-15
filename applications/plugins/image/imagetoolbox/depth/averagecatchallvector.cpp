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
#define SOFA_IMAGE_AVERAGECATCHALLVECTOR_CPP

#include "averagecatchallvector.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(AverageCatchAllVector)

int AverageCatchAllVectorClass = core::RegisterObject("AverageCatchAllVector")
        .add<AverageCatchAllVector<float > >(true)
        //.add<AverageCatchAllVector<unsigned float> >()
        .add<AverageCatchAllVector<short > >()
        .add<AverageCatchAllVector<unsigned short > >()
        .add<AverageCatchAllVector<int > >()
        .add<AverageCatchAllVector<unsigned int > >()
        .add<AverageCatchAllVector<double > >()
        //.add<AverageCatchAllVector<unsigned double> >()
        .add<AverageCatchAllVector<long > >()
        .add<AverageCatchAllVector<unsigned long > >()
        .add<AverageCatchAllVector<bool > >()
        .add<AverageCatchAllVector<sofa::defaulttype::Vec3f> >()
        .add<AverageCatchAllVector<sofa::defaulttype::Vec3d> >()
        ;

template class SOFA_IMAGE_GUI_API AverageCatchAllVector<float >;
//template class SOFA_IMAGE_API AverageCatchAllVector<unsigned float >;
template class SOFA_IMAGE_GUI_API AverageCatchAllVector<short >;
template class SOFA_IMAGE_GUI_API AverageCatchAllVector<unsigned short >;
template class SOFA_IMAGE_GUI_API AverageCatchAllVector<int >;
template class SOFA_IMAGE_GUI_API AverageCatchAllVector<unsigned int >;
template class SOFA_IMAGE_GUI_API AverageCatchAllVector<double >;
//template class SOFA_IMAGE_API AverageCatchAllVector<unsigned double >;
template class SOFA_IMAGE_GUI_API AverageCatchAllVector<long >;
template class SOFA_IMAGE_GUI_API AverageCatchAllVector<unsigned long >;
template class SOFA_IMAGE_GUI_API AverageCatchAllVector<bool >;
template class SOFA_IMAGE_GUI_API AverageCatchAllVector<sofa::defaulttype::Vec3f >;
template class SOFA_IMAGE_GUI_API AverageCatchAllVector<sofa::defaulttype::Vec3d >;




} //
} // namespace component
} // namespace sofa

