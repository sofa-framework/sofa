/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "NewHookeMaterial.inl"
#include <sofa/core/ObjectFactory.h>
#include "AffineTypes.h"
#include "QuadraticTypes.h"
#include "DeformationGradientTypes.h"


namespace sofa
{
namespace component
{
namespace material
{
using namespace sofa::defaulttype;

SOFA_DECL_CLASS (HookeMaterial3);
// Register in the Factory

int HookeMaterial3Class = core::RegisterObject ( "Hooke material for deformable objects" )
#ifndef SOFA_FLOAT
        .add<HookeMaterial3<Material3d> >()
#endif
#ifndef SOFA_DOUBLE
        .add<HookeMaterial3<Material3f> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_FRAME_API HookeMaterial3<Material3d>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API HookeMaterial3<Material3f>;
#endif



}

} // namespace component

} // namespace sofa

