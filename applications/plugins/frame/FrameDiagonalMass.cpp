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
#define FRAME_FRAMEDIAGONALMASS_CPP
#include "FrameDiagonalMass.inl"
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(FrameDiagonalMass)

// Register in the Factory
int FrameDiagonalMassClass = core::RegisterObject("Define a specific mass for each particle")
#ifndef SOFA_FLOAT
        .add< FrameDiagonalMass<Rigid3dTypes,Frame3x6dMass> >()
        .add< FrameDiagonalMass<Affine3dTypes,Frame3x12dMass> >()
        .add< FrameDiagonalMass<Quadratic3dTypes,Frame3x30dMass> >()
#endif
#ifndef SOFA_DOUBLE
        .add< FrameDiagonalMass<Rigid3fTypes,Frame3x6fMass> >()
        .add< FrameDiagonalMass<Affine3fTypes,Frame3x12fMass> >()
        .add< FrameDiagonalMass<Quadratic3fTypes,Frame3x30fMass> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameDiagonalMass<Rigid3dTypes,Frame3x6dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API FrameDiagonalMass<Rigid3fTypes,Frame3x6fMass>;
#endif


///////////////////////////////////////////////////////////////////////////////
//                     Affine3dTypes, Frame3x12dMass                         //
///////////////////////////////////////////////////////////////////////////////
template<>
void FrameDiagonalMass<Affine3dTypes, Frame3x12dMass>::updateMass()
{
}


template<>
void FrameDiagonalMass<Affine3dTypes, Frame3x12dMass>::computeRelRot ( Mat33& , const Coord& , const Coord& )
{
}

///////////////////////////////////////////////////////////////////////////////
//                     Affine3fTypes, Frame3x12fMass                         //
///////////////////////////////////////////////////////////////////////////////
template<>
void FrameDiagonalMass<Affine3fTypes, Frame3x12fMass>::updateMass()
{
}


template<>
void FrameDiagonalMass<Affine3fTypes, Frame3x12fMass>::computeRelRot ( Mat33& , const Coord& , const Coord& )
{
}


#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameDiagonalMass<Affine3dTypes,Frame3x12dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API FrameDiagonalMass<Affine3fTypes,Frame3x12fMass>;
#endif



///////////////////////////////////////////////////////////////////////////////
//                    Quadratic3dTypes, Frame3x30dMass                       //
///////////////////////////////////////////////////////////////////////////////

template<>
void FrameDiagonalMass<Quadratic3dTypes, Frame3x30dMass>::updateMass()
{
}

template<>
void FrameDiagonalMass<Quadratic3dTypes, Frame3x30dMass>::computeRelRot ( Mat33& , const Coord& , const Coord& )
{
}

///////////////////////////////////////////////////////////////////////////////
//                    Quadratic3fTypes, Frame3x30fMass                       //
///////////////////////////////////////////////////////////////////////////////

template<>
void FrameDiagonalMass<Quadratic3fTypes, Frame3x30fMass>::updateMass()
{
}


template<>
void FrameDiagonalMass<Quadratic3fTypes, Frame3x30fMass>::computeRelRot ( Mat33& , const Coord& , const Coord& )
{
}


#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameDiagonalMass<Quadratic3dTypes,Frame3x30dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API FrameDiagonalMass<Quadratic3fTypes,Frame3x30fMass>;
#endif


} // namespace mass

} // namespace component

} // namespace sofa

