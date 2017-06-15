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
#define FRAME_FRAMEDIAGONALMASS_CPP

#include "FrameDiagonalMass.inl"
#include <sofa/core/ObjectFactory.h>

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

template<> void FrameDiagonalMass<Affine3dTypes, Frame3x12dMass>::rotateMass() {}

template<> void FrameDiagonalMass<Affine3fTypes, Frame3x12fMass>::rotateMass() {}

#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameDiagonalMass<Affine3dTypes,Frame3x12dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API FrameDiagonalMass<Affine3fTypes,Frame3x12fMass>;
#endif

template<> void FrameDiagonalMass<Quadratic3dTypes, Frame3x30dMass>::rotateMass() {}

template<> void FrameDiagonalMass<Quadratic3fTypes, Frame3x30fMass>::rotateMass() {}

#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameDiagonalMass<Quadratic3dTypes,Frame3x30dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API FrameDiagonalMass<Quadratic3fTypes,Frame3x30fMass>;
#endif


} // namespace mass

} // namespace component

} // namespace sofa

