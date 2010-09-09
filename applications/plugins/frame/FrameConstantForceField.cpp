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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define FRAME_FRAMECONSTANTFROCEFIELD_CPP

#include "AffineTypes.h"
#include "QuadraticTypes.h"
#include "FrameConstantForceField.h"
#include <sofa/component/forcefield/ConstantForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(FrameConstantForceField)

int FrameConstantForceFieldClass = core::RegisterObject("Attach given particles to their initial positions")
#ifndef SOFA_FLOAT
        .add< ConstantForceField<Affine3dTypes> >()
        .add< ConstantForceField<Quadratic3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ConstantForceField<Affine3fTypes> >()
        .add< ConstantForceField<Quadratic3fTypes> >()
#endif
        ;


#ifndef SOFA_FLOAT
template <>
double ConstantForceField<Affine3dTypes>::getPotentialEnergy(const VecCoord& ) const
{
    serr<<"ConstantForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}
template <>
double ConstantForceField<Quadratic3dTypes>::getPotentialEnergy(const VecCoord& ) const
{
    serr<<"ConstantForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}
#endif

#ifndef SOFA_DOUBLE
template <>
double ConstantForceField<Affine3fTypes>::getPotentialEnergy(const VecCoord& ) const
{
    serr<<"ConstantForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}

template <>
double ConstantForceField<Quadratic3fTypes>::getPotentialEnergy(const VecCoord& ) const
{
    serr<<"ConstantForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}
#endif



template class SOFA_FRAME_API ConstantForceField<Affine3dTypes>;

template class SOFA_FRAME_API ConstantForceField<Affine3fTypes>;

template class SOFA_FRAME_API ConstantForceField<Quadratic3dTypes>;

template class SOFA_FRAME_API ConstantForceField<Quadratic3fTypes>;

} // namespace forcefield

} // namespace component

} // namespace sofa
