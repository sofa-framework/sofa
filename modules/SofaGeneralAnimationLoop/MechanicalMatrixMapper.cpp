/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_ANIMATIONLOOP_MECHANICALMATRIXMAPPER_CPP
#include "MechanicalMatrixMapper.inl"
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;

////////////////////////////////////////////    FACTORY    //////////////////////////////////////////////
int MechanicalMatrixMapperClass = core::RegisterObject("This component allows to map the stiffness (and mass) matrix through a mapping.")

        .add< MechanicalMatrixMapper<Rigid3Types, Rigid3Types> >(true)
        .add< MechanicalMatrixMapper<Vec3Types, Rigid3Types> >(true)
        .add< MechanicalMatrixMapper<Vec3Types, Vec3Types> >(true)
        .add< MechanicalMatrixMapper<Vec1Types, Rigid3Types> >(true)
        .add< MechanicalMatrixMapper<Vec1Types, Vec1Types> >(true)
        .add< MechanicalMatrixMapper<Rigid3Types, Vec1Types> >(true)

        ;
////////////////////////////////////////////////////////////////////////////////////////////////////////

template class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalMatrixMapper<Rigid3Types, Rigid3Types>;
template class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalMatrixMapper<Vec3Types, Rigid3Types>;
template class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalMatrixMapper<Vec3Types, Vec3Types>;
template class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalMatrixMapper<Vec1Types, Rigid3Types>;
template class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalMatrixMapper<Vec1Types, Vec1Types>;
template class SOFA_GENERAL_ANIMATION_LOOP_API MechanicalMatrixMapper<Rigid3Types, Vec1Types> ;



} // namespace forcefield

} // namespace component

} // namespace sofa
