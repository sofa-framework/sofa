/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "MechanicalMatrixMapper.inl"
#include <sofa/defaulttype/Vec3Types.h>
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
SOFA_DECL_CLASS(MechanicalMatrixMapper)

int MechanicalMatrixMapperClass = core::RegisterObject("Partially rigidify a mechanical object using a rigid mapping.")
#ifdef SOFA_WITH_FLOAT
        .add< MechanicalMatrixMapper<Vec3fTypes, Rigid3fTypes> >()
        .add< MechanicalMatrixMapper<Vec3fTypes, Vec3fTypes> >()
        .add< MechanicalMatrixMapper<Vec1fTypes, Rigid3fTypes> >()
        .add< MechanicalMatrixMapper<Vec1fTypes, Vec1fTypes> >()
#endif
#ifdef SOFA_WITH_DOUBLE
        .add< MechanicalMatrixMapper<Vec3dTypes, Rigid3dTypes> >(true)
        .add< MechanicalMatrixMapper<Vec3dTypes, Vec3dTypes> >(true)
        .add< MechanicalMatrixMapper<Vec1dTypes, Rigid3dTypes> >(true)
        .add< MechanicalMatrixMapper<Vec1dTypes, Vec1dTypes> >(true)
#endif
        ;
////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef SOFA_WITH_DOUBLE
template class MechanicalMatrixMapper<Vec3dTypes, Rigid3dTypes>;
template class MechanicalMatrixMapper<Vec3dTypes, Vec3dTypes>;
template class MechanicalMatrixMapper<Vec1dTypes, Rigid3dTypes>;
template class MechanicalMatrixMapper<Vec1dTypes, Vec1dTypes>;
#endif
#ifdef SOFA_WITH_FLOAT
template class MechanicalMatrixMapper<Vec3fTypes, Rigid3fTypes>;
template class MechanicalMatrixMapper<Vec3fTypes, Vec3fTypes>;
template class MechanicalMatrixMapper<Vec1fTypes, Rigid3fTypes>;
template class MechanicalMatrixMapper<Vec1fTypes, Vec1fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
