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
#define SOFA_COMPONENT_MASS_MESHMATRIXMASS_CPP
#include <sofa/component/mass/MeshMatrixMass.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MeshMatrixMass)

// Register in the Factory
int MeshMatrixMassClass = core::RegisterObject("Define a specific mass for each particle")
#ifndef SOFA_FLOAT
        .add< MeshMatrixMass<Vec3dTypes,double> >()
        .add< MeshMatrixMass<Vec2dTypes,double> >()
        .add< MeshMatrixMass<Vec1dTypes,double> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MeshMatrixMass<Vec3fTypes,float> >()
        .add< MeshMatrixMass<Vec2fTypes,float> >()
        .add< MeshMatrixMass<Vec1fTypes,float> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec3dTypes,double>;
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec2dTypes,double>;
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec1dTypes,double>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec3fTypes,float>;
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec2fTypes,float>;
template class SOFA_MISC_FORCEFIELD_API MeshMatrixMass<Vec1fTypes,float>;
#endif


} // namespace mass

} // namespace component

} // namespace sofa

