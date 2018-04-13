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
#include "MappedMatrixForceFieldAndMass.inl"
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
// Registering the component
// see: http://wiki.sofa-framework.org/wiki/ObjectFactory
// 1-SOFA_DECL_CLASS(componentName) : Set the class name of the component
// 2-RegisterObject("description") + .add<> : Register the component
// 3-.add<>(true) : Set default template
SOFA_DECL_CLASS(MappedMatrixForceFieldAndMass)

int MappedMatrixForceFieldAndMassClass = core::RegisterObject("Partially rigidify a mechanical object using a rigid mapping.")
#ifdef SOFA_WITH_FLOAT
        .add< MappedMatrixForceFieldAndMass<Vec3fTypes, Rigid3fTypes> >()
        .add< MappedMatrixForceFieldAndMass<Vec3fTypes, Vec3fTypes> >()
        .add< MappedMatrixForceFieldAndMass<Vec1fTypes, Rigid3fTypes> >()
        .add< MappedMatrixForceFieldAndMass<Vec1fTypes, Vec1fTypes> >()
#endif
#ifdef SOFA_WITH_DOUBLE
        .add< MappedMatrixForceFieldAndMass<Vec3dTypes, Rigid3dTypes> >()
        .add< MappedMatrixForceFieldAndMass<Vec3dTypes, Vec3dTypes> >()
        .add< MappedMatrixForceFieldAndMass<Vec1dTypes, Rigid3dTypes> >()
        .add< MappedMatrixForceFieldAndMass<Vec1dTypes, Vec1dTypes> >()
#endif
        ;
////////////////////////////////////////////////////////////////////////////////////////////////////////

// Force template specialization for the most common sofa floating point related type.
// This goes with the extern template declaration in the .h. Declaring extern template
// avoid the code generation of the template for each compilation unit.
// see: http://www.stroustrup.com/C++11FAQ.html#extern-templates
#ifdef SOFA_WITH_DOUBLE
template class MappedMatrixForceFieldAndMass<Vec3dTypes, Rigid3dTypes>;
template class MappedMatrixForceFieldAndMass<Vec3dTypes, Vec3dTypes>;
template class MappedMatrixForceFieldAndMass<Vec1dTypes, Rigid3dTypes>;
template class MappedMatrixForceFieldAndMass<Vec1dTypes, Vec1dTypes>;
#endif
#ifdef SOFA_WITH_FLOAT
template class MappedMatrixForceFieldAndMass<Vec3fTypes, Rigid3fTypes>;
template class MappedMatrixForceFieldAndMass<Vec3fTypes, Vec3fTypes>;
template class MappedMatrixForceFieldAndMass<Vec1fTypes, Rigid3fTypes>;
template class MappedMatrixForceFieldAndMass<Vec1fTypes, Vec1fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
