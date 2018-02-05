/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_CPP

#include <SofaGeneralDeformable/TriangularQuadraticSpringsForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <sofa/defaulttype/Vec3Types.h>



// #define DEBUG_TRIANGLEFEM

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(TriangularQuadraticSpringsForceField)

using namespace sofa::defaulttype;



// Register in the Factory
int TriangularQuadraticSpringsForceFieldClass = core::RegisterObject("Quadratic Springs on a Triangular Mesh")
#ifndef SOFA_FLOAT
        .add< TriangularQuadraticSpringsForceField<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TriangularQuadraticSpringsForceField<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_DEFORMABLE_API TriangularQuadraticSpringsForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_DEFORMABLE_API TriangularQuadraticSpringsForceField<Vec3fTypes>;
#endif



} // namespace forcefield

} // namespace component

} // namespace sofa

