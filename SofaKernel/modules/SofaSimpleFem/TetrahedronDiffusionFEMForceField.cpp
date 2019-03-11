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
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONDIFFUSIONFEMFORCEFIELD_CPP

#include "TetrahedronDiffusionFEMForceField.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


// Register in the Factory
int TetrahedronDiffusionFEMForceFieldClass = core::RegisterObject("Isotropic or anisotropic diffusion on Tetrahedral Meshes")
  .add< TetrahedronDiffusionFEMForceField<Vec1Types> >()
  .add< TetrahedronDiffusionFEMForceField<Vec2Types> >(true)
  .add< TetrahedronDiffusionFEMForceField<Vec3Types> >()

;

  template class SOFA_SIMPLE_FEM_API TetrahedronDiffusionFEMForceField<Vec1Types>;
  template class SOFA_SIMPLE_FEM_API TetrahedronDiffusionFEMForceField<Vec2Types>;
  template class SOFA_SIMPLE_FEM_API TetrahedronDiffusionFEMForceField<Vec3Types>;




} // namespace forcefield

} // namespace component

} // namespace sofa

