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
#ifndef FRAME_INIT_H
#define FRAME_INIT_H

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_FRAME
#define SOFA_FRAME_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_FRAME_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif //FRAME_INIT_H

/** \mainpage
The control nodes (master degrees of freedom, DOFs) of the simulation are moving frames, stored in a sofa MechanicalObject template class instanciated on the type of frame.
The master DOFs can be of different types:
- rigid frames, with 6 degrees of freedom (DOF) per frame, are implemented using standard sofa RigidTypes.
- affine deformable frames (12 DOF/node) are implemented in sofa::defaulttype::StdAffineTypes.
- quadraticallydeformable frames (30 DOF/node) are implemented in sofa::defaulttype::StdQuadraticTypes.

The MechanicalObject is attached to a scenegraph node (do not mix simulation nodes, like particles or frames, with scenegraph nodes, which define the structure of the SOFA scenegraph).
Deformable material attached to the control frames is stored in a child scenegraph node.
The slave DOFs can be of different types:
- standard sofa particles for moving points. This is used to attach a deformable mesh.
- deformation gradient, to represent the local deformation of a continuum, using class sofa::defaulttype::DeformationGradientTypes. There are two flavors of these:
    - DeformationGradient331dTypes, which correspond to standard Deformation Gradients (somehow redundant with affine deformable frames)
    - DeformationGradient332dTypes, which correspond to <A href="http://graphics.ethz.ch/~smartin/data/elastons.pdf"> Elastons </A>
- affine deformable frames

The connection between the control (master) frames and the deformable (slave) material is done by a sofa::component::mapping::FrameBlendingMapping.
This component is templated on two classes, the first corresponds to the type of master frames, and the second corresponds to the type of slave DOFs.
The actual blending is performed using auxiliary classes: sofa::defaulttype::LinearBlending for linear blending, or sofa::defaulttype::DualQuatBlending for dual quaternion blending.
The default implementations of these classes do nothing, all the implementation is in their specializations.

Variants of standard sofa classes adapted to moving frames have been developed:
- sofa::component::mass::FrameDiagonalMass  to store each frame's mass and inertia
- ConstantForceField (FrameConstantForceField.h)
- sofa::component::projectiveconstraintset::FrameFixedConstraint  to maintain frames  at the same place
- LinearMovementConstraint to impose translations (FrameLinearMovementConstraint.h)
- sofa::component::projectiveconstraintset::FrameRigidConstraint to rigidify a deformable master frame
- sofa::component::forcefield::FrameSpringForceField2

These are ForceFields to implement material constitutive laws:
- sofa::component::forcefield::FrameVolumePreservationForceField
- sofa::component::forcefield::GreenLagrangeForceField
- sofa::component::forcefield::CorotationalForceField
- sofa::component::forcefield::FrameHookeForceField

Materials implement the stress-strain constitutive law of the continuum. They are implemented in:
- sofa::component::material::HookeMaterial3 : a very simple uniform Hooke material
- sofa::component::material::GridMaterial : a material distribution represented by a voxel grid. This class also optimizes the distribution of the master DOFs, the shape functions and the integration point.

The source of this main page is in initFrame.h

  */
