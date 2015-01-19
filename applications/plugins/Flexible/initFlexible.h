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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef INITFlexible_H
#define INITFlexible_H

#include <Flexible/Flexible.h>

/** \mainpage
   Flexible plugin provides a unified approach to the simulation of deformable solids. It uses a three-level kinematic structure:
  control nodes, deformation gradients,  strain measures, and new mappings between these. 
  * \image html threeLevels.png 
  * This approach maximizes the modularity of the implementation.
  * \section section1 The Flexible plugin provides: 
  * \li \b The \b control \b nodes carry the independent degrees of freedom of the object in two state vectors one for positions and one for velocities.
  * \n The control nodes are: finite elements nodes, moving frames(sofa#defaulttype#DeformableFrameMass), deformation modes.
  * \li \b The \b shape \b functions represent how the material space of the object is mapped to the world space, based on the DOF.
  * \n The shape functions are: shepard(sofa#component#shapefunction#ShepardShapeFunction), barycentric(sofa#component#shapefunction#BarycentricShapeFunction), natural neighbors(sofa#component#shapefunction#VoronoiShapeFunction).
  * \li \b The \b deformation \b gradients represent the local state of the continuum. Their basis vectors are orthonormal in the undeformed configuration,
  * while departure from unity corresponds to compression or extension, and departure from orthogonality corresponds to shear.
  * \n The deformation gradient F are defined with the notation: "Type(F) SpatialDimension MaterialDimension Order" in most cases F331 or F332 (elaston).
  * \li \b The \b strain \b measures correspond to one deformation gradient at the upper level.
  *\n The strain mapping are: Green-Lagrangian strain (sofa#component#mapping#GreenStrainMapping), Corotational Strain (sofa#component#mapping#CorotationalStrainMapping), Principal stretches (sofa#component#mapping#PrincipalStretchesMapping), 
  * Plastic Strain (sofa#component#mapping#PlasticStrainMapping), Invariants of deformation tensor(sofa#component#mapping#InvariantMapping).
  * \li \b The \b constitutive \b law of the object material computes stress based on strain.
  * \n The different constitutive laws are: Hooke force field(sofa#component#forcefield#HookeForceField), Mooney Rivlin force field(sofa#component#forcefield#MooneyRivlinForceField), Volume preservation force field (sofa#component#forcefield#VolumePreservationForceField).
  *\section section2 Flexible plugin contains a test suite FlexibleTest. 
  *\li This tests are tests for flexible components like strain mapping (sofa#StrainMappingTest) or higher level tests like patch test (sofa#AffinePatch_test and sofa#Patch_test)
  *
  * @author Benjamin Gilles, Francois Faure, Matthieu Nesme
  */
    
#endif // INITFlexible_H
