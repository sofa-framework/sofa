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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_TYPEDEF_MAPPING_COMBINATION_H
#define SOFA_TYPEDEF_MAPPING_COMBINATION_H


#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/component/visualmodel/OglModel.h>


#include <sofa/component/mapping/ArticulatedSystemMapping.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/mapping/BeamLinearMapping.h>
#include <sofa/component/mapping/CenterOfMassMapping.h>
#include <sofa/component/mapping/CurveMapping.h>
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/component/mapping/ImplicitSurfaceMapping.h>
#include <sofa/component/mapping/LaparoscopicRigidMapping.h>
#include <sofa/component/mapping/LineSetSkinningMapping.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/mapping/RigidRigidMapping.h>
#include <sofa/component/mapping/SkinningMapping.h>
#include <sofa/component/mapping/SPHFluidSurfaceMapping.h>
#include <sofa/component/mapping/SubsetMapping.h>
#include <sofa/component/mapping/VoidMapping.h>


//ArticulatedSystemMapping
//---------------------

typedef sofa::component::mapping::ArticulatedSystemMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > ArticulatedSystemMapping1d_to_Rigid3f;

typedef sofa::component::mapping::ArticulatedSystemMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > ArticulatedSystemMapping1f_to_Rigid3d;





//BarycentricMapping
//---------------------
typedef sofa::component::mapping::BarycentricMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > BarycentricMechanicalMapping3d_to_3f;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > BarycentricMechanicalMapping3f_to_3d;


typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > BarycentricMapping3d_to_3f;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > BarycentricMapping3f_to_3d;




//BeamLinearMapping
//---------------------

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > >    BeamLinearMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > >    BeamLinearMechanicalMappingRigid3f_to_3d;



typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > >    BeamLinearMappingRigid3d_to_3f;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > >    BeamLinearMappingRigid3f_to_3d;






//CenterOfMassMapping
//---------------------
typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > CenterOfMassMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > CenterOfMassMechanicalMappingRigid3f_to_3d;



typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes> > > CenterOfMassMechanicalMappingRigid2d_to_2f;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes> > > CenterOfMassMechanicalMappingRigid2f_to_2d;




typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > CenterOfMassMappingRigid3d_to_3f;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > CenterOfMassMappingRigid3f_to_3d;




//CurveMapping
//---------------------
typedef sofa::component::mapping::CurveMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > CurveMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::CurveMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > CurveMechanicalMappingRigid3f_to_3d;


//IdentityMapping
//---------------------

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes> > > IdentityMechanicalMapping1d_to_1f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes> > > IdentityMechanicalMapping1f_to_1d;



typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes> > > IdentityMechanicalMapping2d_to_2f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes> > > IdentityMechanicalMapping2f_to_2d;




typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > IdentityMechanicalMapping3d_to_3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > IdentityMechanicalMapping3f_to_3d;




typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6fTypes> > > IdentityMechanicalMapping6d_to_6f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6dTypes> > > IdentityMechanicalMapping6f_to_6d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes> > > IdentityMechanicalMappingRigid2d_to_Rigid2f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes> > > IdentityMechanicalMappingRigid2f_to_Rigid2d;




typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > IdentityMechanicalMappingRigid3d_to_Rigid3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > IdentityMechanicalMappingRigid3f_to_Rigid3d;




typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > IdentityMapping3d_to_3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > IdentityMapping3f_to_3d;







typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid2fTypes> > > IdentityMappingRigid2d_to_Rigid2f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid2dTypes> > > IdentityMappingRigid2f_to_Rigid2d;




typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3fTypes> > > IdentityMappingRigid3d_to_Rigid3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3dTypes> > > IdentityMappingRigid3f_to_Rigid3d;



//ImplicitSurfaceMapping
//---------------------
typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes>  > ImplicitSurfaceMapping3d_to_3f;

typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes>  > ImplicitSurfaceMapping3f_to_3d;






//LaparoscopicRigidMapping
//---------------------


//LineSetSkinningMapping
//---------------------
typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > LineSetSkinningMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > LineSetSkinningMechanicalMappingRigid3f_to_3d;



typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > LineSetSkinningMappingRigid3d_to_3f;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > LineSetSkinningMappingRigid3f_to_3d;




//RigidMapping
//---------------------
typedef sofa::component::mapping::RigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > RigidMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::RigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > RigidMechanicalMappingRigid3f_to_3d;



typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > RigidMappingRigid3d_to_3f;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > RigidMappingRigid3f_to_3d;





//RigidRigidMapping
//---------------------
typedef sofa::component::mapping::RigidRigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > RigidRigidMechanicalMappingRigid3d_to_Rigid3f;

typedef sofa::component::mapping::RigidRigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > RigidRigidMechanicalMappingRigid3f_to_Rigid3d;




typedef sofa::component::mapping::RigidRigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3fTypes> > > RigidRigidMappingRigid3d_to_Rigid3f;

typedef sofa::component::mapping::RigidRigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3dTypes> > > RigidRigidMappingRigid3f_to_Rigid3d;


//SPHFluidSurfaceMapping
//---------------------

typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes>  > SPHFluidSurfaceMapping3d_to_3f;

typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes>  > SPHFluidSurfaceMapping3f_to_3d;






//SkinningMapping
//---------------------
typedef sofa::component::mapping::SkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > SkinningMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::SkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > SkinningMechanicalMappingRigid3f_to_3d;




typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > SkinningMappingRigid3d_to_3f;

typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > SkinningMappingRigid3f_to_3d;







//SubsetMapping
//---------------------

typedef sofa::component::mapping::SubsetMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > SubsetMechanicalMapping3d_to_3f;

typedef sofa::component::mapping::SubsetMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > SubsetMechanicalMapping3f_to_3d;




typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > SubsetMapping3d_to_3f;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > SubsetMapping3f_to_3d;

#endif
