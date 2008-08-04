/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_TYPEDEF_MAPPING_FLOAT_H
#define SOFA_TYPEDEF_MAPPING_FLOAT_H


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
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > ArticulatedSystemMapping1f_to_Rigid3f;



//BarycentricMapping
//---------------------
typedef sofa::component::mapping::BarycentricMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > BarycentricMechanicalMapping3f_to_3f;


typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > BarycentricMapping3f_to_3f;


typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > BarycentricMapping3f_to_Ext3;


//BeamLinearMapping
//---------------------
typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > >    BeamLinearMechanicalMappingRigid3f_to_3f;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > >    BeamLinearMappingRigid3f_to_3f;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > BeamLinearMapping3f_to_Ext3;




//CenterOfMassMapping
//---------------------
typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > CenterOfMassMechanicalMappingRigid3f_to_3f;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes> > > CenterOfMassMechanicalMappingRigid2f_to_2f;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > CenterOfMassMappingRigid3f_to_3f;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > CenterOfMassMappingRigid3f_to_Ext3;


//CurveMapping
//---------------------
typedef sofa::component::mapping::CurveMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > CurveMechanicalMappingRigid3f_to_3f;

//IdentityMapping
//---------------------
typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes> > > IdentityMechanicalMapping1f_to_1f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes> > > IdentityMechanicalMapping2f_to_2f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > IdentityMechanicalMapping3f_to_3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6fTypes> > > IdentityMechanicalMapping6f_to_6f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes> > > IdentityMechanicalMappingRigid2f_to_Rigid2f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > IdentityMechanicalMappingRigid3f_to_Rigid3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > IdentityMapping3f_to_3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > IdentityMapping3f_to_Ext3;


typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid2fTypes> > > IdentityMappingRigid2f_to_Rigid2f;


typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3fTypes> > > IdentityMappingRigid3f_to_Rigid3f;


//ImplicitSurfaceMapping
//---------------------
typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes>  > ImplicitSurfaceMapping3f_to_3f;


typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes>  > ImplicitSurfaceMapping3f_to_Ext3;



//LaparoscopicRigidMapping
//---------------------


//LineSetSkinningMapping
//---------------------
typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > LineSetSkinningMechanicalMappingRigid3f_to_3f;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > LineSetSkinningMappingRigid3f_to_3f;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > LineSetSkinningMappingRigid3f_to_Ext3;

//RigidMapping
//---------------------
typedef sofa::component::mapping::RigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > RigidMechanicalMappingRigid3f_to_3f;

typedef sofa::component::mapping::RigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes> > > RigidMechanicalMappingRigid2f_to_2f;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > RigidMappingRigid3f_to_3f;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > RigidMappingRigid3f_to_Ext3;

//RigidRigidMapping
//---------------------
typedef sofa::component::mapping::RigidRigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > RigidRigidMechanicalMappingRigid3f_to_Rigid3f;


typedef sofa::component::mapping::RigidRigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3fTypes> > > RigidRigidMappingRigid3f_to_Rigid3f;

//SPHFluidSurfaceMapping
//---------------------
typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes>  > SPHFluidSurfaceMapping3f_to_3f;

typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes>  > SPHFluidSurfaceMapping3f_to_Ext3;

//SkinningMapping
//---------------------
typedef sofa::component::mapping::SkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > SkinningMechanicalMappingRigid3f_to_3f;


typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > SkinningMappingRigid3f_to_3f;


typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > SkinningMappingRigid3f_to_Ext3;




//SubsetMapping
//---------------------
typedef sofa::component::mapping::SubsetMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > SubsetMechanicalMapping3f_to_3f;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > SubsetMapping3f_to_3f;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > SubsetMapping3f_to_Ext3;





#ifdef SOFA_FLOAT
typedef        ArticulatedSystemMapping1f_to_Rigid3f                   ArticulatedSystemMapping1f_to_Rigid3;
typedef        BarycentricMechanicalMapping3f_to_3f		       BarycentricMechanicalMapping3_to_3;
typedef        BarycentricMapping3f_to_3f			       BarycentricMapping3_to_3;
typedef        BarycentricMapping3f_to_Ext3			       BarycentricMapping3_to_Ext3;
typedef        BeamLinearMechanicalMappingRigid3f_to_3f	               BeamLinearMechanicalMappingRigid3_to_3;
typedef        BeamLinearMappingRigid3f_to_3f			       BeamLinearMappingRigid3_to_3;
typedef        BeamLinearMapping3f_to_Ext3			       BeamLinearMapping3_to_Ext3;
typedef        CenterOfMassMechanicalMappingRigid3f_to_3f	       CenterOfMassMechanicalMappingRigid3_to_3;
typedef        CenterOfMassMechanicalMappingRigid2f_to_2f	       CenterOfMassMechanicalMappingRigid2_to_2;
typedef        CenterOfMassMappingRigid3f_to_3f	        	       CenterOfMassMappingRigid3_to_3;
typedef        CenterOfMassMappingRigid3f_to_Ext3		       CenterOfMassMappingRigid3_to_Ext3;
typedef        CurveMechanicalMappingRigid3f_to_3f		       CurveMechanicalMappingRigid3_to_3;
typedef        IdentityMechanicalMapping1f_to_1f		       IdentityMechanicalMapping1f_to_1f;
typedef        IdentityMechanicalMapping2f_to_2f		       IdentityMechanicalMapping2_to_2;
typedef        IdentityMechanicalMapping3f_to_3f		       IdentityMechanicalMapping3_to_3;
typedef        IdentityMechanicalMapping6f_to_6f		       IdentityMechanicalMapping6_to_6;
typedef        IdentityMechanicalMappingRigid2f_to_Rigid2f	       IdentityMechanicalMappingRigid2_to_Rigid2;
typedef        IdentityMechanicalMappingRigid3f_to_Rigid3f	       IdentityMechanicalMappingRigid3_to_Rigid3;
typedef        IdentityMapping3f_to_3f				       IdentityMapping3_to_3;
typedef        IdentityMapping3f_to_Ext3			       IdentityMapping3_to_Ext3;
typedef        IdentityMappingRigid2f_to_Rigid2f		       IdentityMappingRigid2_to_Rigid2;
typedef        IdentityMappingRigid3f_to_Rigid3f		       IdentityMappingRigid3_to_Rigid3;
typedef        ImplicitSurfaceMapping3f_to_3f			       ImplicitSurfaceMapping3_to_3;
typedef        ImplicitSurfaceMapping3f_to_Ext3		               ImplicitSurfaceMapping3_to_Ext3;
typedef        LineSetSkinningMechanicalMappingRigid3f_to_3f	       LineSetSkinningMechanicalMappingRigid3_to_3;
typedef        LineSetSkinningMappingRigid3f_to_3f		       LineSetSkinningMappingRigid3_to_3;
typedef        LineSetSkinningMappingRigid3f_to_Ext3		       LineSetSkinningMappingRigid3_to_Ext3;
typedef        RigidMechanicalMappingRigid3f_to_3f		       RigidMechanicalMappingRigid3_to_3;
typedef        RigidMechanicalMappingRigid2f_to_2f		       RigidMechanicalMappingRigid2_to_2;
typedef        RigidMappingRigid3f_to_3f			       RigidMappingRigid3_to_3;
typedef        RigidMappingRigid3f_to_Ext3			       RigidMappingRigid3_to_Ext3;
typedef        RigidRigidMechanicalMappingRigid3f_to_Rigid3f	       RigidRigidMechanicalMappingRigid3_to_Rigid3;
typedef        RigidRigidMappingRigid3f_to_Rigid3f		       RigidRigidMappingRigid3_to_Rigid3;
typedef        SPHFluidSurfaceMapping3f_to_3f			       SPHFluidSurfaceMapping3_to_3;
typedef        SPHFluidSurfaceMapping3f_to_Ext3		               SPHFluidSurfaceMapping3_to_Ext3;
typedef        SkinningMechanicalMappingRigid3f_to_3f		       SkinningMechanicalMappingRigid3_to_3;
typedef        SkinningMappingRigid3f_to_3f			       SkinningMappingRigid3_to_3;
typedef        SkinningMappingRigid3f_to_Ext3			       SkinningMappingRigid3_to_Ext3;
typedef        SubsetMechanicalMapping3f_to_3f			       SubsetMechanicalMapping3_to_3;
typedef        SubsetMapping3f_to_3f				       SubsetMapping3_to_3;
typedef        SubsetMapping3f_to_Ext3				       SubsetMapping3_to_Ext3;
#endif

#endif
