#ifndef SOFA_TYPEDEF_MAPPING_H
#define SOFA_TYPEDEF_MAPPING_H


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
#include <sofa/component/mapping/SurfaceIdentityMapping.h>
#include <sofa/component/mapping/VoidMapping.h>


//ArticulatedSystemMapping
//---------------------
typedef sofa::component::mapping::ArticulatedSystemMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > ArticulatedSystemMapping1d_to_Rigid3d;

typedef sofa::component::mapping::ArticulatedSystemMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > ArticulatedSystemMapping1d_to_Rigid3f;

typedef sofa::component::mapping::ArticulatedSystemMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > ArticulatedSystemMapping1f_to_Rigid3d;

typedef sofa::component::mapping::ArticulatedSystemMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > ArticulatedSystemMapping1f_to_Rigid3f;




//BarycentricMapping
//---------------------
typedef sofa::component::mapping::BarycentricMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > BarycentricMechanicalMapping3d_to_3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > BarycentricMechanicalMapping3d_to_3f;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > BarycentricMechanicalMapping3f_to_3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > BarycentricMechanicalMapping3f_to_3f;


typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > BarycentricMapping3d_to_3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > BarycentricMapping3d_to_3f;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > BarycentricMapping3f_to_3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > BarycentricMapping3f_to_3f;



typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > BarycentricMapping3d_to_Ext3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > BarycentricMapping3d_to_Ext3f;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > BarycentricMapping3f_to_Ext3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > BarycentricMapping3f_to_Ext3f;


//BeamLinearMapping
//---------------------
typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > >    BeamLinearMechanicalMappingRigid3d_to_3d;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > >    BeamLinearMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > >    BeamLinearMechanicalMappingRigid3f_to_3d;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > >    BeamLinearMechanicalMappingRigid3f_to_3f;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > >    BeamLinearMappingRigid3d_to_3d;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > >    BeamLinearMappingRigid3d_to_3f;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > >    BeamLinearMappingRigid3f_to_3d;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > >    BeamLinearMappingRigid3f_to_3f;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > BeamLinearMappingRigid3d_to_Ext3d;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > BeamLinearMappingRigid3d_to_Ext3f;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > BeamLinearMappingRigid3f_to_Ext3d;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > BeamLinearMapping3f_to_Ext3f;




//CenterOfMassMapping
//---------------------
typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > CenterOfMassMechanicalMappingRigid3d_to_3d;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > CenterOfMassMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > CenterOfMassMechanicalMappingRigid3f_to_3d;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > CenterOfMassMechanicalMappingRigid3f_to_3f;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes> > > CenterOfMassMechanicalMappingRigid2d_to_2d;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes> > > CenterOfMassMechanicalMappingRigid2d_to_2f;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes> > > CenterOfMassMechanicalMappingRigid2f_to_2d;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes> > > CenterOfMassMechanicalMappingRigid2f_to_2f;


typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > CenterOfMassMappingRigid3d_to_3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > CenterOfMassMappingRigid3d_to_3f;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > CenterOfMassMappingRigid3f_to_3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > CenterOfMassMappingRigid3f_to_3f;


typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > CenterOfMassMappingRigid3d_to_Ext3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > CenterOfMassMappingRigid3d_to_Ext3f;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > CenterOfMassMappingRigid3f_to_Ext3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > CenterOfMassMappingRigid3f_to_Ext3f;


//CurveMapping
//---------------------
typedef sofa::component::mapping::CurveMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > CurveMechanicalMappingRigid3d_to_3d;

typedef sofa::component::mapping::CurveMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > CurveMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::CurveMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > CurveMechanicalMappingRigid3f_to_3d;

typedef sofa::component::mapping::CurveMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > CurveMechanicalMappingRigid3f_to_3f;

//IdentityMapping
//---------------------
typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes> > > IdentityMechanicalMapping1d_to_1d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes> > > IdentityMechanicalMapping1d_to_1f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes> > > IdentityMechanicalMapping1f_to_1d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1fTypes> > > IdentityMechanicalMapping1f_to_1f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes> > > IdentityMechanicalMapping2d_to_2d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes> > > IdentityMechanicalMapping2d_to_2f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes> > > IdentityMechanicalMapping2f_to_2d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2fTypes> > > IdentityMechanicalMapping2f_to_2f;


typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > IdentityMechanicalMapping3d_to_3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > IdentityMechanicalMapping3d_to_3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > IdentityMechanicalMapping3f_to_3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > IdentityMechanicalMapping3f_to_3f;


typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6dTypes> > > IdentityMechanicalMapping6d_to_6d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6fTypes> > > IdentityMechanicalMapping6d_to_6f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6dTypes> > > IdentityMechanicalMapping6f_to_6d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6fTypes> > > IdentityMechanicalMapping6f_to_6f;


typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes> > > IdentityMechanicalMappingRigid2d_to_Rigid2d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes> > > IdentityMechanicalMappingRigid2d_to_Rigid2f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes> > > IdentityMechanicalMappingRigid2f_to_Rigid2d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2fTypes> > > IdentityMechanicalMappingRigid2f_to_Rigid2f;


typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > IdentityMechanicalMappingRigid3d_to_Rigid3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > IdentityMechanicalMappingRigid3d_to_Rigid3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > IdentityMechanicalMappingRigid3f_to_Rigid3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > IdentityMechanicalMappingRigid3f_to_Rigid3f;




typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > IdentityMapping3d_to_3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > IdentityMapping3d_to_3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > IdentityMapping3f_to_3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > IdentityMapping3f_to_3f;



typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > IdentityMapping3d_to_Ext3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > IdentityMapping3d_to_Ext3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > IdentityMapping3f_to_Ext3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > IdentityMapping3f_to_Ext3f;




typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid2dTypes> > > IdentityMappingRigid2d_to_Rigid2d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid2fTypes> > > IdentityMappingRigid2d_to_Rigid2f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid2dTypes> > > IdentityMappingRigid2f_to_Rigid2d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid2fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid2fTypes> > > IdentityMappingRigid2f_to_Rigid2f;


typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3dTypes> > > IdentityMappingRigid3d_to_Rigid3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3fTypes> > > IdentityMappingRigid3d_to_Rigid3f;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3dTypes> > > IdentityMappingRigid3f_to_Rigid3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3fTypes> > > IdentityMappingRigid3f_to_Rigid3f;


//ImplicitSurfaceMapping
//---------------------
typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes>  > ImplicitSurfaceMapping3d_to_3d;

typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes>  > ImplicitSurfaceMapping3d_to_3f;

typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes>  > ImplicitSurfaceMapping3f_to_3d;

typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes>  > ImplicitSurfaceMapping3f_to_3f;


typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes>  > ImplicitSurfaceMapping3d_to_Ext3d;

typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes>  > ImplicitSurfaceMapping3d_to_Ext3f;

typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes>  > ImplicitSurfaceMapping3f_to_Ext3d;

typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes>  > ImplicitSurfaceMapping3f_to_Ext3f;



//LaparoscopicRigidMapping
//---------------------

typedef sofa::component::mapping::LaparoscopicRigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::LaparoscopicRigid3Types>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3dTypes> > > LaparoscopicRigidMapping;

typedef sofa::component::mapping::LaparoscopicRigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::LaparoscopicRigid3Types>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3dTypes> > > LaparoscopicRigidMapping;


//LineSetSkinningMapping
//---------------------

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > LineSetSkinningMechanicalMappingRigid3d_to_3d;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > LineSetSkinningMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > LineSetSkinningMechanicalMappingRigid3f_to_3d;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > LineSetSkinningMechanicalMappingRigid3f_to_3f;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > LineSetSkinningMappingRigid3d_to_3d;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > LineSetSkinningMappingRigid3d_to_3f;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > LineSetSkinningMappingRigid3f_to_3d;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > LineSetSkinningMappingRigid3f_to_3f;


typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > LineSetSkinningMappingRigid3d_to_Ext3d;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > LineSetSkinningMappingRigid3d_to_Ext3f;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > LineSetSkinningMappingRigid3f_to_Ext3d;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > LineSetSkinningMappingRigid3f_to_Ext3f;

//RigidMapping
//---------------------
typedef sofa::component::mapping::RigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > RigidMechanicalMappingRigid3d_to_3d;

typedef sofa::component::mapping::RigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > RigidMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::RigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > RigidMechanicalMappingRigid3f_to_3d;

typedef sofa::component::mapping::RigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > RigidMechanicalMappingRigid3f_to_3f;


typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > RigidMappingRigid3d_to_3d;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > RigidMappingRigid3d_to_3f;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > RigidMappingRigid3f_to_3d;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > RigidMappingRigid3f_to_3f;



typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > RigidMappingRigid3d_to_Ext3d;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > RigidMappingRigid3d_to_Ext3f;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > RigidMappingRigid3f_to_Ext3d;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > RigidMappingRigid3f_to_Ext3f;

//RigidRigidMapping
//---------------------
typedef sofa::component::mapping::RigidRigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > RigidRigidMechanicalMappingRigid3d_to_Rigid3d;

typedef sofa::component::mapping::RigidRigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > RigidRigidMechanicalMappingRigid3d_to_Rigid3f;

typedef sofa::component::mapping::RigidRigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > RigidRigidMechanicalMappingRigid3f_to_Rigid3d;

typedef sofa::component::mapping::RigidRigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes> > > RigidRigidMechanicalMappingRigid3f_to_Rigid3f;


typedef sofa::component::mapping::RigidRigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3dTypes> > > RigidRigidMappingRigid3d_to_Rigid3d;

typedef sofa::component::mapping::RigidRigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3fTypes> > > RigidRigidMappingRigid3d_to_Rigid3f;

typedef sofa::component::mapping::RigidRigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3dTypes> > > RigidRigidMappingRigid3f_to_Rigid3d;

typedef sofa::component::mapping::RigidRigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3fTypes> > > RigidRigidMappingRigid3f_to_Rigid3f;

//SPHFluidSurfaceMapping
//---------------------
typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes>  > SPHFluidSurfaceMapping3d_to_3d;

typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes>  > SPHFluidSurfaceMapping3d_to_3f;

typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes>  > SPHFluidSurfaceMapping3f_to_3d;

typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes>  > SPHFluidSurfaceMapping3f_to_3f;


typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes>  > SPHFluidSurfaceMapping3d_to_Ext3d;

typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes>  > SPHFluidSurfaceMapping3d_to_Ext3f;

typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes>  > SPHFluidSurfaceMapping3f_to_Ext3d;

typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes>  > SPHFluidSurfaceMapping3f_to_Ext3f;

//SkinningMapping
//---------------------

typedef sofa::component::mapping::SkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > SkinningMechanicalMappingRigid3d_to_3d;

typedef sofa::component::mapping::SkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > SkinningMechanicalMappingRigid3d_to_3f;

typedef sofa::component::mapping::SkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > SkinningMechanicalMappingRigid3f_to_3d;

typedef sofa::component::mapping::SkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > SkinningMechanicalMappingRigid3f_to_3f;


typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > SkinningMappingRigid3d_to_3d;

typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > SkinningMappingRigid3d_to_3f;

typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > SkinningMappingRigid3f_to_3d;

typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > SkinningMappingRigid3f_to_3f;



typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > SkinningMappingRigid3d_to_Ext3d;

typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > SkinningMappingRigid3d_to_Ext3f;

typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > SkinningMappingRigid3f_to_Ext3d;

typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > SkinningMappingRigid3f_to_Ext3f;




//SubsetMapping
//---------------------
typedef sofa::component::mapping::SubsetMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > SubsetMechanicalMapping3d_to_3d;

typedef sofa::component::mapping::SubsetMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > SubsetMechanicalMapping3d_to_3f;

typedef sofa::component::mapping::SubsetMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > SubsetMechanicalMapping3f_to_3d;

typedef sofa::component::mapping::SubsetMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > SubsetMechanicalMapping3f_to_3f;


typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > SubsetMapping3d_to_3d;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > SubsetMapping3d_to_3f;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > SubsetMapping3f_to_3d;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > SubsetMapping3f_to_3f;



typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > SubsetMapping3d_to_Ext3d;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > SubsetMapping3d_to_Ext3f;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > SubsetMapping3f_to_Ext3d;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > SubsetMapping3f_to_Ext3f;



//SurfaceIdentityMapping
//---------------------
typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > SurfaceIdentityMechanicalMapping3d_to_3d;

typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > SurfaceIdentityMechanicalMapping3d_to_3f;

typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > SurfaceIdentityMechanicalMapping3f_to_3d;

typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3fTypes> > > SurfaceIdentityMechanicalMapping3f_to_3f;


typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > SurfaceIdentityMapping3d_to_3d;

typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > SurfaceIdentityMapping3d_to_3f;

typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > SurfaceIdentityMapping3f_to_3d;

typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3fTypes> > > SurfaceIdentityMapping3f_to_3f;



typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > SurfaceIdentityMapping3d_to_Ext3d;

typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > SurfaceIdentityMapping3d_to_Ext3f;

typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3dTypes> > > SurfaceIdentityMapping3f_to_Ext3d;

typedef sofa::component::mapping::SurfaceIdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3fTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > SurfaceIdentityMapping3f_to_Ext3f;

#endif

