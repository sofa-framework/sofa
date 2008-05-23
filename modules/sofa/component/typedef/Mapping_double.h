#ifndef SOFA_TYPEDEF_MAPPING_DOUBLE_H
#define SOFA_TYPEDEF_MAPPING_DOUBLE_H


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
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > ArticulatedSystemMapping1d_to_Rigid3d;



//BarycentricMapping
//---------------------
typedef sofa::component::mapping::BarycentricMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > BarycentricMechanicalMapping3d_to_3d;


typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > BarycentricMapping3d_to_3d;


typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > BarycentricMapping3d_to_Ext3;


//BeamLinearMapping
//---------------------
typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > >    BeamLinearMechanicalMappingRigid3d_to_3d;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > >    BeamLinearMappingRigid3d_to_3d;

typedef sofa::component::mapping::BeamLinearMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > BeamLinearMapping3d_to_Ext3;




//CenterOfMassMapping
//---------------------
typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > CenterOfMassMechanicalMappingRigid3d_to_3d;

typedef sofa::component::mapping::CenterOfMassMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes> > > CenterOfMassMechanicalMappingRigid2d_to_2d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > CenterOfMassMappingRigid3d_to_3d;

typedef sofa::component::mapping::BarycentricMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > CenterOfMassMappingRigid3d_to_Ext3;


//CurveMapping
//---------------------
typedef sofa::component::mapping::CurveMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > CurveMechanicalMappingRigid3d_to_3d;

//IdentityMapping
//---------------------
typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes> > > IdentityMechanicalMapping1d_to_1d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes> > > IdentityMechanicalMapping2d_to_2d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > IdentityMechanicalMapping3d_to_3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec6dTypes> > > IdentityMechanicalMapping6d_to_6d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes> > > IdentityMechanicalMappingRigid2d_to_Rigid2d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > IdentityMechanicalMappingRigid3d_to_Rigid3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > IdentityMapping3d_to_3d;

typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > IdentityMapping3d_to_Ext3;


typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid2dTypes> > > IdentityMappingRigid2d_to_Rigid2d;


typedef sofa::component::mapping::IdentityMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3dTypes> > > IdentityMappingRigid3d_to_Rigid3d;


//ImplicitSurfaceMapping
//---------------------
typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes>  > ImplicitSurfaceMapping3d_to_3d;


typedef sofa::component::mapping::ImplicitSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes>  > ImplicitSurfaceMapping3d_to_Ext3;



//LaparoscopicRigidMapping
//---------------------


//LineSetSkinningMapping
//---------------------
typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > LineSetSkinningMechanicalMappingRigid3d_to_3d;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > LineSetSkinningMappingRigid3d_to_3d;

typedef sofa::component::mapping::LineSetSkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > LineSetSkinningMappingRigid3d_to_Ext3;

//RigidMapping
//---------------------
typedef sofa::component::mapping::RigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > RigidMechanicalMappingRigid3d_to_3d;

typedef sofa::component::mapping::RigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid2dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec2dTypes> > > RigidMechanicalMappingRigid2d_to_2d;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > RigidMappingRigid3d_to_3d;

typedef sofa::component::mapping::RigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > RigidMappingRigid3d_to_Ext3;

//RigidRigidMapping
//---------------------
typedef sofa::component::mapping::RigidRigidMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes> > > RigidRigidMechanicalMappingRigid3d_to_Rigid3d;


typedef sofa::component::mapping::RigidRigidMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Rigid3dTypes> > > RigidRigidMappingRigid3d_to_Rigid3d;

//SPHFluidSurfaceMapping
//---------------------
typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes>  > SPHFluidSurfaceMapping3d_to_3d;

typedef sofa::component::mapping::SPHFluidSurfaceMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes>  > SPHFluidSurfaceMapping3d_to_Ext3;

//SkinningMapping
//---------------------
typedef sofa::component::mapping::SkinningMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > SkinningMechanicalMappingRigid3d_to_3d;


typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > SkinningMappingRigid3d_to_3d;


typedef sofa::component::mapping::SkinningMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Rigid3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > SkinningMappingRigid3d_to_Ext3;




//SubsetMapping
//---------------------
typedef sofa::component::mapping::SubsetMapping< sofa::core::componentmodel::behavior::MechanicalMapping<
sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::Vec3dTypes> > > SubsetMechanicalMapping3d_to_3d;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::Vec3dTypes> > > SubsetMapping3d_to_3d;

typedef sofa::component::mapping::SubsetMapping< sofa::core::Mapping<
sofa::core::componentmodel::behavior::State<sofa::defaulttype::Vec3dTypes>,
     sofa::core::componentmodel::behavior::MappedModel<sofa::defaulttype::ExtVec3fTypes> > > SubsetMapping3d_to_Ext3;



#ifndef SOFA_FLOAT
typedef        ArticulatedSystemMapping1d_to_Rigid3d                   ArticulatedSystemMapping1d_to_Rigid3;
typedef        BarycentricMechanicalMapping3d_to_3d		       BarycentricMechanicalMapping3_to_3;
typedef        BarycentricMapping3d_to_3d			       BarycentricMapping3_to_3;
typedef        BarycentricMapping3d_to_Ext3			       BarycentricMapping3_to_Ext3;
typedef        BeamLinearMechanicalMappingRigid3d_to_3d	               BeamLinearMechanicalMappingRigid3_to_3;
typedef        BeamLinearMappingRigid3d_to_3d			       BeamLinearMappingRigid3_to_3;
typedef        BeamLinearMapping3d_to_Ext3			       BeamLinearMapping3_to_Ext3;
typedef        CenterOfMassMechanicalMappingRigid3d_to_3d	       CenterOfMassMechanicalMappingRigid3_to_3;
typedef        CenterOfMassMechanicalMappingRigid2d_to_2d	       CenterOfMassMechanicalMappingRigid2_to_2;
typedef        CenterOfMassMappingRigid3d_to_3d	        	       CenterOfMassMappingRigid3_to_3;
typedef        CenterOfMassMappingRigid3d_to_Ext3		       CenterOfMassMappingRigid3_to_Ext3;
typedef        CurveMechanicalMappingRigid3d_to_3d		       CurveMechanicalMappingRigid3_to_3;
typedef        IdentityMechanicalMapping1d_to_1d		       IdentityMechanicalMapping1d_to_1d;
typedef        IdentityMechanicalMapping2d_to_2d		       IdentityMechanicalMapping2_to_2;
typedef        IdentityMechanicalMapping3d_to_3d		       IdentityMechanicalMapping3_to_3;
typedef        IdentityMechanicalMapping6d_to_6d		       IdentityMechanicalMapping6_to_6;
typedef        IdentityMechanicalMappingRigid2d_to_Rigid2d	       IdentityMechanicalMappingRigid2_to_Rigid2;
typedef        IdentityMechanicalMappingRigid3d_to_Rigid3d	       IdentityMechanicalMappingRigid3_to_Rigid3;
typedef        IdentityMapping3d_to_3d				       IdentityMapping3_to_3;
typedef        IdentityMapping3d_to_Ext3			       IdentityMapping3_to_Ext3;
typedef        IdentityMappingRigid2d_to_Rigid2d		       IdentityMappingRigid2_to_Rigid2;
typedef        IdentityMappingRigid3d_to_Rigid3d		       IdentityMappingRigid3_to_Rigid3;
typedef        ImplicitSurfaceMapping3d_to_3d			       ImplicitSurfaceMapping3_to_3;
typedef        ImplicitSurfaceMapping3d_to_Ext3		               ImplicitSurfaceMapping3_to_Ext3;
typedef        LineSetSkinningMechanicalMappingRigid3d_to_3d	       LineSetSkinningMechanicalMappingRigid3_to_3;
typedef        LineSetSkinningMappingRigid3d_to_3d		       LineSetSkinningMappingRigid3_to_3;
typedef        LineSetSkinningMappingRigid3d_to_Ext3		       LineSetSkinningMappingRigid3_to_Ext3;
typedef        RigidMechanicalMappingRigid3d_to_3d		       RigidMechanicalMappingRigid3_to_3;
typedef        RigidMechanicalMappingRigid2d_to_2d		       RigidMechanicalMappingRigid2_to_2;
typedef        RigidMappingRigid3d_to_3d			       RigidMappingRigid3_to_3;
typedef        RigidMappingRigid3d_to_Ext3			       RigidMappingRigid3_to_Ext3;
typedef        RigidRigidMechanicalMappingRigid3d_to_Rigid3d	       RigidRigidMechanicalMappingRigid3_to_Rigid3;
typedef        RigidRigidMappingRigid3d_to_Rigid3d		       RigidRigidMappingRigid3_to_Rigid3;
typedef        SPHFluidSurfaceMapping3d_to_3d			       SPHFluidSurfaceMapping3_to_3;
typedef        SPHFluidSurfaceMapping3d_to_Ext3		               SPHFluidSurfaceMapping3_to_Ext3;
typedef        SkinningMechanicalMappingRigid3d_to_3d		       SkinningMechanicalMappingRigid3_to_3;
typedef        SkinningMappingRigid3d_to_3d			       SkinningMappingRigid3_to_3;
typedef        SkinningMappingRigid3d_to_Ext3			       SkinningMappingRigid3_to_Ext3;
typedef        SubsetMechanicalMapping3d_to_3d			       SubsetMechanicalMapping3_to_3;
typedef        SubsetMapping3d_to_3d				       SubsetMapping3_to_3;
typedef        SubsetMapping3d_to_Ext3				       SubsetMapping3_to_Ext3;
#endif

#endif
