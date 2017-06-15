/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "TableDataWidget.h"
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/Factory.inl>
#include <iostream>

namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::helper::Creator;
using sofa::helper::fixed_array;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(TableDataWidget);

Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<int>, TABLE_HORIZONTAL > > DWClass_vectori("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<unsigned int>, TABLE_HORIZONTAL > > DWClass_vectoru("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<float>, TABLE_HORIZONTAL > > DWClass_vectorf("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<double>, TABLE_HORIZONTAL > > DWClass_vectord("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<std::string> > > DWClass_vectorstring("default",true);

//Creator<DataWidgetFactory, TableDataWidget< sofa::component::topology::PointData<int>, TABLE_HORIZONTAL > > DWClass_Pointi("default",true);
//Creator<DataWidgetFactory, TableDataWidget< sofa::component::topology::PointData<unsigned int>, TABLE_HORIZONTAL > > DWClass_Pointu("default",true);
//Creator<DataWidgetFactory, TableDataWidget< sofa::component::topology::PointData<float>, TABLE_HORIZONTAL > > DWClass_Pointf("default",true);
//Creator<DataWidgetFactory, TableDataWidget< sofa::component::topology::PointData<double>, TABLE_HORIZONTAL > > DWClass_Pointd("default",true);

#ifdef TODOTOPO
Creator<DataWidgetFactory, TableDataWidget< sofa::component::topology::PointSubset, TABLE_HORIZONTAL > > DWClass_PointSubset("default",true);
#endif

Creator<DataWidgetFactory, TableDataWidget< sofa::core::topology::BaseMeshTopology::SeqEdges      > > DWClass_SeqEdges     ("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::core::topology::BaseMeshTopology::SeqTriangles  > > DWClass_SeqTriangles ("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::core::topology::BaseMeshTopology::SeqQuads      > > DWClass_SeqQuads     ("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::core::topology::BaseMeshTopology::SeqTetrahedra > > DWClass_SeqTetrahedra("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::core::topology::BaseMeshTopology::SeqHexahedra  > > DWClass_SeqHexahedra ("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<1,int> > > > DWClass_vectorVec1i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<1,unsigned int> > > > DWClass_vectorVec1u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<1,float> > > > DWClass_vectorVec1f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<1,double> > > > DWClass_vectorVec1d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<2,int> > > > DWClass_vectorVec2i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<2,unsigned int> > > > DWClass_vectorVec2u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<2,float> > > > DWClass_vectorVec2f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<2,double> > > > DWClass_vectorVec2d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<3,int> > > > DWClass_vectorVec3i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<3,unsigned int> > > > DWClass_vectorVec3u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<3,float> > > > DWClass_vectorVec3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<3,double> > > > DWClass_vectorVec3d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<4,int> > > > DWClass_vectorVec4i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<4,unsigned int> > > > DWClass_vectorVec4u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<4,float> > > > DWClass_vectorVec4f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<4,double> > > > DWClass_vectorVec4d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<6,int> > > > DWClass_vectorVec6i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<6,unsigned int> > > > DWClass_vectorVec6u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<6,float> > > > DWClass_vectorVec6f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<6,double> > > > DWClass_vectorVec6d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<8,int> > > > DWClass_vectorVec8i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<Vec<8,unsigned int> > > > DWClass_vectorVec8u("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<int,1> > > > DWClass_vectorA1i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<unsigned int,1> > > > DWClass_vectorA1u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<int,2> > > > DWClass_vectorA2i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<unsigned int,2> > > > DWClass_vectorA2u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<int,3> > > > DWClass_vectorA3i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<unsigned int,3> > > > DWClass_vectorA3u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<float,3> > > > DWClass_vectorA3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<double,3> > > > DWClass_vectorA3d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<int,4> > > > DWClass_vectorA4i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<unsigned int,4> > > > DWClass_vectorA4u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<int,6> > > > DWClass_vectorA6i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<unsigned int,6> > > > DWClass_vectorA6u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<int,8> > > > DWClass_vectorA8i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<fixed_array<unsigned int,8> > > > DWClass_vectorA8u("default",true);

#if !defined(_MSC_VER) && !defined(__clang__)
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,1> > > > DWClass_stdvectorA1i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,1> > > > DWClass_stdvectorA1u("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,2> > > > DWClass_stdvectorA2i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,2> > > > DWClass_stdvectorA2u("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,3> > > > DWClass_stdvectorA3i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,3> > > > DWClass_stdvectorA3u("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,4> > > > DWClass_stdvectorA4i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,4> > > > DWClass_stdvectorA4u("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,6> > > > DWClass_stdvectorA6i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,6> > > > DWClass_stdvectorA6u("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,8> > > > DWClass_stdvectorA8i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,8> > > > DWClass_stdvectorA8u("default",true);
#endif

Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<int>, TABLE_HORIZONTAL > > DWClass_ResizableExtVectori("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<unsigned int>, TABLE_HORIZONTAL > > DWClass_ResizableExtVectoru("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<float>, TABLE_HORIZONTAL > > DWClass_ResizableExtVectorf("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<double>, TABLE_HORIZONTAL > > DWClass_ResizableExtVectord("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<std::string> > > DWClass_ResizableExtVectorstring("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<fixed_array<int,3> > > > DWClass_ResizableExtVectorA3i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<fixed_array<int,4> > > > DWClass_ResizableExtVectorA4i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<fixed_array<unsigned int,3> > > > DWClass_ResizableExtVectorA3u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<fixed_array<unsigned int,4> > > > DWClass_ResizableExtVectorA4u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<Vec<2, float> > > > DWClass_ResizableExtVectorVec2f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<Vec<2, double> > > > DWClass_ResizableExtVectorVec2d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<Vec<3, float> > > > DWClass_ResizableExtVectorVec3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<Vec<3, double> > > > DWClass_ResizableExtVectorVec3d("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<sofa::core::topology::Topology::Edge > > > DWClass_ResizableExtVectorEdge("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<sofa::core::topology::Topology::Triangle > > > DWClass_ResizableExtVectorTriangle("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<sofa::core::topology::Topology::Quad > > > DWClass_ResizableExtVectorQuad("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<sofa::core::topology::Topology::Tetrahedron > > > DWClass_ResizableExtVectorTetrahedron("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::defaulttype::ResizableExtVector<sofa::core::topology::Topology::Hexahedron > > > DWClass_ResizableExtVectorHexahedron("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::helper::Quater<float> > > > DWClass_vectorQuatf("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::helper::Quater<double> > > > DWClass_vectorQuatd("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::helper::Polynomial_LD<double,5> > > > DWClass_vectorPolynomialLD5d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::helper::Polynomial_LD<double,4> > > > DWClass_vectorPolynomialLD4d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::helper::Polynomial_LD<double,3> > > > DWClass_vectorPolynomialLD3d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::helper::Polynomial_LD<double,2> > > > DWClass_vectorPolynomialLD2d("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::helper::Polynomial_LD<float,5> > > > DWClass_vectorPolynomialLD5f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::helper::Polynomial_LD<float,4> > > > DWClass_vectorPolynomialLD4f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::helper::Polynomial_LD<float,3> > > > DWClass_vectorPolynomialLD3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::helper::Polynomial_LD<float,2> > > > DWClass_vectorPolynomialLD2f("default",true);

#ifdef TODOLINK
Creator<DataWidgetFactory,TableDataWidget< sofa::core::objectmodel::VectorObjectRef >  >  DWClass_DataVectorRefWidget("default",true);
#endif

Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::defaulttype::RigidCoord<2,float> > > > DWClass_vectorRigidCoord2f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::defaulttype::RigidCoord<2,double> > > > DWClass_vectorRigidCoord2d("default",true);
//Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::defaulttype::RigidDeriv<2,float> > > > DWClass_vectorRigidDeriv2f("default",true);
//Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::defaulttype::RigidDeriv<2,double> > > > DWClass_vectorRigidDeriv2d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::defaulttype::RigidCoord<3,float> > > > DWClass_vectorRigidCoord3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::defaulttype::RigidCoord<3,double> > > > DWClass_vectorRigidCoord3d("default",true);
//Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::defaulttype::RigidDeriv<3,float> > > > DWClass_vectorRigidDeriv3f("default",true);
//Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::defaulttype::RigidDeriv<3,double> > > > DWClass_vectorRigidDeriv3d("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::defaulttype::LaparoscopicRigid3Types::Coord > > > DWClass_vectorLaparoRigidCoord3("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::defaulttype::LaparoscopicRigid3Types::Deriv > > > DWClass_vectorLaparoRigidDeriv3("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::component::interactionforcefield::LinearSpring<float> > > > DWClass_vectorLinearSpringf("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::component::interactionforcefield::LinearSpring<double> > > > DWClass_vectorLinearSpringd("default",true);

//Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::component::interactionforcefield::JointSpring<sofa::defaulttype::Rigid2fTypes> > > > DWClass_vectorJointSpring2f("default",true);
//Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::component::interactionforcefield::JointSpring<sofa::defaulttype::Rigid2dTypes> > > > DWClass_vectorJointSpring2d("default",true);
#ifndef SOFA_DOUBLE
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::component::interactionforcefield::JointSpring<sofa::defaulttype::Rigid3fTypes> > > > DWClass_vectorJointSpring3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::component::interactionforcefield::GearSpring<sofa::defaulttype::Rigid3fTypes> > > > DWClass_vectorGearSpring3f("default",true);
#endif

#ifndef SOFA_FLOAT
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::component::interactionforcefield::JointSpring<sofa::defaulttype::Rigid3dTypes> > > > DWClass_vectorJointSpring3d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::component::interactionforcefield::GearSpring<sofa::defaulttype::Rigid3dTypes> > > > DWClass_vectorGearSpring3d("default",true);
#endif



// Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::component::DiscreteElementModelInternalData<sofa::defaulttype::Vec3fTypes> > > > DWClass_vectorDiscreteElementModelInternalData3f("default", true);
// Creator<DataWidgetFactory, TableDataWidget< sofa::helper::vector<sofa::component::DiscreteElementModelInternalData<sofa::defaulttype::Vec3dTypes> > > > DWClass_vectorDiscreteElementModelInternalData3d("default", true);

} // namespace qt

} // namespace gui

} // namespace sofa
