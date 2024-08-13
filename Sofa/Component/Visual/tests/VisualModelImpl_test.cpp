/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <gtest/gtest.h>
#include <sofa/component/visual/VisualModelImpl.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa {

template <typename _DataTypes>
struct VisualModelImpl_test : public ::testing::Test
{
    typedef _DataTypes DataTypes;

    VisualModelImpl_test()
	{
		//Init
	}

};

struct StubVisualModelImpl : public component::visual::VisualModelImpl {};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    defaulttype::Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_SUITE(VisualModelImpl_test, DataTypes);

template <class T>
bool Vector_Comparison(type::vector< T > expected, type::vector< T > actual)
{
    if (expected.size() != actual.size())
        return false;
    return true;
}

// Ctor test case
TEST( VisualModelImpl_test , checkThatMembersAreCorrectlyConstructed )
{
    StubVisualModelImpl visualModel;

    // crazy way to remove warnings
    bool false_var = false;
    bool true_var  = true;

    ASSERT_EQ(false_var, visualModel.useTopology);
    ASSERT_EQ(-1,        visualModel.lastMeshRev);
    ASSERT_EQ(true_var,  visualModel.castShadow);
    ASSERT_EQ(false_var, visualModel.d_initRestPositions.getValue());
    ASSERT_EQ(true_var,  visualModel.getUseNormals());
    ASSERT_EQ(true_var,  visualModel.d_updateNormals.getValue());
    ASSERT_EQ(false_var, visualModel.d_computeTangents.getValue());
    ASSERT_EQ(true_var,  visualModel.d_updateTangents.getValue());
    ASSERT_EQ(true_var,  visualModel.d_handleDynamicTopology.getValue());
    ASSERT_EQ(true_var,  visualModel.d_fixMergedUVSeams.getValue());

    ASSERT_EQ(true_var, Vector_Comparison(component::visual::VisualModelImpl::VecCoord(), visualModel.d_vertices2.getValue()));
    ASSERT_EQ(true_var, Vector_Comparison(component::visual::VisualModelImpl::VecCoord(), visualModel.d_vtangents.getValue()));
    ASSERT_EQ(true_var, Vector_Comparison(component::visual::VisualModelImpl::VecCoord(), visualModel.d_vbitangents.getValue()));

    ASSERT_EQ(true_var, Vector_Comparison(component::visual::VisualModelImpl::VecVisualEdge(), visualModel.d_edges.getValue()));
    ASSERT_EQ(true_var, Vector_Comparison(component::visual::VisualModelImpl::VecVisualTriangle(), visualModel.d_triangles.getValue()));
    ASSERT_EQ(true_var, Vector_Comparison(component::visual::VisualModelImpl::VecVisualQuad(), visualModel.d_quads.getValue()));
    ASSERT_EQ(true_var, Vector_Comparison(type::vector<component::visual::VisualModelImpl::visual_index_type>(), visualModel.d_vertPosIdx.getValue()));
    ASSERT_EQ(true_var, Vector_Comparison(type::vector<component::visual::VisualModelImpl::visual_index_type>(), visualModel.d_vertNormIdx.getValue()));

    ASSERT_EQ(core::objectmodel::DataFileName().getValue(), visualModel.d_fileMesh.getValue());
    ASSERT_EQ(core::objectmodel::DataFileName().getValue(), visualModel.d_texturename.getValue());
    ASSERT_EQ(component::visual::VisualModelImpl::Vec3Real(), visualModel.d_translation.getValue());
    ASSERT_EQ(component::visual::VisualModelImpl::Vec3Real(), visualModel.d_rotation.getValue());
    ASSERT_EQ(component::visual::VisualModelImpl::Vec3Real(1.0,1.0,1.0), visualModel.d_scale.getValue());
    ASSERT_EQ(component::visual::VisualModelImpl::TexCoord(1.0,1.0), visualModel.d_scaleTex.getValue());
    ASSERT_EQ(component::visual::VisualModelImpl::TexCoord(0.0,0.0), visualModel.d_translationTex.getValue());

    ASSERT_EQ(sofa::type::Material().name, visualModel.d_material.getValue().name);
    ASSERT_EQ(false_var, visualModel.d_putOnlyTexCoords.getValue());
    ASSERT_EQ(false_var, visualModel.d_srgbTexturing.getValue());
    ASSERT_EQ(false_var, visualModel.xformsModified);
    ASSERT_EQ(nullptr, visualModel.m_topology);
    ASSERT_EQ(true_var, visualModel.getDataAliases().find("filename") != visualModel.getDataAliases().end());

    ASSERT_EQ("Vector", std::string(visualModel.d_vertices2.getGroup()));
    ASSERT_EQ("Vector", std::string(visualModel.m_vnormals.getGroup()));
    ASSERT_EQ("Vector", std::string(visualModel.d_vtexcoords.getGroup()));
    ASSERT_EQ("Vector", std::string(visualModel.d_vtangents.getGroup()));
    ASSERT_EQ("Vector", std::string(visualModel.d_vbitangents.getGroup()));
    ASSERT_EQ("Vector", std::string(visualModel.d_edges.getGroup()));
    ASSERT_EQ("Vector", std::string(visualModel.d_triangles.getGroup()));
    ASSERT_EQ("Vector", std::string(visualModel.d_quads.getGroup()));

    ASSERT_EQ("Transformation", std::string(visualModel.d_translation.getGroup()));
    ASSERT_EQ("Transformation", std::string(visualModel.d_rotation.getGroup()));
    ASSERT_EQ("Transformation", std::string(visualModel.d_scale.getGroup()));

    ASSERT_EQ(false_var, visualModel.d_edges.getFlag(core::objectmodel::BaseData::FLAG_AUTOLINK));

    ASSERT_EQ(1u, visualModel.xforms.size());
}

} //sofa
