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
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/helper/BackTrace.h>

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <sofa/component/engine/transform/SmoothMeshEngine.h>
using sofa::component::engine::transform::SmoothMeshEngine ;

using sofa::type::vector;


namespace sofa
{

template <typename _DataTypes>
struct SmoothMeshEngine_test : public BaseSimulationTest,
		SmoothMeshEngine<_DataTypes>
{
	typedef SmoothMeshEngine<_DataTypes> ThisClass ;
	typedef _DataTypes DataTypes;
	typedef typename DataTypes::Real Real;
	typedef sofa::type::Vec<3,Real> Vec3;

	Simulation* m_simu;
	Node::SPtr m_node;
	typename ThisClass::SPtr m_thisObject;

	void SetUp() override
	{
		m_simu = sofa::simulation::getSimulation();
		ASSERT_NE(m_simu, nullptr);

		m_node = m_simu->createNewGraph("root");
		m_thisObject = New<ThisClass >() ;
		m_node->addObject(m_thisObject) ;
	}

	void test_computeBBox()
	{
		m_thisObject->findData("input_position")->read("0. 0. 0.  1. 0. 0.  0. 1. 0.");

		m_thisObject->computeBBox(nullptr, true);

		EXPECT_EQ(m_thisObject->f_bbox.getValue().minBBox(), Vec3(0,0,0));
		EXPECT_EQ(m_thisObject->f_bbox.getValue().maxBBox(), Vec3(1,1,0));
	}
};

using ::testing::Types;
typedef Types<sofa::defaulttype::Vec3Types> DataTypes;

TYPED_TEST_SUITE(SmoothMeshEngine_test, DataTypes);

TYPED_TEST( SmoothMeshEngine_test, test_computeBBox )
{
	ASSERT_NO_THROW(this->test_computeBBox()) ;
}

} // end namespace sofa
