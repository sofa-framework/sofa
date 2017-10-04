#include <SofaTest/Sofa_test.h>
#include <sofa/helper/BackTrace.h>

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <SofaGeneralEngine/SmoothMeshEngine.h>
using sofa::component::engine::SmoothMeshEngine ;

using sofa::helper::vector;


namespace sofa
{

template <typename _DataTypes>
struct SmoothMeshEngine_test : public Sofa_test<typename _DataTypes::Real>,
		SmoothMeshEngine<_DataTypes>
{
	typedef SmoothMeshEngine<_DataTypes> ThisClass ;
	typedef _DataTypes DataTypes;
	typedef typename DataTypes::Real Real;
	typedef sofa::defaulttype::Vec<3,Real> Vec3;

	Simulation* m_simu;
	Node::SPtr m_node;
	typename ThisClass::SPtr m_thisObject;

	void SetUp()
	{
		setSimulation(m_simu = new DAGSimulation());
		m_node = m_simu->createNewGraph("root");
		m_thisObject = New<ThisClass >() ;
		m_node->addObject(m_thisObject) ;
	}

	void test_computeBBox()
	{
		m_thisObject->findData("input_position")->read("0. 0. 0.  1. 0. 0.  0. 1. 0.");

		m_thisObject->computeBBox(NULL, true);

		EXPECT_EQ(m_thisObject->f_bbox.getValue().minBBox(), Vec3(0,0,0));
		EXPECT_EQ(m_thisObject->f_bbox.getValue().maxBBox(), Vec3(1,1,0));
	}
};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(SmoothMeshEngine_test, DataTypes);

TYPED_TEST( SmoothMeshEngine_test, test_computeBBox )
{
	ASSERT_NO_THROW(this->test_computeBBox()) ;
}

} // end namespace sofa
