#include <SofaValidation/Monitor.h>
using sofa::component::misc::Monitor;
#include <SofaBaseMechanics/MechanicalObject.h>
using sofa::component::container::MechanicalObject;
#include <SofaTest/Sofa_test.h>

#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Simulation.h>
using sofa::core::objectmodel::BaseObject;
using sofa::simulation::Simulation;
using sofa::simulation::Node;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML;
using sofa::core::ExecParams;

namespace sofa
{
struct MonitorTest : public Monitor<Rigid3>
{
	void testInit(MechanicalObject<Rigid3>* mo)
	{
		helper::vector<unsigned> i1 = indices.getValue();
		const Rigid3::VecCoord& i2 = mo->x.getValue();

		ASSERT_TRUE(i1.size() <= i2.size());
		for (int i = 0; i < i1.size(); ++i) EXPECT_EQ(i1[i], i2[i]);
	}

	void testModif(MechanicalObject<Rigid3>* mo)
	{
		helper::vector<unsigned int> idx = indices.getValue();
		const Rigid3::VecCoord& i1 = *X;
		const Rigid3::VecCoord& i2 = mo->x.getValue();
		const Rigid3::VecDeriv& f1 = *F;
		const Rigid3::VecDeriv& f2 = mo->f.getValue();
		const Rigid3::VecDeriv& v1 = *V;
		const Rigid3::VecDeriv& v2 = mo->v.getValue();

		ASSERT_TRUE(idx.size() <= i2.size());
		for (int i = 0; i < idx.size(); ++i)
			EXPECT_EQ(i1[idx[i]], i2[idx[i]]);

		for (int i = 0; i < idx.size(); ++i)
			EXPECT_EQ(f1[idx[i]], f2[idx[i]]);

		for (int i = 0; i < idx.size(); ++i)
			EXPECT_EQ(v1[idx[i]], v2[idx[i]]);
	}
};

struct Monitor_test : public sofa::Sofa_test<>
{
  sofa::simulation::Node::SPtr root;
	MonitorTest* monitor;
	MechanicalObject<Rigid3>* mo;

	Monitor_test() {}

	void testInit() { monitor->testInit(mo); }
	void testModiff() { monitor->testModif(mo); }
	void SetUp()
  {
		std::string scene = "examples/Components/misc/MonitorTest.scn";

    root = sofa::simulation::SceneLoaderXML::load(scene.c_str());
    root->init(sofa::core::ExecParams::defaultInstance());

    monitor = dynamic_cast<Monitor*>(root->getObject("monitor"));
    mo = dynamic_cast<MechanicalObject*>(root->getObject("MO"));
  }

  void TearDown() {}
};

/// Checks whether the video is well loaded and 1st frame retrieved at init()
TEST_F(Monitor_test, testInit) { this->testInit(); }
TEST_F(Monitor_test, testModif) { this->testModif(); }

}  // namespace sofa
