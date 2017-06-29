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
		const Rigid3::VecCoord& i1 = *X;
		const Rigid3::VecCoord& i2 = mo->x.getValue();

		EXPECT_TRUE(i1.size() == i2.size());
		for (size_t i = 0; i < i1.size(); ++i) EXPECT_EQ(i1[i], i2[i]);
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

		EXPECT_TRUE(idx.size() <= i2.size());
		for (size_t i = 0; i < idx.size(); ++i) EXPECT_EQ(i1[idx[i]], i2[idx[i]]);

		for (size_t i = 0; i < idx.size(); ++i) EXPECT_EQ(f1[idx[i]], f2[idx[i]]);

		for (size_t i = 0; i < idx.size(); ++i) EXPECT_EQ(v1[idx[i]], v2[idx[i]]);
  }
};

struct Monitor_test : public sofa::Sofa_test<>
{
  sofa::simulation::Node::SPtr root;
  sofa::simulation::SceneLoaderXML loader;
	MonitorTest* monitor;
	MechanicalObject<Rigid3>::SPtr mo;

  Monitor_test() {}

	void testInit()
	{
		// Checks that monitor gets the correct values at init
		monitor->testInit(mo.get());
	}
  void testModif()
  {
		// make a few steps before checkinf if values are correctly updated in Monitor
		for (int i = 0 ; i < 10 ; ++i)
			simulation::getSimulation()->animate(root.get(), 1.0);

		monitor->testModif(mo.get());
  }
  void SetUp()
  {
		std::string scene =
				"<Node name='root' gravity='0 -9.81 0'>"
				"<DefaultAnimationLoop/>"
				"<Node name='node'>"
				"<EulerImplicit rayleighStiffness='0' printLog='false' />"
				"<CGLinearSolver iterations='100' threshold='0.00000001' "
				"tolerance='1e-5'/>"
				"<MeshGmshLoader name='loader' filename='mesh/smCube27.msh' "
				"createSubelements='true' />"
				"<MechanicalObject template='Rigid3d' src='@loader' name='MO' />"
				"<Monitor template='Rigid3d' name='monitor' listening='1' indices='0' "
				"ExportPositions='true' ExportVelocities='true' ExportForces='true' />"
				"<UniformMass totalmass='1' />"
				"</Node>"
				"</Node>";

		root = SceneLoaderXML::loadFromMemory("MonitorTest", scene.c_str(),
																					scene.size());
    root->init(sofa::core::ExecParams::defaultInstance());

		std::string s = "/node/monitor";
		Monitor<Rigid3>* ptr = NULL;
		ptr = root->get<Monitor<Rigid3> >(s);
		EXPECT_FALSE(ptr == NULL);

		monitor = reinterpret_cast<MonitorTest*>(ptr);
		EXPECT_FALSE(monitor == 0);

		root->get<MechanicalObject<Rigid3> >(mo, "/node/MO");
		EXPECT_FALSE(mo == 0);
	}

  void TearDown() {}
};

/// Checks whether the video is well loaded and 1st frame retrieved at init()
TEST_F(Monitor_test, testInit) { this->testInit(); }
TEST_F(Monitor_test, testModif) { this->testModif(); }

}  // namespace sofa
