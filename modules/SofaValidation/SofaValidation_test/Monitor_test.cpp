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

#include <fstream>
#include <streambuf>
#include <string>
#include <cstdio>

namespace sofa
{
struct MonitorTest : public Monitor<Rigid3Types>
{
    void testInit(MechanicalObject<Rigid3Types>* mo)
    {
        const Rigid3Types::VecCoord& i1 = *m_X;
        const Rigid3Types::VecCoord& i2 = mo->x.getValue();

        EXPECT_TRUE(i1.size() == i2.size());
        for (size_t i = 0; i < i1.size(); ++i) EXPECT_EQ(i1[i], i2[i]);

        EXPECT_EQ(d_fileName, std::string("./") + getName());
    }

    void testModif(MechanicalObject<Rigid3Types>* mo)
    {
        helper::vector<unsigned int> idx = d_indices.getValue();
        const Rigid3Types::VecCoord& i1 = *m_X;
        const Rigid3Types::VecCoord& i2 = mo->x.getValue();
        const Rigid3Types::VecDeriv& f1 = *m_F;
        const Rigid3Types::VecDeriv& f2 = mo->f.getValue();
        const Rigid3Types::VecDeriv& v1 = *m_V;
        const Rigid3Types::VecDeriv& v2 = mo->v.getValue();

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
    MechanicalObject<Rigid3Types>::SPtr mo;

    Monitor_test() {}

    void testInit()
    {
        // Checks that monitor gets the correct values at init
        monitor->testInit(mo.get());
    }

    std::string readWholeFile(const std::string& fileName)
    {
        std::ifstream t(fileName);
        std::string str;
        EXPECT_TRUE(t.is_open());
        t.seekg(0, std::ios::end);
        str.reserve(t.tellg());
        t.seekg(0, std::ios::beg);

        str.assign((std::istreambuf_iterator<char>(t)),
                   std::istreambuf_iterator<char>());
        return str;
    }

    void testModif()
    {
        std::string str_x =
                "# Gnuplot File : positions of 1 particle(s) Monitored\n# 1st Column : "
                "time, others : particle(s) number 0 \n1	-3.5 -12.4182 -3.5 0 0 "
                "0 1	\n2	-3.5 -29.4438 -3.5 0 0 0 1	\n3	-3.5 -53.8398 "
                "-3.5 0 0 0 1	\n4	-3.5 -84.9362 -3.5 0 0 0 1	\n5	-3.5 "
                "-122.124 -3.5 0 0 0 1	\n6	-3.5 -164.849 -3.5 0 0 0 1	"
                "\n7	"
                "-3.5 -212.608 -3.5 0 0 0 1	\n8	-3.5 -264.944 -3.5 0 0 0 "
                "1	"
                "\n9	-3.5 -321.44 -3.5 0 0 0 1	\n10	-3.5 -381.718 -3.5 0 0 "
                "0 1	\n";
        std::string str_f =
                "# Gnuplot File : forces of 1 particle(s) Monitored\n# 1st Column : "
                "time, others : particle(s) number 0 \n1	0 -0.363333 0 0 0 "
                "0	"
                "\n2	0 -0.363333 0 0 0 0	\n3	0 -0.363333 0 0 0 0	"
                "\n4	"
                "0 -0.363333 0 0 0 0	\n5	0 -0.363333 0 0 0 0	\n6	0 "
                "-0.363333 0 0 0 0	\n7	0 -0.363333 0 0 0 0	\n8	0 "
                "-0.363333 0 0 0 0	\n9	0 -0.363333 0 0 0 0	\n10	0 "
                "-0.363333 0 0 0 0	\n";
        std::string str_v =
                "# Gnuplot File : velocities of 1 particle(s) Monitored\n# 1st Column "
                ": time, others : particle(s) number 0 \n1	0 -8.91818 0 0 0 "
                "0	"
                "\n2	0 -17.0256 0 0 0 0	\n3	0 -24.396 0 0 0 0	"
                "\n4	"
                "0 -31.0964 0 0 0 0	\n5	0 -37.1876 0 0 0 0	\n6	0 "
                "-42.7251 0 0 0 0	\n7	0 -47.7592 0 0 0 0	\n8	0 "
                "-52.3356 0 0 0 0	\n9	0 -56.496 0 0 0 0	\n10	0 "
                "-60.2782 0 0 0 0	\n";

        // make a few steps before checkinf if values are correctly updated in
        // Monitor
        for (int i = 0; i < 10; ++i)
            simulation::getSimulation()->animate(root.get(), 1.0);

        monitor->testModif(mo.get());

        std::string s_x = readWholeFile(monitor->d_fileName.getFullPath() + "_x.txt");
        std::string s_f = readWholeFile(monitor->d_fileName.getFullPath() + "_f.txt");
        std::string s_v = readWholeFile(monitor->d_fileName.getFullPath() + "_v.txt");
        EXPECT_EQ(s_x, str_x);
        EXPECT_EQ(s_f, str_f);
        EXPECT_EQ(s_v, str_v);
        std::remove(std::string(monitor->d_fileName.getFullPath() + "_x.txt").c_str());
        std::remove(std::string(monitor->d_fileName.getFullPath() + "_f.txt").c_str());
        std::remove(std::string(monitor->d_fileName.getFullPath() + "_v.txt").c_str());
    }
    void SetUp()
    {
        std::string scene =
                "<Node name='root' gravity='0 -9.81 0'>"
                "<DefaultAnimationLoop/>"
                "<Node name='node'>"
                "<EulerImplicit rayleighStiffness='0' printLog='false' rayleighMass='0.1'/>"
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
        Monitor<Rigid3Types>* ptr = NULL;
        ptr = root->get<Monitor<Rigid3Types> >(s);
        EXPECT_FALSE(ptr == NULL);

        monitor = reinterpret_cast<MonitorTest*>(ptr);
        EXPECT_FALSE(monitor == 0);

        root->get<MechanicalObject<Rigid3Types> >(mo, "/node/MO");
        EXPECT_FALSE(mo == 0);
    }

    void TearDown()
    {
    }
};

/// Checks whether the video is well loaded and 1st frame retrieved at init()
TEST_F(Monitor_test, testInit) { this->testInit(); }
TEST_F(Monitor_test, testModif) { this->testModif(); }

}  // namespace sofa
