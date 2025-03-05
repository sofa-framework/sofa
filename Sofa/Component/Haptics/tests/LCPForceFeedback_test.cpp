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

#include <sofa/testing/BaseTest.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/haptics/LCPForceFeedback.h>
#include <thread>
#include <sofa/simulation/Node.h>

namespace sofa 
{
using sofa::simulation::Simulation;
using sofa::simulation::Node;
using sofa::core::ExecParams;
using namespace sofa::helper::system::thread;

//template <typename _DataTypes>
class LCPForceFeedback_test : public sofa::testing::BaseTest
{
public:
    typedef sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Rigid3Types> MecaRig;
    typedef sofa::component::haptics::LCPForceFeedback<sofa::defaulttype::Rigid3Types> LCPRig;

    typedef typename MecaRig::Coord    Coord;
    typedef typename MecaRig::VecCoord VecCoord;
    typedef typename MecaRig::Deriv Deriv;
    typedef typename MecaRig::VecDeriv VecDeriv;
    typedef typename MecaRig::MatrixDeriv MatrixDeriv;

    bool test_InitScene();

    bool test_SimpleCollision();

    bool test_Collision();

    bool test_multiThread();

    /// General Haptic thread methods
    static void HapticsThread(std::atomic<bool>& terminate, void * p_this);

public:
    // point to the LCP for haptic access
    LCPRig::SPtr m_LCPFFBack;

    /// values to exchange info with haptic thread test
    int m_cptLoop = 0;
    int m_cptLoopContact = 0;
    sofa::type::Vec3 m_meanForceFFBack = sofa::type::Vec3(0, 0, 0);
    
    sofa::type::Vec3 m_currentPosition = sofa::type::Vec3(0, 0, 0);
    std::mutex mtxPosition;
protected:
    /// Internal method to load a scene test file
    void loadTestScene(const std::string& filename);

    /// pointer to the simulation root node loaded by @sa loadTestScene
    Node::SPtr m_root;

    /// Epsilon value for the numerical check
    SReal epsilonTest = 1e-6;
    
    /// variables for thread test
    std::thread haptic_thread;    
    std::atomic<bool> m_terminate;
};



void LCPForceFeedback_test::HapticsThread(std::atomic<bool>& terminate, void * p_this)
{
    LCPForceFeedback_test* driverTest = static_cast<LCPForceFeedback_test*>(p_this);

    // Loop Timer
    const long targetSpeedLoop = 1; // Target loop speed: 1ms

    const ctime_t refTicksPerMs = CTime::getRefTicksPerSec() / 1000;
    const ctime_t targetTicksPerLoop = targetSpeedLoop * refTicksPerMs;
    
    // Haptics Loop
    while (!terminate)
    {
        const ctime_t startTime = CTime::getRefTime();

        driverTest->mtxPosition.lock();
        sofa::type::Vec3 posInSofa = driverTest->m_currentPosition; // will apply -1 on y to simulate penetration
        driverTest->mtxPosition.unlock();
        sofa::type::Vec3 force;
        driverTest->m_LCPFFBack->computeForce(posInSofa[0], posInSofa[1]-1.0, posInSofa[2], 0, 0, 0, 0,
            force[0], force[1], force[2]);

        if (force.norm() > 0.0) // in contact
        {
            driverTest->m_cptLoopContact++;
            driverTest->m_meanForceFFBack += force;
        }


        ctime_t endTime = CTime::getRefTime();
        ctime_t duration = endTime - startTime;

        // If loop is quicker than the target loop speed. Wait here.
        while (duration < targetTicksPerLoop)
        {
            endTime = CTime::getRefTime();
            duration = endTime - startTime;
        }
        
        driverTest->m_cptLoop++;
    }
}


void LCPForceFeedback_test::loadTestScene(const std::string& filename)
{
    simulation::Simulation* simu = sofa::simulation::getSimulation();
    ASSERT_NE(simu, nullptr);

    /// Load the scene
    const std::string sceneFilename = std::string(SOFA_COMPONENT_HAPTICS_TEST_SCENES_DIR) + "/" + filename;
    m_root = simu->createNewGraph("root");    
    m_root = sofa::simulation::node::load(sceneFilename.c_str());

    EXPECT_NE(m_root, nullptr);

    sofa::simulation::node::initRoot(m_root.get());
}


bool LCPForceFeedback_test::test_InitScene()
{
    loadTestScene("ToolvsFloorCollision_test.scn");

    const simulation::Node::SPtr instruNode = m_root->getChild("Instrument");
    EXPECT_NE(instruNode, nullptr);
    const MecaRig::SPtr meca = instruNode->get<MecaRig>(instruNode->SearchDown);
    m_LCPFFBack = instruNode->get<LCPRig>(instruNode->SearchDown);

    // Check components access
    EXPECT_NE(meca, nullptr);
    EXPECT_NE(m_LCPFFBack, nullptr);

    // Check meca size and init position
    EXPECT_EQ(meca->getSize(), 1);
    if (meca->getSize() > 0)
    {
        const Coord rigZero;
        const VecCoord& coords = meca->x.getValue();
        EXPECT_EQ(coords[0], rigZero);
    }
    
    // check meca constraint, expect no cons in this world
    const MatrixDeriv& cons = meca->c.getValue();
    EXPECT_EQ(cons.size(), 0);

    return true;
}


bool LCPForceFeedback_test::test_SimpleCollision()
{
    loadTestScene("ToolvsFloorCollision_test.scn");
    const simulation::Node::SPtr instruNode = m_root->getChild("Instrument");
    EXPECT_NE(instruNode, nullptr);
    const MecaRig::SPtr meca = instruNode->get<MecaRig>(instruNode->SearchDown);
    const LCPRig::SPtr lcp = instruNode->get<LCPRig>(instruNode->SearchDown);


    // Check components access
    EXPECT_NE(meca, nullptr);
    EXPECT_NE(lcp, nullptr);

    // Check meca size and init position
    EXPECT_EQ(meca->getSize(), 1);

    VecCoord truthCoords;
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -0.002498750625, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -0.1646431247, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -0.5752928747, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -1.233208884, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -2.137158214, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -3.285914075, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -4.678255793, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -6.312968782, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -8.188844511, 0), sofa::type::Quat<double>(0, 0, 0, 1)));

    truthCoords.push_back(Coord(sofa::type::Vec3d(0.06312707665, -9.252446766, 0.01034522507), sofa::type::Quat<double>(0.01791466055, -0.001121278545, -0.1466133921, 0.989031001)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0.1068031131, -9.480637263, 0.01138742455), sofa::type::Quat<double>(0.01596551667, -0.006985361948, -0.4382452548, 0.8986864879)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(-0.003396912202, -9.692178925, 0.01301318567), sofa::type::Quat<double>(0.01059102598, -0.01374254084, -0.7148386272, 0.6990741805)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(-0.1668556563, -9.577363026, 0.03455744119), sofa::type::Quat<double>(-0.02439727795, -0.04585925265, -0.9016493065, 0.4293369653)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(-0.230611987, -9.409244076, 0.05034655108), sofa::type::Quat<double>(-0.06676044546, -0.08462859852, -0.9839281746, 0.1423600732)));
    
    int pctTru = 0; 
    for (int step = 0; step < 140; step++)
    {
        sofa::simulation::node::animate(m_root.get());

        if (step % 10 == 0) 
        {
            const VecCoord& coords = meca->x.getValue();
            const Coord& truthC = truthCoords[pctTru];

            // test with groundtruth, do it index by index for better log
            // position
            EXPECT_FLOAT_EQ(coords[0][0], truthC[0]) << "Iteration " << step;
            EXPECT_FLOAT_EQ(coords[0][1], truthC[1]) << "Iteration " << step;
            EXPECT_FLOAT_EQ(coords[0][2], truthC[2]) << "Iteration " << step;

            // orientation
            EXPECT_FLOAT_EQ(coords[0][3], truthC[3]) << "Iteration " << step;
            EXPECT_FLOAT_EQ(coords[0][4], truthC[4]) << "Iteration " << step;
            EXPECT_FLOAT_EQ(coords[0][5], truthC[5]) << "Iteration " << step;
            EXPECT_FLOAT_EQ(coords[0][6], truthC[6]) << "Iteration " << step;

            pctTru++;
        }
    }

    return true;
}


bool LCPForceFeedback_test::test_Collision()
{
    loadTestScene("ToolvsFloorCollision_test.scn");

    simulation::Node::SPtr instruNode = m_root->getChild("Instrument");
    EXPECT_NE(instruNode, nullptr);
    MecaRig::SPtr meca = instruNode->get<MecaRig>(instruNode->SearchDown);
    m_LCPFFBack = instruNode->get<LCPRig>(instruNode->SearchDown);
    
    // Force only 2 iteration max for ci tests
    m_LCPFFBack->d_solverMaxIt.setValue(2);

    // Check components access
    EXPECT_NE(meca, nullptr);
    EXPECT_NE(m_LCPFFBack, nullptr);

    // Check meca size and init position
    EXPECT_EQ(meca->getSize(), 1);

    for (int step = 0; step < 100; step++)
    {
        sofa::simulation::node::animate(m_root.get());
    }

    const VecCoord& coords = meca->x.getValue();
    const MatrixDeriv& cons = meca->c.getValue();

    // check position and constraint problem
    EXPECT_LT(coords[0][1], -9.0);
    EXPECT_EQ(cons.size(), 84);

    // check LCP computeForce method
    sofa::type::Vec3 position;
    sofa::type::Vec3 force;
    sofa::type::Vec3 trueForce;

    // check out of problem position
    m_LCPFFBack->computeForce(position[0], position[1], position[2], 0, 0, 0, 0, force[0], force[1], force[2]);
    trueForce = sofa::type::Vec3(0.0, 0.0, 0.0);
    EXPECT_EQ(force, trueForce);
    
    
    // check position in contact
    m_LCPFFBack->computeForce(coords[0][0], coords[0][1], coords[0][2], 0, 0, 0, 0, force[0], force[1], force[2]);

    // test with groundtruth, do it index by index for better log
    Coord coordT = Coord(sofa::type::Vec3d(0.1083095508, -9.45640795, 0.01134330546), sofa::type::Quat<double>(0.01623300333, -0.006386979003, -0.408876291, 0.9124230788));
    //// position
    EXPECT_FLOAT_EQ(coords[0][0], coordT[0]);
    EXPECT_FLOAT_EQ(coords[0][1], coordT[1]);
    EXPECT_FLOAT_EQ(coords[0][2], coordT[2]);

    //// orientation
    EXPECT_FLOAT_EQ(coords[0][3], coordT[3]);
    EXPECT_FLOAT_EQ(coords[0][4], coordT[4]);
    EXPECT_FLOAT_EQ(coords[0][5], coordT[5]);
    EXPECT_FLOAT_EQ(coords[0][6], coordT[6]);

    //// force
    trueForce = sofa::type::Vec3(-0.001655988795, 0.002759984308, -2.431849862e-06);
    EXPECT_FLOAT_EQ(force[0], trueForce[0]);
    EXPECT_FLOAT_EQ(force[1], trueForce[1]);
    EXPECT_FLOAT_EQ(force[2], trueForce[2]);

    // check position inside collision
    Coord inside = Coord(sofa::type::Vec3d(coords[0][0], coords[0][1] - 1.0, coords[0][2]), sofa::type::Quat<double>(0.01623300333, -0.006386979003, -0.408876291, 0.9124230788));
    m_LCPFFBack->computeForce(inside[0], inside[1], inside[2], 0, 0, 0, 0, force[0], force[1], force[2]);

    // test with groundtruth, do it index by index for better log
    coordT = Coord(sofa::type::Vec3d(0.1083095508, -10.45640795, 0.01134330546), sofa::type::Quat<double>(0.01623300333, -0.006386979003, -0.408876291, 0.9124230788));
    //// position
    EXPECT_FLOAT_EQ(inside[0], coordT[0]);
    EXPECT_FLOAT_EQ(inside[1], coordT[1]);
    EXPECT_FLOAT_EQ(inside[2], coordT[2]);

    //// orientation
    EXPECT_FLOAT_EQ(inside[3], coordT[3]);
    EXPECT_FLOAT_EQ(inside[4], coordT[4]);
    EXPECT_FLOAT_EQ(inside[5], coordT[5]);
    EXPECT_FLOAT_EQ(inside[6], coordT[6]);

    //// force
    trueForce = sofa::type::Vec3(-0.1450155705, 8.930516304, 0.1567013005);
    EXPECT_FLOAT_EQ(force[0], trueForce[0]);
    EXPECT_FLOAT_EQ(force[1], trueForce[1]);
    EXPECT_FLOAT_EQ(force[2], trueForce[2]);

    // check rigidTypes computeForce method
    VecDeriv forces;
    m_LCPFFBack->computeForce(coords, forces);
         
    EXPECT_EQ(forces.size(), 1);
    EXPECT_FLOAT_EQ(forces[0][0], -0.00164953925);
    EXPECT_FLOAT_EQ(forces[0][1], 0.002749336856);
    EXPECT_FLOAT_EQ(forces[0][2], -1.032894327e-05);
    EXPECT_FLOAT_EQ(forces[0][3], 0.0001298280752);
    EXPECT_FLOAT_EQ(forces[0][4], 7.443984612e-05);
    EXPECT_FLOAT_EQ(forces[0][5], -0.0009855082698);

    return true;
}


bool LCPForceFeedback_test::test_multiThread()
{
    loadTestScene("ToolvsFloorCollision_test.scn");

    const simulation::Node::SPtr instruNode = m_root->getChild("Instrument");
    EXPECT_NE(instruNode, nullptr);
    const MecaRig::SPtr meca = instruNode->get<MecaRig>(instruNode->SearchDown);
    m_LCPFFBack = instruNode->get<LCPRig>(instruNode->SearchDown);
    
    // Force only 2 iteration max for ci tests
    m_LCPFFBack->d_solverMaxIt.setValue(2);

    // Check components access
    EXPECT_NE(meca, nullptr);
    EXPECT_NE(m_LCPFFBack, nullptr);

    // create and launch haptic thread
    m_terminate = false;
    haptic_thread = std::thread(HapticsThread, std::ref(this->m_terminate), this);

    // run simulation for n steps
    for (int step = 0; step < 500; step++)
    {
        sofa::simulation::node::animate(m_root.get());
        
        const VecCoord& coords = meca->x.getValue();        
        mtxPosition.lock();
        m_currentPosition[0] = coords[0][0];
        m_currentPosition[1] = coords[0][1];
        m_currentPosition[2] = coords[0][2];
        mtxPosition.unlock();
        CTime::sleep(0.001);
    }

    // stop thread
    m_terminate = true;
    haptic_thread.join();

    // get back info from haptic thread    
    m_meanForceFFBack = m_meanForceFFBack / m_cptLoopContact;

    EXPECT_GT(m_cptLoop, 500);
    EXPECT_GT(m_cptLoopContact, 400);

	// make a simple test FFBack not equal to 0. Not possible to test exact value as CI have different thread speed
	EXPECT_NE(m_meanForceFFBack[0], 0.0);
	EXPECT_NE(m_meanForceFFBack[1], 0.0);
	EXPECT_NE(m_meanForceFFBack[2], 0.0);

    return true;
}



TEST_F(LCPForceFeedback_test, test_InitScene)
{
    ASSERT_TRUE(test_InitScene());
}

TEST_F(LCPForceFeedback_test, test_SimpleCollision)
{
    ASSERT_TRUE(test_SimpleCollision());
}

TEST_F(LCPForceFeedback_test, test_Collision)
{
    ASSERT_TRUE(test_Collision());
}

TEST_F(LCPForceFeedback_test, test_multiThread)
{
    ASSERT_TRUE(test_multiThread());
}


} // namespace sofa
