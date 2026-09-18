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
#include <sofa/simulation/Simulation.h>
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
    simulation::Simulation* simu = sofa::simulation::MainSimulation::getSimulation();
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
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -0.0024875621, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -0.16148664, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -0.55601037, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -1.1746, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -2.0063543, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -3.0409024, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -4.2683778, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -5.6793942, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -7.2650223, 0), sofa::type::Quat<double>(0, 0, 0, 1)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0, -9.0167665, 0), sofa::type::Quat<double>(0,0,0,1)));

    truthCoords.push_back(Coord(sofa::type::Vec3d(0.081810147, -9.3104954, 0.0084159356), sofa::type::Quat<double>(0.0175437, -0.0027353484, -0.22570916, 0.97403294)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(0.080605865, -9.5144129, 0.0084154662), sofa::type::Quat<double>(0.015405586, -0.0075208684, -0.47918597, 0.87754595)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(-0.038712446, -9.6903, 0.0090729725), sofa::type::Quat<double>(0.011461957, -0.012498143, -0.71390605, 0.70003611)));
    truthCoords.push_back(Coord(sofa::type::Vec3d(-0.17837876, -9.6024733, 0.020258242), sofa::type::Quat<double>(-0.0072201439, -0.031115066, -0.87544507, 0.48226097)));
    
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
    EXPECT_EQ(cons.size(), 105);

    // check LCP computeForce method
    sofa::type::Vec3 position = sofa::type::Vec3(0, 0, 0);
    sofa::type::Vec3 force = sofa::type::Vec3(0, 0, 0);
    sofa::type::Vec3 trueForce = sofa::type::Vec3(0, 0, 0);

    // check out of problem position
    m_LCPFFBack->computeForce(position[0], position[1], position[2], 0, 0, 0, 0, force[0], force[1], force[2]);
    trueForce = sofa::type::Vec3(0.0, 0.0, 0.0);
    EXPECT_EQ(force, trueForce);
    
    
    // check position in contact
    m_LCPFFBack->computeForce(coords[0][0], coords[0][1], coords[0][2], 0, 0, 0, 0, force[0], force[1], force[2]);

    // test with groundtruth, do it index by index for better log
    Coord coordT = Coord(sofa::type::Vec3d(0.07618425, -9.2916698, 0.0084328074), sofa::type::Quat<double>(0.017680431, -0.0022677642, -0.20060691, 0.97950965));
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
    trueForce = sofa::type::Vec3(-0.00084725959, 0.0024373089, -4.2111449e-05);
    EXPECT_FLOAT_EQ(force[0], trueForce[0]);
    EXPECT_FLOAT_EQ(force[1], trueForce[1]);
    EXPECT_FLOAT_EQ(force[2], trueForce[2]);

    // check position inside collision
    Coord inside = Coord(sofa::type::Vec3d(coords[0][0], coords[0][1] - 1.0, coords[0][2]), sofa::type::Quat<double>(0.01623300333, -0.006386979003, -0.408876291, 0.9124230788));
    m_LCPFFBack->computeForce(inside[0], inside[1], inside[2], 0, 0, 0, 0, force[0], force[1], force[2]);

    // test with groundtruth, do it index by index for better log
    coordT = Coord(sofa::type::Vec3d(0.07618425, -10.29167, 0.0084328074), sofa::type::Quat<double>(0.01623300333, -0.006386979003, -0.408876291, 0.9124230788));
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
    trueForce = sofa::type::Vec3(0.27008709, 9.1463537, 0.060468301);
    EXPECT_FLOAT_EQ(force[0], trueForce[0]);
    EXPECT_FLOAT_EQ(force[1], trueForce[1]);
    EXPECT_FLOAT_EQ(force[2], trueForce[2]);

    // check rigidTypes computeForce method
    VecDeriv forces;
    m_LCPFFBack->computeForce(coords, forces);
         
    EXPECT_EQ(forces.size(), 1);
    EXPECT_FLOAT_EQ(forces[0][0], -0.00013606942);
    EXPECT_FLOAT_EQ(forces[0][1], 0.0027710579);
    EXPECT_FLOAT_EQ(forces[0][2], -0.00090467848);
    EXPECT_FLOAT_EQ(forces[0][3], 0.00030387595);
    EXPECT_FLOAT_EQ(forces[0][4], -0.00031411531);
    EXPECT_FLOAT_EQ(forces[0][5], -0.0010078497);

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
    int nbSimuSteps = 100;
    for (int step = 0; step < nbSimuSteps; step++)
    {
        sofa::simulation::node::animate(m_root.get());
        
        const VecCoord& coords = meca->x.getValue();        
        mtxPosition.lock();
        m_currentPosition[0] = coords[0][0];
        m_currentPosition[1] = coords[0][1];
        m_currentPosition[2] = coords[0][2];
        mtxPosition.unlock();
        CTime::sleep(0.01); // wait more time between simulation step to let the LCP runs multiple loops
    }

    // stop thread
    m_terminate = true;
    haptic_thread.join();

    // get back info from haptic thread    
    m_meanForceFFBack = m_meanForceFFBack / float(m_cptLoopContact);

    EXPECT_GT(m_cptLoop, nbSimuSteps); // we assume the LCP runs faster than the current simulation speed
    EXPECT_GT(m_cptLoopContact, 0); // check that the simulation reached collision between instrument and floor

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

