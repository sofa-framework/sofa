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

#include <sofa/helper/testing/BaseTest.h>

#include <SofaSimulationGraph/DAGSimulation.h>

#include <SofaCommon/initSofaCommon.h>
#include <SofaBase/initSofaBase.h>
#include <SofaGeneral/initSofaGeneral.h>

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaHaptics/LCPForceFeedback.h>


namespace sofa 
{
using sofa::simulation::Simulation;
using sofa::simulation::Node;
using sofa::core::ExecParams;


//template <typename _DataTypes>
class LCPForceFeedback_test : public sofa::helper::testing::BaseTest
{
public:
    typedef sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3Types> MecaRig;
    typedef sofa::component::controller::LCPForceFeedback<sofa::defaulttype::Rigid3Types> LCPRig;
    typedef typename MecaRig::Coord    Coord;
    typedef typename MecaRig::VecCoord VecCoord;
    typedef typename MecaRig::MatrixDeriv MatrixDeriv;

    bool test_InitScene();

protected:
    void loadTestScene(const std::string& filename);

    Node::SPtr m_root;
};


void LCPForceFeedback_test::loadTestScene(const std::string& filename)
{
    sofa::component::initSofaBase();
    sofa::component::initSofaCommon();
    sofa::component::initSofaGeneral();

    simulation::Simulation* simu;
    sofa::simulation::setSimulation(simu = new sofa::simulation::graph::DAGSimulation());

    /// Load the scene
    std::string sceneFilename = std::string(SOFAHAPTICS_TEST_SCENES_DIR) + "/" + filename;
    m_root = simu->createNewGraph("root");    
    m_root = sofa::simulation::getSimulation()->load(sceneFilename.c_str());

    EXPECT_NE(m_root, nullptr);

    sofa::simulation::getSimulation()->init(m_root.get());
}

bool LCPForceFeedback_test::test_InitScene()
{
    loadTestScene("ToolvsFloorCollision_test.scn");

    simulation::Node::SPtr instruNode = m_root->getChild("Instrument");
    EXPECT_NE(instruNode, nullptr);
    MecaRig::SPtr meca = instruNode->get<MecaRig>(instruNode->SearchDown);
    LCPRig::SPtr lcp = instruNode->get<LCPRig>(instruNode->SearchDown);

    // Check components access
    EXPECT_NE(meca, nullptr);
    EXPECT_NE(lcp, nullptr);

    // Check meca size and init position
    EXPECT_EQ(meca->getSize(), 1);
    if (meca->getSize() > 0)
    {
        Coord rigZero;
        const VecCoord& coords = meca->x.getValue();
        EXPECT_EQ(coords[0], rigZero);
    }
    
    // check meca constraint, expect no cons in this world
    const MatrixDeriv& cons = meca->c.getValue();
    EXPECT_EQ(cons.size(), 0);

    return true;
}

TEST_F(LCPForceFeedback_test, test_InitScene)
{
    ASSERT_TRUE(test_InitScene());
}


} // namespace sofa
