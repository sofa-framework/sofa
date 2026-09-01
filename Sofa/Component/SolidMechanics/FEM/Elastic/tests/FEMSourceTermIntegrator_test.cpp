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
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/testing/BaseTest.h>
#include <sofa/testing/NumericTest.h>
#include <sofa/core/MechanicalParams.h>

#include <sofa/component/solidmechanics/fem/elastic/FEMSourceTermIntegrator.h>
#include <sofa/component/solidmechanics/fem/elastic/ConstantSourceTerm.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/geometry/Triangle.h>

namespace sofa
{
using sofa::simpleapi::createObject;
using sofa::simpleapi::createRootNode;
using DataTypes = defaulttype::Vec2Types;
using Integrator = component::solidmechanics::fem::elastic::FEMSourceTermIntegrator<DataTypes, geometry::Triangle>;
using DOF = component::statecontainer::MechanicalObject<DataTypes>;
using VecCoord = DataTypes::VecCoord;
using VecDeriv = DataTypes::VecDeriv;

class FEMSourceTermIntegrator_test : public testing::BaseTest
{
protected:
    simulation::Simulation* m_simulation = nullptr;
    simulation::Node::SPtr m_root;

    void doSetUp() override
    {
        m_simulation = sofa::simulation::getSimulation();
    }

    void doTearDown() override
    {
        if (m_root != nullptr)
            sofa::simulation::node::unload(m_root);
    }

    simulation::Node::SPtr makeMesh()
    {
        this->loadPlugins({"Sofa.Component.StateContainer",
            "Sofa.Component.Topology.Container.Constant", "Sofa.Component.SolidMechanics.FEM.Elastic"});

        auto root = createRootNode(m_simulation, "root");
        createObject(root, "MechanicalObject", {{"template", "Vec2"}, {"position", "0 0  1 0  1 1  0 1"}});
        createObject(root, "MeshTopology", {{"name", "mesh"}, {"triangles", "0 1 2  0 2 3"}});
        return root;
    }

    static VecDeriv addForce(Integrator* integrator)
    {
        core::MechanicalParams mparams;
        Data<VecDeriv> f;
        f.setValue(VecDeriv(4));
        Data<VecCoord> x;
        Data<VecDeriv> v;
        integrator->addForce(&mparams, f, x, v);
        return f.getValue();
    }
};

// Splitting one ConstantSourceTerm into several must not change the integrated force.
TEST_F(FEMSourceTermIntegrator_test, MultipleSourcesSumToOne)
{
    m_root = makeMesh();

    createObject(m_root, "ConstantSourceTerm", {{"name", "full"}, {"template", "Vec2"}, {"property", "300 -600"}});
    auto* one = dynamic_cast<Integrator*>(createObject(m_root, "FEMSourceTermIntegrator",
        {{"name", "one"}, {"template", "Vec2,Triangle"}, {"topology", "@mesh"}, {"constantSources", "@full"}}).get());

    createObject(m_root, "ConstantSourceTerm", {{"name", "a"}, {"template", "Vec2"}, {"property", "100 -200"}});
    createObject(m_root, "ConstantSourceTerm", {{"name", "b"}, {"template", "Vec2"}, {"property", "100 -200"}});
    createObject(m_root, "ConstantSourceTerm", {{"name", "c"}, {"template", "Vec2"}, {"property", "100 -200"}});
    auto* three = dynamic_cast<Integrator*>(createObject(m_root, "FEMSourceTermIntegrator",
        {{"name", "three"}, {"template", "Vec2,Triangle"}, {"topology", "@mesh"}, {"constantSources", "@a @b @c"}}).get());

    simulation::node::initRoot(m_root.get());
    ASSERT_NE(one, nullptr);
    ASSERT_NE(three, nullptr);

    const VecDeriv fOne = addForce(one);
    const VecDeriv fThree = addForce(three);
    ASSERT_EQ(fOne.size(), fThree.size());
    for (std::size_t i = 0; i < fOne.size(); ++i)
        EXPECT_EQ(fOne[i], fThree[i]);
}

// dE = -dX . F, the conservative-force identity getPotentialEnergy relies on.
TEST_F(FEMSourceTermIntegrator_test, PotentialEnergyMatchesWork)
{
    m_root = makeMesh();

    createObject(m_root, "ConstantSourceTerm", {{"name", "bodyForce"}, {"template", "Vec2"}, {"property", "300 -600"}});
    auto* integrator = dynamic_cast<Integrator*>(createObject(m_root, "FEMSourceTermIntegrator",
        {{"name", "source"}, {"template", "Vec2,Triangle"}, {"topology", "@mesh"}, {"constantSources", "@bodyForce"}}).get());

    simulation::node::initRoot(m_root.get());
    ASSERT_NE(integrator, nullptr);

    const VecDeriv f = addForce(integrator);

    VecCoord x0(4);
    testing::copyFromData(x0, m_root->get<DOF>()->readPositions());

    VecCoord x1 = x0;
    for (auto& xi : x1)
        xi += DataTypes::Coord(0.01, -0.02);

    core::MechanicalParams mparams;
    Data<VecCoord> x0Data;
    x0Data.setValue(x0);
    Data<VecCoord> x1Data;
    x1Data.setValue(x1);

    const SReal e0 = integrator->getPotentialEnergy(&mparams, x0Data);
    const SReal e1 = integrator->getPotentialEnergy(&mparams, x1Data);

    SReal work = 0;
    for (std::size_t i = 0; i < f.size(); ++i)
        work += dot(f[i], x1[i] - x0[i]);

    EXPECT_NEAR(e1 - e0, -work, 1e-9);
}

}  // namespace sofa
