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
#include <sofa/component/solidmechanics/fem/elastic/TractionSourceTerm.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
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

// The non-constant (traction) path needs Vec3: the pressure/traction load is expressed
// through the triangle's normal, which only exists in 3D.
using DataTypes3 = defaulttype::Vec3Types;
using Integrator3 = component::solidmechanics::fem::elastic::FEMSourceTermIntegrator<DataTypes3, geometry::Triangle>;
using Traction = component::solidmechanics::fem::elastic::TractionSourceTerm<DataTypes3, geometry::Triangle>;
using DOF3 = component::statecontainer::MechanicalObject<DataTypes3>;
using VecCoord3 = DataTypes3::VecCoord;
using VecDeriv3 = DataTypes3::VecDeriv;
using Coord3 = DataTypes3::Coord;
using Deriv3 = DataTypes3::Deriv;

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

    simulation::Node::SPtr makeTractionMesh(Integrator3*& integrator, Traction*& load)
    {
        this->loadPlugins({"Sofa.Component.StateContainer",
            "Sofa.Component.Topology.Container.Constant", "Sofa.Component.SolidMechanics.FEM.Elastic"});

        auto root = createRootNode(m_simulation, "root");
        createObject(root, "MechanicalObject", {{"template", "Vec3"}, {"position", "0 0 0  1 0 0  1 1 0  0 1 0"}});
        createObject(root, "MeshTopology", {{"name", "mesh"}, {"triangles", "0 1 2  0 2 3"}});

        load = dynamic_cast<Traction*>(createObject(root, "TractionSourceTerm",
            {{"name", "load"}, {"template", "Vec3,Triangle"}, {"pressure", "1000"}}).get());
        integrator = dynamic_cast<Integrator3*>(createObject(root, "FEMSourceTermIntegrator",
            {{"name", "traction"}, {"template", "Vec3,Triangle"}, {"topology", "@mesh"},
             {"quadratureDegree", "1"}, {"nonConstantSources", "@load"}}).get());

        return root;
    }

    // Minimal dense accumulator: enough for the tiny meshes these tests use.
    struct DenseStiffnessAccumulator : public core::behavior::StiffnessMatrixAccumulator
    {
        explicit DenseStiffnessAccumulator(sofa::Size size) : K(size, sofa::type::vector<SReal>(size, 0.0)) {}

        void add(sofa::SignedIndex row, sofa::SignedIndex col, const sofa::type::Mat<3, 3, double>& value) override
        {
            for (sofa::Size i = 0; i < 3; ++i)
                for (sofa::Size j = 0; j < 3; ++j)
                    K[row + i][col + j] += value(i, j);
        }

        sofa::type::vector<sofa::type::vector<SReal>> K;
    };
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

// addDForce is the analytic tangent of a non-constant source (TractionSourceTerm). Check it
// against a central finite difference of addForce itself, taken at the same configuration.
TEST_F(FEMSourceTermIntegrator_test, TractionAddDForceMatchesFiniteDifference)
{
    Integrator3* integrator = nullptr;
    Traction* load = nullptr;
    m_root = makeTractionMesh(integrator, load);
    simulation::node::initRoot(m_root.get());
    ASSERT_NE(integrator, nullptr);
    ASSERT_NE(load, nullptr);

    const std::size_t n = m_root->get<DOF3>()->getSize();
    VecCoord3 x0(n);
    testing::copyFromData(x0, m_root->get<DOF3>()->readPositions());

    VecDeriv3 dx(n);
    dx[0] = Deriv3(0.3, -0.2, 0.1);
    dx[1] = Deriv3(-0.1, 0.4, -0.3);
    dx[2] = Deriv3(0.2, 0.2, 0.2);
    dx[3] = Deriv3(-0.2, 0.1, 0.3);

    const SReal h = 1e-6;
    core::MechanicalParams mparams;
    mparams.setKFactor(1.0);

    const auto evaluateForceAt = [&](SReal sign)
    {
        VecCoord3 x(n);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = x0[i] + dx[i] * (sign * h);

        Data<VecCoord3> xData;
        xData.setValue(x);
        Data<VecDeriv3> vData;
        Data<VecDeriv3> fData;
        fData.setValue(VecDeriv3(n));
        integrator->addForce(&mparams, fData, xData, vData);
        return fData.getValue();
    };

    const VecDeriv3 fPlus = evaluateForceAt(1.0);
    const VecDeriv3 fMinus = evaluateForceAt(-1.0);

    VecDeriv3 finiteDifference(n);
    for (std::size_t i = 0; i < finiteDifference.size(); ++i)
        finiteDifference[i] = (fPlus[i] - fMinus[i]) / (2 * h);

    Data<VecDeriv3> dfData;
    dfData.setValue(VecDeriv3(n));
    Data<VecDeriv3> dxData;
    dxData.setValue(dx);
    integrator->addDForce(&mparams, dfData, dxData);
    const VecDeriv3 df = dfData.getValue();

    for (std::size_t i = 0; i < df.size(); ++i)
        for (unsigned d = 0; d < 3; ++d)
            EXPECT_NEAR(df[i][d], finiteDifference[i][d], 1e-7)
                << "node " << i << ", component " << d;
}

// buildStiffnessMatrix and addDForce are two independently-written paths over the same
// per-element tangent (NonConstantSourceTerm::evaluateStiffness); they must agree exactly.
TEST_F(FEMSourceTermIntegrator_test, TractionBuildStiffnessMatrixMatchesAddDForce)
{
    Integrator3* integrator = nullptr;
    Traction* load = nullptr;
    m_root = makeTractionMesh(integrator, load);
    simulation::node::initRoot(m_root.get());
    ASSERT_NE(integrator, nullptr);
    ASSERT_NE(load, nullptr);

    auto* dof = m_root->get<DOF3>();
    const std::size_t n = dof->getSize();

    VecDeriv3 dx(n);
    dx[0] = Deriv3(0.3, -0.2, 0.1);
    dx[1] = Deriv3(-0.1, 0.4, -0.3);
    dx[2] = Deriv3(0.2, 0.2, 0.2);
    dx[3] = Deriv3(-0.2, 0.1, 0.3);

    core::MechanicalParams mparams;
    mparams.setKFactor(1.0);

    Data<VecDeriv3> dfData;
    dfData.setValue(VecDeriv3(n));
    Data<VecDeriv3> dxData;
    dxData.setValue(dx);
    integrator->addDForce(&mparams, dfData, dxData);
    const VecDeriv3 dfFromAddDForce = dfData.getValue();

    DenseStiffnessAccumulator accumulator(n * 3);
    core::behavior::StiffnessMatrix stiffnessMatrix;
    stiffnessMatrix.setMatrixAccumulator(&accumulator, dof);
    stiffnessMatrix.setMechanicalParams(&mparams);
    integrator->buildStiffnessMatrix(&stiffnessMatrix);

    VecDeriv3 dfFromK(n);
    for (std::size_t a = 0; a < n; ++a)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            for (unsigned row = 0; row < 3; ++row)
            {
                SReal sum = 0;
                for (unsigned col = 0; col < 3; ++col)
                    sum += accumulator.K[a * 3 + row][j * 3 + col] * dx[j][col];
                dfFromK[a][row] += sum;
            }
        }
    }

    for (std::size_t i = 0; i < dfFromK.size(); ++i)
        for (unsigned d = 0; d < 3; ++d)
            EXPECT_NEAR(dfFromK[i][d], dfFromAddDForce[i][d], 1e-13)
                << "node " << i << ", component " << d;
}

}  // namespace sofa
