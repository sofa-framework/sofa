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
using sofa::testing::BaseTest;

#include <sofa/simpleapi/SimpleApi.h>
using namespace sofa::simpleapi;

#include <sofa/component/solidmechanics/spring/RestShapeSpringsForceField.h>
#include <sofa/simulation/Node.h>

#include <sofa/component/statecontainer/MechanicalObject.h>
using sofa::component::statecontainer::MechanicalObject;

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/linearalgebra/FullMatrix.h>

using sofa::defaulttype::Vec3Types;
using sofa::defaulttype::Rigid3dTypes;
using sofa::helper::ReadAccessor;
using sofa::Data;

namespace
{
/// Adds the entries of the stiffness matrix of a force field (buildStiffnessMatrix) to a FullMatrix
struct StiffnessMatrixAccumulatorInto : public sofa::core::behavior::StiffnessMatrixAccumulator
{
    explicit StiffnessMatrixAccumulatorInto(sofa::linearalgebra::FullMatrix<SReal>& m) : matrix(m) {}
    void add(sofa::SignedIndex row, sofa::SignedIndex col, float value) override { matrix.add(row, col, value); }
    void add(sofa::SignedIndex row, sofa::SignedIndex col, double value) override { matrix.add(row, col, value); }

    sofa::linearalgebra::FullMatrix<SReal>& matrix;
};
}

/// Test suite for RestShapeSpringsForceField
class RestSpringsForceField_test : public BaseTest
{
public:
    ~RestSpringsForceField_test() override;
    sofa::simulation::Node::SPtr createScene(const std::string& type);

    template<class Type>
    void testDefaultBehavior(sofa::simulation::Node::SPtr root);

    template<class Type>
    void checkDifference(MechanicalObject<Type>& mo, bool isFixed);
};

RestSpringsForceField_test::~RestSpringsForceField_test()
{
}

sofa::simulation::Node::SPtr RestSpringsForceField_test::createScene(const std::string& type)
{
    const auto theSimulation = createSimulation();
    auto theRoot = createRootNode(theSimulation, "root");

    this->loadPlugins({
        Sofa.Component.IntegrationScheme.Backward,
        Sofa.Component.LinearSolver.Iterative,
        Sofa.Component.StateContainer,
        Sofa.Component.Mass,
        Sofa.Component.SolidMechanics.Spring
    });
    
    createObject(theRoot, "DefaultAnimationLoop");
    createObject(theRoot, "EulerImplicitIntegrationScheme");
    createObject(theRoot, "CGLinearSolver", {{ "iterations", "25" }, { "tolerance", "1e-5" }, {"threshold", "1e-5"}});

    /// Create an object with a mass and use a rest shape spring ff so it stays
    /// at the initial position
    const auto fixedObject = createChild(theRoot, "fixedObject");
    auto fixedObject_dofs = createObject(fixedObject, "MechanicalObject", {{"name","dofs"},
                                                                           {"size","10"},
                                                                           {"template",type}});
    createObject(fixedObject, "UniformMass", {{"totalMass", "1"}});

    createObject(fixedObject, "RestShapeSpringsForceField", {{"stiffness","1000"}});

    const auto movingObject = createChild(theRoot, "movingObject");
    auto movingObject_dofs =createObject(movingObject, "MechanicalObject", {{"name","dofs"},
                                                                            {"size","10"},
                                                                            {"template",type}});
    createObject(movingObject, "UniformMass", {{"totalMass", "1"}});

    sofa::simulation::node::initRoot(theRoot.get());
    for(unsigned int i=0;i<20;i++)
    {
        sofa::simulation::node::animate(theRoot.get(), 0.01_sreal);
    }
    return theRoot;
}

template<class Type>
void RestSpringsForceField_test::checkDifference(MechanicalObject<Type>& mo, bool isFixed)
{
    ReadAccessor< Data<typename Type::VecCoord> > positions = mo.x;
    ReadAccessor< Data<typename Type::VecCoord> > rest_positions = mo.x0;
    for(size_t i=0;i<positions.size();i++)
    {
        sofa::type::Vec3 pos = Type::getCPos(positions[i]) ;
        sofa::type::Vec3 rpos = Type::getCPos(rest_positions[i]) ;

        if(isFixed)
        {
            ASSERT_NEAR( pos.x(), rpos.x(), 0.1 );
            ASSERT_NEAR( pos.y(), rpos.y(), 0.1 );
            ASSERT_NEAR( pos.z(), rpos.z(), 0.1 );
        }
        else
        {
            ASSERT_LT( fabs(pos.x()-rpos.x()), 1 );
            ASSERT_LT( fabs(pos.y()-rpos.y()), 1 );
            ASSERT_LT( fabs(pos.z()-rpos.z()), 1 );
        }
    }
}

template<class Type>
void RestSpringsForceField_test::testDefaultBehavior(sofa::simulation::Node::SPtr root)
{
    auto fixedDofs = dynamic_cast<MechanicalObject<Type>*>(root->getChild("fixedObject")->getObject("dofs"));
    ASSERT_TRUE( fixedDofs != nullptr );

    auto movingDofs = dynamic_cast<MechanicalObject<Type>*>(root->getChild("movingObject")->getObject("dofs"));
    ASSERT_TRUE( movingDofs != nullptr );

    checkDifference(*fixedDofs, true);
    checkDifference(*movingDofs, false);
}


TEST_F(RestSpringsForceField_test, defaultBehaviorVec3)
{
    this->testDefaultBehavior<Vec3Types>(this->createScene("Vec3"));
}

TEST_F(RestSpringsForceField_test, defaultBehaviorRigid3)
{
    this->testDefaultBehavior<sofa::defaulttype::Rigid3Types>(this->createScene("Rigid3"));
}

/// The torque of the rigid springs is -angularStiffness * (rotation vector from the rest orientation),
/// also for rotations so small that the real part of the quaternion rounds to 1 (below ~3e-8 rad).
/// With a large angularStiffness, the equilibrium is in this range: a torque wrongly set to zero there
/// makes the Newton iterations stall. The stiffness (addDForce and buildStiffnessMatrix) is checked with
/// central finite differences, the orientation being perturbed as in RigidCoord::operator+=.
TEST_F(RestSpringsForceField_test, torqueAndStiffnessForSmallRotationsRigid3)
{
    using DataTypes = Rigid3dTypes;
    using Coord = DataTypes::Coord;
    using Deriv = DataTypes::Deriv;
    using VecCoord = DataTypes::VecCoord;
    using VecDeriv = DataTypes::VecDeriv;
    using Quat = sofa::type::Quat<SReal>;
    using sofa::type::Vec3;
    using RestShapeSprings = sofa::component::solidmechanics::spring::RestShapeSpringsForceField<DataTypes>;

    this->loadPlugins({Sofa.Component.StateContainer, Sofa.Component.SolidMechanics.Spring});

    const auto root = createRootNode(createSimulation(), "root");
    createObject(root, "DefaultAnimationLoop");
    createObject(root, "MechanicalObject", {{"name", "dofs"}, {"template", "Rigid3"}, {"size", "1"}});
    createObject(root, "RestShapeSpringsForceField", {{"name", "springs"}, {"stiffness", "3"}, {"angularStiffness", "7"}});
    sofa::simulation::node::initRoot(root.get());

    auto* dofs = dynamic_cast<MechanicalObject<DataTypes>*>(root->getObject("dofs"));
    ASSERT_NE(dofs, nullptr);
    auto* springs = dynamic_cast<RestShapeSprings*>(root->getObject("springs"));
    ASSERT_NE(springs, nullptr);

    sofa::core::MechanicalParams mparams;
    mparams.setKFactor(1);

    const auto setRestPosition = [&](const Coord& rest)
    {
        auto x0 = sofa::helper::getWriteOnlyAccessor(*dofs->write(sofa::core::vec_id::write_access::restPosition));
        x0[0] = rest;
    };
    const auto computeForce = [&](const Coord& x) -> Deriv
    {
        Data<VecCoord> position(VecCoord{x});
        Data<VecDeriv> velocity(VecDeriv{Deriv()});
        Data<VecDeriv> force(VecDeriv{Deriv()});
        springs->addForce(&mparams, force, position, velocity);
        return force.getValue()[0];
    };

    const Vec3 axis = Vec3(0.2, 1, -0.3).normalized();
    const std::vector<Coord> restPositions {
        Coord(Vec3(0, 0, 0), Quat::identity()),
        Coord(Vec3(1, -2, 0.5), Quat(Vec3(-0.5, 0.1, 0.8).normalized(), 0.8))
    };

    for (const Coord& rest : restPositions)
    {
        setRestPosition(rest);

        // torque for rotations from 1e-10 to 1e-3 rad
        for (const SReal angle : {1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-3})
        {
            Coord x = rest;
            x += Deriv(Vec3(), axis * angle);
            const Vec3 torque = computeForce(x).getVOrientation();
            for (sofa::Index i = 0; i < 3; ++i)
            {
                // relative tolerance, plus the round-off of the quaternion products (~1e-16 rad)
                EXPECT_NEAR(torque[i], -7 * angle * axis[i], 7 * (1e-6 * angle + 1e-15))
                    << "rotation of " << angle << " rad, component " << i;
            }
        }

        // stiffness: derivative of the force around a configuration rotated by 1e-9 rad from the rest
        Coord current = rest;
        current += Deriv(Vec3(1e-3, -2e-3, 1e-3), axis * 1e-9);
        {
            auto x = sofa::helper::getWriteOnlyAccessor(*dofs->write(sofa::core::vec_id::write_access::position));
            x[0] = current;
        }

        sofa::linearalgebra::FullMatrix<SReal> K(6, 6);
        K.clear();
        StiffnessMatrixAccumulatorInto accumulator(K);
        sofa::core::behavior::StiffnessMatrix stiffnessMatrix;
        stiffnessMatrix.setMatrixAccumulator(&accumulator, dofs);
        stiffnessMatrix.setMechanicalParams(&mparams);
        springs->buildStiffnessMatrix(&stiffnessMatrix);

        constexpr SReal delta = 1e-9;
        for (sofa::Index j = 0; j < 6; ++j)
        {
            Deriv dx;
            dx[j] = delta;
            Coord xPlus = current;
            xPlus += dx;
            Coord xMinus = current;
            xMinus += -dx;
            const Deriv finiteDifference = (computeForce(xPlus) - computeForce(xMinus)) / (2 * delta);

            Data<VecDeriv> dxData(VecDeriv{dx / delta});
            Data<VecDeriv> dfData(VecDeriv{Deriv()});
            springs->addDForce(&mparams, dfData, dxData);
            const Deriv df = dfData.getValue()[0];

            for (sofa::Index i = 0; i < 6; ++i)
            {
                EXPECT_NEAR(df[i], finiteDifference[i], 1e-5) << "addDForce, row " << i << ", column " << j;
                EXPECT_NEAR(K.element(i, j), finiteDifference[i], 1e-5) << "buildStiffnessMatrix, row " << i << ", column " << j;
            }
        }
    }
}
