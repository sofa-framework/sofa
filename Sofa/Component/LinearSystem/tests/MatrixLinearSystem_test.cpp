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
#include <sofa/component/linearsystem/TypedMatrixLinearSystem.inl>
#include <sofa/component/linearsystem/MatrixLinearSystem.inl>
#include <sofa/linearalgebra/FullMatrix.h>

#include <sofa/testing/TestMessageHandler.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/Node.h>

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/solidmechanics/spring/SpringForceField.h>
#include <sofa/core/behavior/ForceField.h>

#include <sofa/core/behavior/MultiVec.h>
#include <sofa/testing/NumericTest.h>

TEST(LinearSystem, MatrixSystem_noContext)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using MatrixSystem = sofa::component::linearsystem::MatrixLinearSystem<MatrixType, sofa::linearalgebra::FullVector<SReal> >;
    const MatrixSystem::SPtr linearSystem = sofa::core::objectmodel::New<MatrixSystem>();
    EXPECT_NE(linearSystem, nullptr);

    EXPECT_TRUE(linearSystem->d_assembleStiffness.getValue());
    EXPECT_TRUE(linearSystem->d_assembleMass.getValue());
    EXPECT_TRUE(linearSystem->d_assembleGeometricStiffness.getValue());
    EXPECT_TRUE(linearSystem->d_applyProjectiveConstraints.getValue());
    EXPECT_TRUE(linearSystem->d_applyMappedComponents.getValue());
    EXPECT_FALSE(linearSystem->d_checkIndices.getValue());

    EXPECT_EQ(linearSystem->getMatrixSize(), sofa::type::Vec2u{});
    EXPECT_EQ(linearSystem->getSystemBaseMatrix(), nullptr);

    linearSystem->buildSystemMatrix(sofa::core::MechanicalParams::defaultInstance());

    EXPECT_EQ(linearSystem->getMatrixSize(), sofa::type::Vec2u{});
}


TEST(LinearSystem, MatrixSystem)
{
    const sofa::simulation::Node::SPtr root = sofa::core::objectmodel::New<sofa::simulation::Node>();

    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using MatrixSystem = sofa::component::linearsystem::MatrixLinearSystem<MatrixType, sofa::linearalgebra::FullVector<SReal> >;
    const MatrixSystem::SPtr linearSystem = sofa::core::objectmodel::New<MatrixSystem>();
    EXPECT_NE(linearSystem, nullptr);

    root->addObject(linearSystem);

    EXPECT_EQ(linearSystem->getMatrixSize(), sofa::type::Vec2u{});
    EXPECT_EQ(linearSystem->getSystemBaseMatrix(), nullptr);

    linearSystem->buildSystemMatrix(sofa::core::MechanicalParams::defaultInstance());

    EXPECT_EQ(linearSystem->getMatrixSize(), sofa::type::Vec2u{});


    const auto mstate = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    root->addObject(mstate);
    mstate->resize(10);

    linearSystem->buildSystemMatrix(sofa::core::MechanicalParams::defaultInstance());
    EXPECT_EQ(linearSystem->getMatrixSize(), sofa::type::Vec2u(mstate->getMatrixSize(), mstate->getMatrixSize()));
}

TEST(LinearSystem, MatrixSystem_springForceField)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    sofa::simulation::Node::SPtr root = sofa::core::objectmodel::New<sofa::simulation::Node>();

    using MatrixType = sofa::linearalgebra::FullMatrix<SReal>;
    using MatrixSystem = sofa::component::linearsystem::MatrixLinearSystem<MatrixType, sofa::linearalgebra::FullVector<SReal> >;
    MatrixSystem::SPtr linearSystem = sofa::core::objectmodel::New<MatrixSystem>();
    linearSystem->f_printLog.setValue(true);
    EXPECT_NE(linearSystem, nullptr);

    root->addObject(linearSystem);

    EXPECT_EQ(linearSystem->getMatrixSize(), sofa::type::Vec2u{});
    EXPECT_EQ(linearSystem->getSystemBaseMatrix(), nullptr);

    {
        EXPECT_MSG_EMIT(Error);
        linearSystem->buildSystemMatrix(sofa::core::MechanicalParams::defaultInstance());
    }

    EXPECT_MSG_NOEMIT(Error);
    EXPECT_EQ(linearSystem->getMatrixSize(), sofa::type::Vec2u{});

    //Create the Mechanical Object and define its positions
    auto mstate = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    mstate->setName("mstate");
    root->addObject(mstate);
    mstate->resize(2);
    auto writeAccessor = mstate->writePositions();
    writeAccessor[0] = {};
    writeAccessor[1] = sofa::type::Vec3{0, 0, 1};

    //Create a spring connecting two particles of the Mechanical Object
    auto spring = sofa::core::objectmodel::New<sofa::component::solidmechanics::spring::SpringForceField<sofa::defaulttype::Vec3Types> >();
    spring->setName(root->getNameHelper().resolveName(spring->getClassName(), sofa::core::ComponentNameHelper::Convention::xml));
    root->addObject(spring);
    const SReal springStiffness = 1_sreal;
    spring->addSpring(0, 1, springStiffness, 0_sreal, 0_sreal);

    auto mparams = *sofa::core::MechanicalParams::defaultInstance();
    mparams.setKFactor(1._sreal);

    root->init(&mparams);

    // Compute the external force. This step is mandatory because most of the time force fields
    // pre-computes required elements for the matrix assembly in the addForce method
    sofa::core::MultiVecDerivId ffId = sofa::core::vec_id::write_access::externalForce;
    ((sofa::core::behavior::BaseForceField*)spring.get())->addForce(&mparams, ffId, mparams.x(), mparams.v());


    // Finally build the system matrix, which is composed of only the stiffness matrix from the spring force field
    linearSystem->buildSystemMatrix(&mparams);
    EXPECT_EQ(linearSystem->getMatrixSize(), sofa::type::Vec2u(mstate->getMatrixSize(), mstate->getMatrixSize()));

    const MatrixType* matrix = linearSystem->getSystemMatrix();

    static constexpr SReal tolerance = 1e-18_sreal;

    // the matrix contains df/dx. It is a 6x6 matrix that can be divided into 4 blocks of 3x3:
    // [ df1/dx  df12/dx ]
    // [ df21/dx df2/dx  ]
    // with
    // df1/dx = df2/dx = -k
    // df12/dx = df21/dx = k

    for (MatrixType::Index p = 0; p < 2; ++p)
    {
        for (MatrixType::Index q = 0; q < 2; ++q)
        {
            const SReal sign = ( p == q ) ? 1_sreal : -1_sreal;
            for (MatrixType::Index i = 0; i < 3; ++i)
            {
                for (MatrixType::Index j = 0; j < 3; ++j)
                {
                    const SReal expectedElement = sign * (i == j) * -springStiffness;
                    EXPECT_NEAR(matrix->element(i + p*3, j + q*3), expectedElement, tolerance)
                        << "with i = " << i << ", j = " << j
                        << ", p = " << p << " and q = " << q;
                }
            }
        }
    }

    const auto slaves0 = spring->getSlaves();
    ASSERT_EQ(slaves0.size(), 2); //2 slaves: 1 for stiffness, 1 for damping

    EXPECT_TRUE(
    std::any_of(slaves0.begin(), slaves0.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::BaseAssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::STIFFNESS>* >(slave.get());})
    );
    EXPECT_TRUE(
    std::none_of(slaves0.begin(), slaves0.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::BaseAssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::MASS>* >(slave.get());})
    );
    EXPECT_TRUE(
    std::any_of(slaves0.begin(), slaves0.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::BaseAssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::DAMPING>* >(slave.get());})
    );

    EXPECT_TRUE(
    std::any_of(slaves0.begin(), slaves0.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::AssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::STIFFNESS>* >(slave.get());})
    );
    EXPECT_TRUE(
    std::none_of(slaves0.begin(), slaves0.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::AssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::MASS>* >(slave.get());})
    );
    EXPECT_TRUE(
    std::any_of(slaves0.begin(), slaves0.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::AssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::DAMPING>* >(slave.get());})
    );

    // build the system again but with checkIndices set to true
    linearSystem->d_checkIndices.setValue(true);
    linearSystem->buildSystemMatrix(&mparams);

    const auto slaves1 = spring->getSlaves();

    ASSERT_EQ(slaves1.size(), 2);

    EXPECT_TRUE(
    std::any_of(slaves1.begin(), slaves1.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::BaseAssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::STIFFNESS>* >(slave.get());})
    );
    EXPECT_TRUE(
    std::none_of(slaves1.begin(), slaves1.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::BaseAssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::MASS>* >(slave.get());})
    );
    EXPECT_TRUE(
    std::any_of(slaves1.begin(), slaves1.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::BaseAssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::DAMPING>* >(slave.get());})
    );

    EXPECT_TRUE(
    std::any_of(slaves1.begin(), slaves1.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::AssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::STIFFNESS, sofa::core::matrixaccumulator::RangeVerification>* >(slave.get());})
    );
    EXPECT_TRUE(
    std::none_of(slaves1.begin(), slaves1.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::AssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::MASS, sofa::core::matrixaccumulator::RangeVerification>* >(slave.get());})
    );
    EXPECT_TRUE(
    std::any_of(slaves1.begin(), slaves1.end(),
        [](const auto slave){ return dynamic_cast<sofa::component::linearsystem::AssemblingMatrixAccumulator<sofa::core::matrixaccumulator::Contribution::DAMPING, sofa::core::matrixaccumulator::RangeVerification>* >(slave.get());})
    );

}

template<class DataTypes>
    class BuggyForceField : public sofa::core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(BuggyForceField, sofa::core::behavior::ForceField<DataTypes>);

    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override
    {
        auto dfdx = matrix->getForceDerivativeIn(this->mstate).withRespectToPositionsIn(this->mstate);
        dfdx(10, 20) += 0.;
    }

    void addForce(const sofa::core::MechanicalParams*, typename Inherit1::DataVecDeriv& f, const typename Inherit1::DataVecCoord& x, const typename Inherit1::DataVecDeriv& v) override
    {
        SOFA_UNUSED(f);
        SOFA_UNUSED(x);
        SOFA_UNUSED(v);
    }
    void addDForce(const sofa::core::MechanicalParams* mparams, typename Inherit1::DataVecDeriv& df, const typename Inherit1::DataVecDeriv& dx ) override
    {
        SOFA_UNUSED(mparams);
        SOFA_UNUSED(df);
        SOFA_UNUSED(dx);
    }
    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const typename Inherit1::DataVecCoord& x) const override
    {
        SOFA_UNUSED(x);
        return 0._sreal;
    }
};

/// Empty matrix class with the interface of a BaseMatrix
/// The class does nothing on purpose. It has been introduced to test that a buggy force field
/// is detected, and it avoids a crash compared to a true matrix type.
class EmptyMatrix : public sofa::linearalgebra::BaseMatrix
{
public:
    using Real = SReal;
    ~EmptyMatrix() override = default;
    Index rowSize() const override
    {
        return 1;
    }
    Index colSize() const override
    {
        return 1;
    }
    SReal element(Index i, Index j) const override
    {
        SOFA_UNUSED(i);
        SOFA_UNUSED(j);
        return {};
    }
    void resize(Index nbRow, Index nbCol) override
    {
        SOFA_UNUSED(nbRow);
        SOFA_UNUSED(nbCol);
    }
    void clear() override
    {
    }
    void set(Index i, Index j, double v) override
    {
        SOFA_UNUSED(i);
        SOFA_UNUSED(j);
        SOFA_UNUSED(v);
    }
    void add(Index row, Index col, double v) override
    {
        SOFA_UNUSED(row);
        SOFA_UNUSED(col);
        SOFA_UNUSED(v);
        //add method is empty to prevent crashes in tests
    }
    void add(Index row, Index col, const sofa::type::Mat3x3d& _M) override
    {
        SOFA_UNUSED(row);
        SOFA_UNUSED(col);
        SOFA_UNUSED(_M);
    }
    void add(Index row, Index col, const sofa::type::Mat3x3f& _M) override
    {
        SOFA_UNUSED(row);
        SOFA_UNUSED(col);
        SOFA_UNUSED(_M);
    }
    static const char* Name() { return "EmptyMatrix"; }
};

TEST(LinearSystem, MatrixSystem_buggyForceField)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    const sofa::simulation::Node::SPtr root = sofa::core::objectmodel::New<sofa::simulation::Node>();

    using MatrixSystem = sofa::component::linearsystem::MatrixLinearSystem<EmptyMatrix, sofa::linearalgebra::FullVector<SReal> >;
    const MatrixSystem::SPtr linearSystem = sofa::core::objectmodel::New<MatrixSystem>();

    root->addObject(linearSystem);

    //Create the Mechanical Object and define its positions
    const auto mstate = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    root->addObject(mstate);
    mstate->resize(2);
    auto writeAccessor = mstate->writePositions();
    writeAccessor[0] = {};
    writeAccessor[1] = sofa::type::Vec3{0, 0, 1};

    const auto bug = sofa::core::objectmodel::New<BuggyForceField<sofa::defaulttype::Vec3Types> >();
    root->addObject(bug);

    auto mparams = *sofa::core::MechanicalParams::defaultInstance();
    mparams.setKFactor(1._sreal);

    root->init(&mparams);

    // Finally build the system matrix, which is composed of only the stiffness matrix from the spring force field
    {
        EXPECT_MSG_NOEMIT(Error);
        linearSystem->buildSystemMatrix(&mparams);
    }

    linearSystem->d_checkIndices.setValue(true);

    // build again but this time the indices are verified and an error is emitted
    {
        EXPECT_MSG_EMIT(Error);
        linearSystem->buildSystemMatrix(&mparams);
    }
}


/// Records the order in which the stiffness matrices of the force fields are built
template<class DataTypes>
class OrderRecordingForceField : public sofa::core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(OrderRecordingForceField, sofa::core::behavior::ForceField<DataTypes>);
    inline static std::vector<std::string> buildOrder;

    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override
    {
        buildOrder.push_back(this->getName());
        auto dfdx = matrix->getForceDerivativeIn(this->mstate).withRespectToPositionsIn(this->mstate);
        dfdx(0, 0) += 1.;
    }
    void addForce(const sofa::core::MechanicalParams*, typename Inherit1::DataVecDeriv&, const typename Inherit1::DataVecCoord&, const typename Inherit1::DataVecDeriv&) override {}
    void addDForce(const sofa::core::MechanicalParams*, typename Inherit1::DataVecDeriv&, const typename Inherit1::DataVecDeriv&) override {}
    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const typename Inherit1::DataVecCoord&) const override { return 0; }
};

/// The force fields contributing to the same entries of the global matrix are assembled in an order which does not
/// depend on their addresses (the rounding of the sums would change from one run to another): here, in the order of
/// their paths, while they are created in the reverse order
TEST(LinearSystem, MatrixSystem_contributionsInAnOrderIndependentOfTheAddresses)
{
    using FF = OrderRecordingForceField<sofa::defaulttype::Vec3Types>;
    const sofa::simulation::Node::SPtr root = sofa::core::objectmodel::New<sofa::simulation::Node>();
    using MatrixSystem = sofa::component::linearsystem::MatrixLinearSystem<sofa::linearalgebra::FullMatrix<SReal>, sofa::linearalgebra::FullVector<SReal> >;
    const MatrixSystem::SPtr linearSystem = sofa::core::objectmodel::New<MatrixSystem>();
    root->addObject(linearSystem);
    const auto mstate = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    root->addObject(mstate);
    mstate->resize(2);

    std::vector<FF::SPtr> forceFields;
    for (int i = 7; i >= 0; --i)
    {
        forceFields.push_back(sofa::core::objectmodel::New<FF>());
        forceFields.back()->setName("forceField" + std::to_string(i));
        root->addObject(forceFields.back());
    }

    auto mparams = *sofa::core::MechanicalParams::defaultInstance();
    mparams.setKFactor(1._sreal);
    root->init(&mparams);

    FF::buildOrder.clear();
    linearSystem->buildSystemMatrix(&mparams);
    const std::vector<std::string> expected { "forceField0", "forceField1", "forceField2", "forceField3",
                                              "forceField4", "forceField5", "forceField6", "forceField7" };
    EXPECT_EQ(FF::buildOrder, expected);
}

/// Adds its value to the entry (0, 0) of the stiffness matrix
template<class DataTypes>
class ValueForceField : public sofa::core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(ValueForceField, sofa::core::behavior::ForceField<DataTypes>);
    SReal value {};

    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override
    {
        auto dfdx = matrix->getForceDerivativeIn(this->mstate).withRespectToPositionsIn(this->mstate);
        dfdx(0, 0) += value;
    }
    void addForce(const sofa::core::MechanicalParams*, typename Inherit1::DataVecDeriv&, const typename Inherit1::DataVecCoord&, const typename Inherit1::DataVecDeriv&) override {}
    void addDForce(const sofa::core::MechanicalParams*, typename Inherit1::DataVecDeriv&, const typename Inherit1::DataVecDeriv&) override {}
    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const typename Inherit1::DataVecCoord&) const override { return 0; }
};

/// The same scene gives the same global matrix, whatever the order in which its components are allocated: three
/// force fields add 1, 1e16 and -1e16 to the same entry, whose floating-point sum is 0 or 1 depending on the order of
/// the additions. The scene (same components, names and graph order) is built twice, with the force fields allocated
/// in two opposite orders: the assembled entries must be identical
TEST(LinearSystem, MatrixSystem_sameMatrixWhateverTheAddressesOfTheComponents)
{
    using FF = ValueForceField<sofa::defaulttype::Vec3Types>;
    using MatrixSystem = sofa::component::linearsystem::MatrixLinearSystem<sofa::linearalgebra::FullMatrix<SReal>, sofa::linearalgebra::FullVector<SReal> >;

    const auto assembledEntry = [](bool reverseAllocation)
    {
        const sofa::simulation::Node::SPtr root = sofa::core::objectmodel::New<sofa::simulation::Node>();
        const MatrixSystem::SPtr linearSystem = sofa::core::objectmodel::New<MatrixSystem>();
        root->addObject(linearSystem);
        const auto mstate = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
        root->addObject(mstate);
        mstate->resize(1);

        const std::vector<std::pair<std::string, SReal>> forceFields { {"a", 1.}, {"b", 1e16}, {"c", -1e16} };
        std::vector<FF::SPtr> allocated(forceFields.size());
        for (std::size_t k = 0; k < forceFields.size(); ++k)
        {
            const std::size_t i = reverseAllocation ? forceFields.size() - 1 - k : k;
            allocated[i] = sofa::core::objectmodel::New<FF>();
        }
        for (std::size_t i = 0; i < forceFields.size(); ++i)
        {
            allocated[i]->setName(forceFields[i].first);
            allocated[i]->value = forceFields[i].second;
            root->addObject(allocated[i]);
        }

        auto mparams = *sofa::core::MechanicalParams::defaultInstance();
        mparams.setKFactor(1._sreal);
        root->init(&mparams);
        linearSystem->buildSystemMatrix(&mparams);
        return linearSystem->getSystemMatrix()->element(0, 0);
    };

    EXPECT_EQ(assembledEntry(false), assembledEntry(true));
}
