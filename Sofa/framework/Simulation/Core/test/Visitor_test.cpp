#include <gtest/gtest.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace
{
/**
 * A visitor counting the number of force fields, mappings and masses visited.
 * This visitor considers interaction force fields as BaseForceField, so they are counted as force
 * fields.
 */
class TestVisitorWithoutInteractionForceField : public simulation::MechanicalVisitor
{
public:
    using simulation::MechanicalVisitor::MechanicalVisitor;

    Result fwdMechanicalMapping(simulation::Node* /*node*/, sofa::core::BaseMapping* map) override
    {
        ++m_mappingCount;
        return Result::RESULT_CONTINUE;
    }

    Result fwdForceField(simulation::Node*, sofa::core::behavior::BaseForceField*) override
    {
        ++m_forceFieldCount;
        return Result::RESULT_CONTINUE;
    }

    Result fwdMass(simulation::Node*, sofa::core::behavior::BaseMass*) override
    {
        ++m_massCount;
        return Result::RESULT_CONTINUE;
    }

    std::size_t m_forceFieldCount {};
    std::size_t m_interactionForceFieldCount {};
    std::size_t m_massCount {};
    std::size_t m_mappingCount {};
};

/**
 * A visitor counting the number of force fields, interaction force fields, mappings and masses visited.
 * This visitor considers interaction force fields as BaseInteractionForceField, so they are counted
 * as interaction force fields, not force fields.
 */
class TestVisitorWithInteractionForceField : public simulation::MechanicalVisitor
{
public:
    using simulation::MechanicalVisitor::MechanicalVisitor;

    Result fwdMechanicalMapping(simulation::Node*, sofa::core::BaseMapping* map) override
    {
        ++m_mappingCount;
        return Result::RESULT_CONTINUE;
    }

    Result fwdForceField(simulation::Node*, sofa::core::behavior::BaseForceField*) override
    {
        ++m_forceFieldCount;
        return Result::RESULT_CONTINUE;
    }

    Result fwdInteractionForceField(simulation::Node*, sofa::core::behavior::BaseInteractionForceField*) override
    {
        ++m_interactionForceFieldCount;
        return Result::RESULT_CONTINUE;
    }

    Result fwdMass(simulation::Node*, sofa::core::behavior::BaseMass*) override
    {
        ++m_massCount;
        return Result::RESULT_CONTINUE;
    }

    std::size_t m_forceFieldCount {};
    std::size_t m_interactionForceFieldCount {};
    std::size_t m_massCount {};
    std::size_t m_mappingCount {};
};

class TestForceField : public core::behavior::BaseForceField
{
public:
    void addForce(const core::MechanicalParams* mparams, core::MultiVecDerivId fId) override {}
    void addDForce(const core::MechanicalParams* mparams, core::MultiVecDerivId dfId) override {}
    SReal getPotentialEnergy(const core::MechanicalParams* mparams) const override { return {}; }
    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override {}
};

class TestMass : public core::behavior::BaseMass
{
public:
    void addMDx(const core::MechanicalParams* mparams, core::MultiVecDerivId fid, SReal factor) override {}
    void accFromF(const core::MechanicalParams* mparams, core::MultiVecDerivId aid) override {}
    void addGravityToV(const core::MechanicalParams* mparams, core::MultiVecDerivId vid) override {}
    SReal getKineticEnergy(const core::MechanicalParams* mparams) const override { return {}; }
    SReal getPotentialEnergy(const core::MechanicalParams* mparams) const override { return {}; }
    type::Vec6 getMomentum(const core::MechanicalParams* mparams) const override { return {}; }
    void addMToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override {}
    void initGnuplot(const std::string path) override {}
    void exportGnuplot(const core::MechanicalParams* mparams, SReal time) override {}
    SReal getElementMass(sofa::Index index) const override { return {}; }
    void getElementMass(sofa::Index index, linearalgebra::BaseMatrix* m) const override {}
    bool isDiagonal() const override { return {}; }
};

class TestInteractionForceField : public core::behavior::BaseInteractionForceField
{
public:
    void addForce(const core::MechanicalParams* mparams, core::MultiVecDerivId fId) override {}
    void addDForce(const core::MechanicalParams* mparams, core::MultiVecDerivId dfId) override {}
    SReal getPotentialEnergy(const core::MechanicalParams* mparams) const override { return {}; }
};

class TestMapping : public core::BaseMapping
{
public:
    void apply(const core::MechanicalParams* mparams, core::MultiVecCoordId outPos, core::ConstMultiVecCoordId inPos) override {}
    void applyJ(const core::MechanicalParams* mparams, core::MultiVecDerivId outVel, core::ConstMultiVecDerivId inVel) override {}
    type::vector<core::BaseState*> getFrom() override { return {} ;}
    type::vector<core::BaseState*> getTo() override { return {} ;}
    void applyJT(const core::MechanicalParams* mparams, core::MultiVecDerivId inForce, core::ConstMultiVecDerivId outForce) override {}
    void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId inForce, core::ConstMultiVecDerivId outForce) override {}
    void applyJT(const core::ConstraintParams* mparams, core::MultiMatrixDerivId inConst, core::ConstMultiMatrixDerivId outConst) override {}
    void computeAccFromMapping(const core::MechanicalParams* mparams, core::MultiVecDerivId outAcc, core::ConstMultiVecDerivId inVel, core::ConstMultiVecDerivId inAcc) override {}
    type::vector<core::behavior::BaseMechanicalState*> getMechFrom() override { return {} ;}
    type::vector<core::behavior::BaseMechanicalState*> getMechTo() override { return {} ;}
    void disable() override {}
};

sofa::simulation::Node::SPtr makeSceneGraph()
{
    const sofa::simulation::Node::SPtr root = sofa::simpleapi::createRootNode(sofa::simulation::getSimulation(), "root");

    const auto ff_0 = sofa::core::objectmodel::New<TestForceField>();
    root->addObject(ff_0);

    const auto mass_0 = sofa::core::objectmodel::New<TestMass>();
    root->addObject(mass_0);

    const auto iff_0 = sofa::core::objectmodel::New<TestInteractionForceField>();
    root->addObject(iff_0);

    const auto mapping_0 = sofa::core::objectmodel::New<TestMapping>();
    root->addObject(mapping_0);

    auto child_0 = root->createChild("child_0");

    // mapping_0 is in two different nodes
    child_0->addObject(mapping_0);

    auto child_1 = root->createChild("child_1");

    auto child_2 = child_1->createChild("child_2");
    // diamond graph
    child_0->addChild(child_2);

    // the two components are in a Node with two parents. We need to check that they are visited
    // only once
    const auto mass_1 = sofa::core::objectmodel::New<TestMass>();
    child_2->addObject(mass_1);

    const auto iff_1 = sofa::core::objectmodel::New<TestInteractionForceField>();
    child_2->addObject(iff_1);

    return root;
}

}

TEST(Visitor, ComplexGraph_visitorWithoutInteractionForceField)
{
    const auto root = makeSceneGraph();

    TestVisitorWithoutInteractionForceField visitor(
        sofa::core::MechanicalParams::defaultInstance());

    static constexpr bool precomputedOrder = false;
    root->executeVisitor(&visitor, precomputedOrder);

    // the visitor considers the interaction force field as a BaseForceField
    EXPECT_EQ(visitor.m_forceFieldCount, 3);
    EXPECT_EQ(visitor.m_interactionForceFieldCount, 0);
    EXPECT_EQ(visitor.m_massCount, 2);
    EXPECT_EQ(visitor.m_mappingCount, 1);
}

TEST(Visitor, ComplexGraph_visitorWithInteractionForceField)
{
    const auto root = makeSceneGraph();

    TestVisitorWithInteractionForceField visitor(sofa::core::MechanicalParams::defaultInstance());

    static constexpr bool precomputedOrder = false;
    root->executeVisitor(&visitor, precomputedOrder);

    EXPECT_EQ(visitor.m_forceFieldCount, 1);
    EXPECT_EQ(visitor.m_interactionForceFieldCount, 2);
    EXPECT_EQ(visitor.m_massCount, 2);
    EXPECT_EQ(visitor.m_mappingCount, 1);
}

}
