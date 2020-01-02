#include <string>
using std::string ;

#include <SofaTest/Sofa_test.h>
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;
using sofa::core::objectmodel::ComponentState;

#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>


class ClassA : public BaseObject
{
public:
    SOFA_CLASS(ClassA, BaseObject);
    
    sofa::Data<bool> input;
    sofa::Data<bool> output;

    ClassA()
        : Inherit1(),
          input(initData(&input, false, "in", "in")),
          output(initData(&output, "out", "out"))
    {
        addUpdateCallback("engineA", {&input}, [&]() -> ComponentState {
            std::cout << "in engineA" << std::endl;
            output.setValue(input.getValue());
            d_componentstate.setValue(sofa::core::objectmodel::ComponentState::Valid);
            e->cleanDirty();
        });
        engine.addOutput(&output);
        engine.addOutput(&d_componentstate);
    }

    ~ClassA() override {}
};

class ClassB : public BaseObject
{
public:
    SOFA_CLASS(ClassB, BaseObject);

    ClassB()
        : Inherit1(),
          inputLink(initDDGLink(this, "in", "help string")),
          output(initData(&output, "out", "out"))
    {
        addUpdateCallback("engineB", {&inputLink}, [&]() -> ComponentState {
            std::cout << "in engineB" << std::endl;
            output.setValue(inputLink.get()->output.getValue());
            d_componentstate.setValue(sofa::core::objectmodel::ComponentState::Valid);
            e->cleanDirty();
        });
        engine.addOutput(&output);
        engine.addOutput(&d_componentstate);
    }

    ~ClassB() override {}

    sofa::core::objectmodel::DDGLink<const ClassA> inputLink;
    sofa::Data<bool> output;
};



namespace sofa
{

struct DDGLink_test: public BaseTest
{
    ClassA::SPtr a;
    ClassB::SPtr b;
    Node::SPtr node;

    void SetUp() override
    {
        sofa::simulation::Simulation* simu;
        setSimulation(simu = new sofa::simulation::graph::DAGSimulation());

        node = simu->createNewGraph("root");

        a = sofa::core::objectmodel::New<ClassA>();
        a->setName("A");
        node->addObject(a);
        sofa::core::objectmodel::BaseObjectDescription bodA("A");
        bodA.setAttribute("in", "false");
        a->parse(&bodA);

        b = sofa::core::objectmodel::New<ClassB>();
        b->setName("B");
        node->addObject(b);
        sofa::core::objectmodel::BaseObjectDescription bodB("B");
        bodB.setAttribute("in", "@/A");
        bodB.setAttribute("out", "false");
        b->parse(&bodB);
    }

    void testGraphConsistency()
    {
        std::cout << "INITIAL STATE (everything but A::in should be dirty):" << std::endl;
        ASSERT_FALSE(a->input.isDirty());
        ASSERT_TRUE(a->output.isDirty());
        ASSERT_TRUE(a->d_componentstate.isDirty());
        ASSERT_TRUE(b->inputLink.isDirty());
        ASSERT_TRUE(b->output.isDirty());
        ASSERT_TRUE(b->d_componentstate.isDirty());

        b->output.getValue();
        std::cout << "\nAFTER accessing B::out (only B::componentState should be dirty):" << std::endl;
        ASSERT_FALSE(a->input.isDirty());
        ASSERT_FALSE(a->output.isDirty());
        ASSERT_FALSE(a->d_componentstate.isDirty());
        ASSERT_FALSE(b->inputLink.isDirty());
        ASSERT_FALSE(b->output.isDirty());
        ASSERT_TRUE(b->d_componentstate.isDirty());



        a->input.setValue(true); // Changing input value should dirtify all descendency...
        std::cout << "\nAFTER modifying A::in (should dirtify all but A::in):" << std::endl;
        ASSERT_FALSE(a->input.isDirty());
        ASSERT_TRUE(a->output.isDirty());
        ASSERT_TRUE(a->d_componentstate.isDirty());
        ASSERT_TRUE(b->inputLink.isDirty());
        ASSERT_TRUE(b->output.isDirty());
        ASSERT_TRUE(b->d_componentstate.isDirty());
    }


    void testDDGLink_methods()
    {

        ASSERT_TRUE(a.get() == b->inputLink.get());
        ASSERT_TRUE(b.get() == b->inputLink.getOwner());

        ClassA::SPtr c = sofa::core::objectmodel::New<ClassA>();
        c->setName("C");
        node->addObject(c);

        b->inputLink.set(c.get());
        ASSERT_TRUE(b->inputLink.get() == c.get());
    }

};

// Test
TEST_F(DDGLink_test, testGraphConsistency )
{
    this->testGraphConsistency();
}

TEST_F(DDGLink_test, testDDGLink_methods )
{
    this->testDDGLink_methods();
}
}  // namespace sofa

