#include <SofaTest/Sofa_test.h>
#include <sofa/helper/BackTrace.h>

#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>

#include <SofaMisc/AddResourceRepository.h>


namespace sofa
{

template <typename _DataTypes>
struct AddResourceRepository_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;

    sofa::simulation::Simulation* m_simu;
    sofa::simulation::Node::SPtr m_node;
    sofa::component::misc::AddResourceRepository::SPtr m_addrepo;

    void SetUp()
    {
        setSimulation(m_simu = new sofa::simulation::graph::DAGSimulation());
        m_node = m_simu->createNewGraph("root");
        m_addrepo = sofa::core::objectmodel::New<component::misc::AddResourceRepository>() ;
        m_addrepo->d_repositoryPath.setValue("");
        m_addrepo->init() ;

        m_node->addObject(m_addrepo) ;
    }

    void normalTests()
    {

        return ;
    }

};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(AddResourceRepository_test, DataTypes);

TYPED_TEST(AddResourceRepository_test, NormalBehavior) {
    ASSERT_NO_THROW(this->normalTests()) ;
}


}
