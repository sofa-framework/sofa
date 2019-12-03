#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SofaBaseTopology/SparseGridTopology.h>

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository;

using sofa::core::objectmodel::New ;
using sofa::defaulttype::Vector3 ;
using namespace sofa::component::topology;
using namespace sofa::helper::testing;


struct SparseGridTopology_test : public BaseTest
{
    void SetUp() override
    {
        DataRepository.addFirstPath(FRAMEWORK_TEST_RESOURCES_DIR);
    }

    bool buildFromMeshFile();
    bool buildFromMeshParams();
};


bool SparseGridTopology_test::buildFromMeshFile()
{
    SparseGridTopology::SPtr sparseGrid1 = New<SparseGridTopology>();
    //ugly but is protected in the Base class, and no constructor
    static_cast<sofa::core::objectmodel::DataFileName*>(sparseGrid1->findData("filename"))->setValue("mesh/dragon.obj");
    EXPECT_NE(sparseGrid1, nullptr);

    sparseGrid1->setN({ 6, 5, 4 });
    sparseGrid1->init();

    EXPECT_EQ(sparseGrid1->getNbPoints(), 110);
    EXPECT_EQ(sparseGrid1->getNbHexahedra(), 50);
    EXPECT_NEAR(sparseGrid1->getPosX(0), -11.6815, 1e-04);
    EXPECT_NEAR(sparseGrid1->getPosY(0), -7.54611, 1e-04);
    EXPECT_NEAR(sparseGrid1->getPosZ(0), -5.14521, 1e-04);

    //SparseGridTopology::SPtr sparseGrid2 = New<SparseGridTopology>();
    //static_cast<sofa::core::objectmodel::DataFileName*>(sparseGrid2->findData("filename"))->setValue("mesh/suzanne.stl");
    //EXPECT_NE(sparseGrid2, nullptr);

    //sparseGrid2->setN({ 6, 5, 4 });
    //sparseGrid2->init();

    //EXPECT_EQ(sparseGrid2->getNbPoints(), 110);
    //EXPECT_EQ(sparseGrid2->getNbHexahedra(), 50);
    //EXPECT_NEAR(sparseGrid2->getPosX(0), -11.6815, 1e-04);
    //EXPECT_NEAR(sparseGrid2->getPosY(0), -7.54611, 1e-04);
    //EXPECT_NEAR(sparseGrid2->getPosZ(0), -5.14521, 1e-04);

    return true;
}

bool SparseGridTopology_test::buildFromMeshParams()
{
    
    return true;
}

TEST_F(SparseGridTopology_test, sparseGridCreation ) { ASSERT_TRUE(buildFromMeshFile()); }
TEST_F(SparseGridTopology_test, sparseGridPosition ) { ASSERT_TRUE(buildFromMeshParams()); }



