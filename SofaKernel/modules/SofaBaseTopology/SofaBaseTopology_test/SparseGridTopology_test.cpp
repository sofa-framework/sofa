#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SofaBaseTopology/SparseGridTopology.h>

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository;

#ifdef SOFABASETOPOLOGY_TEST_WITH_COMMON_LOADERS
#include <SofaLoader/MeshObjLoader.h>
using sofa::component::loader::MeshObjLoader;
#endif

#ifdef SOFABASETOPOLOGY_TEST_WITH_GENERAL_LOADERS
#include <SofaGeneralLoader/MeshSTLLoader.h>
using sofa::component::loader::MeshSTLLoader;
#endif

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
    EXPECT_NE(sparseGrid1, nullptr);

    EXPECT_MSG_EMIT(Error); // STL is not supported
    
    //ugly but is protected in the Base class, and no constructor
    sofa::core::objectmodel::DataFileName* dataFilename = static_cast<sofa::core::objectmodel::DataFileName*>(sparseGrid1->findData("filename"));
    EXPECT_NE(dataFilename, nullptr);

    dataFilename->setValue("mesh/suzanne.stl");
    sparseGrid1->setN({ 6, 5, 4 });
    sparseGrid1->init();
    EXPECT_EQ(sparseGrid1->getNbPoints(), 0);
    EXPECT_EQ(sparseGrid1->getNbHexahedra(), 0);

    // Real case
    EXPECT_MSG_NOEMIT(Error);
    SparseGridTopology::SPtr sparseGrid2 = New<SparseGridTopology>();
    EXPECT_NE(sparseGrid2, nullptr);
    dataFilename = static_cast<sofa::core::objectmodel::DataFileName*>(sparseGrid2->findData("filename"));
    EXPECT_NE(dataFilename, nullptr);
    dataFilename->setValue("mesh/dragon.OBJ");
    sparseGrid2->setN({ 6, 5, 4 });
    sparseGrid2->init();

    EXPECT_EQ(sparseGrid2->getNbPoints(), 110);
    EXPECT_EQ(sparseGrid2->getNbHexahedra(), 50);
    EXPECT_NEAR(sparseGrid2->getPosX(0), -11.6815, 1e-04);
    EXPECT_NEAR(sparseGrid2->getPosY(0), -7.54611, 1e-04);
    EXPECT_NEAR(sparseGrid2->getPosZ(0), -5.14521, 1e-04);

    return true;
}

bool SparseGridTopology_test::buildFromMeshParams()
{
    SparseGridTopology::SPtr sparseGrid1 = New<SparseGridTopology>();
    EXPECT_NE(sparseGrid1, nullptr);

    //Pyramid centered on 0 0 0
    sparseGrid1->seqPoints.setValue({ {0, 0, 1}, {-1, 0, -1}, {0, 1, -1}, {1, 0, -1}, {0, -1, -1} });
    sparseGrid1->seqTriangles.setValue({ {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}, {1, 2, 3}, {3, 4, 1} });
    sparseGrid1->setN({ 10,10,10 });
    sparseGrid1->init();

    EXPECT_EQ(sparseGrid1->getNbPoints(), 392);
    EXPECT_EQ(sparseGrid1->getNbHexahedra(), 209);
    EXPECT_NEAR(sparseGrid1->getPosX(0), -1.02, 1e-04); // 0.02 comes from RegularGridTopology::setPos(), xmin-xmax/(n_x-1)
    EXPECT_NEAR(sparseGrid1->getPosY(0), -0.34, 1e-04);
    EXPECT_NEAR(sparseGrid1->getPosZ(0), -1.02 , 1e-04);

    SparseGridTopology::SPtr sparseGrid2 = New<SparseGridTopology>();
    EXPECT_NE(sparseGrid2, nullptr);
    MeshSTLLoader::SPtr stlLoader = New<MeshSTLLoader>();
    EXPECT_NE(stlLoader, nullptr);
    stlLoader->setFilename("mesh/suzanne.stl");
    EXPECT_TRUE(stlLoader->load());
    EXPECT_EQ(stlLoader->d_positions.getValue().size(), 505);
    EXPECT_EQ(stlLoader->d_triangles.getValue().size(), 968);

    sparseGrid2->seqPoints.setParent(&stlLoader->d_positions);
    sparseGrid2->seqTriangles.setParent(&stlLoader->d_triangles);
    sparseGrid2->seqQuads.setParent(&stlLoader->d_quads);
    sparseGrid2->setN({ 10,10,10 });
    sparseGrid2->init();

    EXPECT_EQ(sparseGrid2->getNbPoints(), 550);
    EXPECT_EQ(sparseGrid2->getNbHexahedra(), 338);

    return true;
}

TEST_F(SparseGridTopology_test, buildFromMeshFile) { ASSERT_TRUE(buildFromMeshFile()); }
TEST_F(SparseGridTopology_test, buildFromMeshParams) { ASSERT_TRUE(buildFromMeshParams()); }



