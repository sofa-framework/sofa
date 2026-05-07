#include <gtest/gtest.h>
#include <sofa/component/topology/mapping/Hexa2TetraTopologicalMapping.h>
#include <sofa/component/topology/container/grid/GridTopology.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>

// Test 1: Invalid input handling
TEST(Hexa2TetraTopologicalMappingTest, InvalidInputHandling)
{
    auto mapping = sofa::core::objectmodel::New<
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();

    // Test without setting input/output
    EXPECT_NE(mapping->d_componentState.getValue(),
              sofa::core::objectmodel::ComponentState::Valid);
}

// Test 2: Verify correct tetrahedron count
TEST(Hexa2TetraTopologicalMappingTest, TetrahedronCountForVariousGridSizes)
{
    // Test different grid sizes
    std::vector<std::pair<sofa::type::Vec3i, size_t>> gridSizes = {
        {{2, 2, 2}, 6},      // 1 hexa -> 6 tetras
        {{3, 2, 2}, 12},     // 2 hexas -> 12 tetras
        {{3, 3, 2}, 24},     // 4 hexas -> 24 tetras
        {{3, 3, 3}, 48}      // 8 hexas -> 48 tetras
    };

    for (const auto& [gridSize, expectedTetras] : gridSizes)
    {
        auto grid = sofa::core::objectmodel::New<
            sofa::component::topology::container::grid::GridTopology>(gridSize);

        auto tetraTopo = sofa::core::objectmodel::New<
            sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
        auto mapping = sofa::core::objectmodel::New<
            sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();

        mapping->setTopologies(grid.get(), tetraTopo.get());
        mapping->init();

        EXPECT_EQ(tetraTopo->getNbTetrahedra(), expectedTetras)
            << "Failed for grid size " << gridSize;
    }
}

// Test 3: Verify vertex consistency
TEST(Hexa2TetraTopologicalMappingTest, VertexConsistency)
{
    auto grid = sofa::core::objectmodel::New<
        sofa::component::topology::container::grid::GridTopology>(3, 3, 3);

    auto tetraTopo = sofa::core::objectmodel::New<
        sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
    auto mapping = sofa::core::objectmodel::New<
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();

    mapping->setTopologies(grid.get(), tetraTopo.get());
    mapping->init();

    // Verify same number of points
    ASSERT_EQ(tetraTopo->getNbPoints(), grid->getNbPoints());
}

// Test 4: Test with swapping enabled
TEST(Hexa2TetraTopologicalMappingTest, SwappingParameter)
{
    auto grid = sofa::core::objectmodel::New<
        sofa::component::topology::container::grid::GridTopology>(3, 3, 3);

    // Test without swapping
    auto tetraTopo1 = sofa::core::objectmodel::New<
        sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
    auto mapping1 = sofa::core::objectmodel::New<
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();
    mapping1->d_swapping.setValue(false);
    mapping1->setTopologies(grid.get(), tetraTopo1.get());
    mapping1->init();

    // Test with swapping
    auto tetraTopo2 = sofa::core::objectmodel::New<
        sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
    auto mapping2 = sofa::core::objectmodel::New<
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();
    mapping2->d_swapping.setValue(true);
    mapping2->setTopologies(grid.get(), tetraTopo2.get());
    mapping2->init();

    // Both should have same number of tetrahedra
    ASSERT_EQ(tetraTopo1->getNbTetrahedra(), tetraTopo2->getNbTetrahedra());
}

// Test 5: Verify tetrahedron validity (valid indices)
TEST(Hexa2TetraTopologicalMappingTest, TetrahedronValidity)
{
    auto grid = sofa::core::objectmodel::New<
        sofa::component::topology::container::grid::GridTopology>(3, 3, 3);

    auto tetraTopo = sofa::core::objectmodel::New<
        sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
    auto mapping = sofa::core::objectmodel::New<
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();

    mapping->setTopologies(grid.get(), tetraTopo.get());
    mapping->init();

    const sofa::Size nbPoints = tetraTopo->getNbPoints();

    // Check all tetrahedra have valid vertex indices
    for (size_t i = 0; i < tetraTopo->getNbTetrahedra(); ++i)
    {
        auto tetra = tetraTopo->getTetrahedron(i);
        EXPECT_LT(tetra[0], nbPoints);
        EXPECT_LT(tetra[1], nbPoints);
        EXPECT_LT(tetra[2], nbPoints);
        EXPECT_LT(tetra[3], nbPoints);
    }
}

// Test 6: Local-to-global mapping consistency
TEST(Hexa2TetraTopologicalMappingTest, LocalToGlobalMapping)
{
    auto grid = sofa::core::objectmodel::New<
        sofa::component::topology::container::grid::GridTopology>(3, 3, 3);

    auto tetraTopo = sofa::core::objectmodel::New<
        sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
    auto mapping = sofa::core::objectmodel::New<
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();

    mapping->setTopologies(grid.get(), tetraTopo.get());
    mapping->init();

    // Verify mapping data structures
    // Each hexahedron should map to 6 tetrahedra
    const sofa::Size expectedSize = grid->getNbHexahedra() * 6;
    EXPECT_EQ(mapping->Loc2GlobDataVec.getValue().size(), expectedSize);
}

// Test 7: Verify tetrahedra output for single hexahedron without swapping
TEST(Hexa2TetraTopologicalMappingTest, SingleHexahedronOutputNoSwapping)
{
    // Create a 2x2x2 grid (1 hexahedron)
    auto grid = sofa::core::objectmodel::New<
        sofa::component::topology::container::grid::GridTopology>(2, 2, 2);

    auto tetraTopo = sofa::core::objectmodel::New<
        sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
    auto mapping = sofa::core::objectmodel::New<
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();

    mapping->d_swapping.setValue(false);
    mapping->setTopologies(grid.get(), tetraTopo.get());
    mapping->init();

    // Get the hexahedron vertices (for a 2x2x2 grid, there's one hexahedron)
    auto hexa = grid->getHexahedron(0);

    // Verify we have exactly 6 tetrahedra
    ASSERT_EQ(tetraTopo->getNbTetrahedra(), 6);

    // Expected tetrahedra based on the decomposition pattern without swapping
    // Pattern from Hexa2TetraTopologicalMapping.cpp:
    const std::array expectedTetras = {
        sofa::topology::Tetrahedron{hexa[0], hexa[5], hexa[1], hexa[6]},
        sofa::topology::Tetrahedron{hexa[0], hexa[1], hexa[3], hexa[6]},
        sofa::topology::Tetrahedron{hexa[1], hexa[3], hexa[6], hexa[2]},
        sofa::topology::Tetrahedron{hexa[6], hexa[3], hexa[0], hexa[7]},
        sofa::topology::Tetrahedron{hexa[6], hexa[7], hexa[0], hexa[5]},
        sofa::topology::Tetrahedron{hexa[7], hexa[5], hexa[4], hexa[0]},
    };

    for (sofa::Size i = 0; i < tetraTopo->getNbTetrahedra(); ++i)
    {
        auto tetra = tetraTopo->getTetrahedron(i);
        for (sofa::Index j = 0; j < 4; ++j)
        {
            EXPECT_EQ(tetra[j], expectedTetras[i][j]) << "Failed for tetra " << i << " at index " << j;
        }
    }
}

// Test 8: Verify tetrahedra output for single hexahedron with swapping
TEST(Hexa2TetraTopologicalMappingTest, SingleHexahedronOutputWithSwapping)
{
    // Create a 2x2x2 grid (1 hexahedron)
    auto grid = sofa::core::objectmodel::New<
        sofa::component::topology::container::grid::GridTopology>(2, 2, 2);

    auto tetraTopo = sofa::core::objectmodel::New<
        sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
    auto mapping = sofa::core::objectmodel::New<
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();

    mapping->d_swapping.setValue(true);
    mapping->setTopologies(grid.get(), tetraTopo.get());
    mapping->init();

    // Get the hexahedron vertices
    auto hexa = grid->getHexahedron(0);

    // Verify we have exactly 6 tetrahedra
    ASSERT_EQ(tetraTopo->getNbTetrahedra(), 6);

    // Expected tetrahedra based on the decomposition pattern with swapping
    const std::array expectedTetras = {
        sofa::topology::Tetrahedron{hexa[1], hexa[4], hexa[7], hexa[0]},
        sofa::topology::Tetrahedron{hexa[1], hexa[0], hexa[7], hexa[2]},
        sofa::topology::Tetrahedron{hexa[0], hexa[2], hexa[3], hexa[7]},
        sofa::topology::Tetrahedron{hexa[7], hexa[2], hexa[6], hexa[1]},
        sofa::topology::Tetrahedron{hexa[7], hexa[6], hexa[4], hexa[1]},
        sofa::topology::Tetrahedron{hexa[6], hexa[4], hexa[1], hexa[5]},
    };

    for (sofa::Size i = 0; i < tetraTopo->getNbTetrahedra(); ++i)
    {
        auto tetra = tetraTopo->getTetrahedron(i);
        for (sofa::Index j = 0; j < 4; ++j)
        {
            EXPECT_EQ(tetra[j], expectedTetras[i][j]) << "Failed for tetra " << i << " at index " << j;
        }
    }
}

// Test 9: Verify all tetrahedra vertices belong to source hexahedron
TEST(Hexa2TetraTopologicalMappingTest, AllTetrahedraVerticesBelongToSourceHexa)
{
    // Create a 3x3x3 grid (8 hexahedra)
    auto grid = sofa::core::objectmodel::New<sofa::component::topology::container::grid::GridTopology>(3, 3, 3);

    auto tetraTopo = sofa::core::objectmodel::New<
        sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
    auto mapping = sofa::core::objectmodel::New<
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();

    mapping->setTopologies(grid.get(), tetraTopo.get());
    mapping->init();

    // For each hexahedron, verify that its 6 tetrahedra use only vertices from that hexahedron
    size_t nbHexas = grid->getNbHexahedra();
    for (size_t hexaIdx = 0; hexaIdx < nbHexas; ++hexaIdx)
    {
        auto hexa = grid->getHexahedron(hexaIdx);
        std::set<sofa::Index> hexaVertices(hexa.begin(), hexa.end());

        // Check the 6 tetrahedra corresponding to this hexahedron
        for (size_t localTetraIdx = 0; localTetraIdx < 6; ++localTetraIdx)
        {
            size_t globalTetraIdx = hexaIdx * 6 + localTetraIdx;
            auto tetra = tetraTopo->getTetrahedron(globalTetraIdx);

            // All 4 vertices of the tetrahedron must be in the hexahedron
            for (int i = 0; i < 4; ++i)
            {
                EXPECT_TRUE(hexaVertices.find(tetra[i]) != hexaVertices.end())
                    << "Tetrahedron " << globalTetraIdx << " vertex " << i
                    << " (index " << tetra[i] << ") does not belong to hexahedron " << hexaIdx;
            }
        }
    }
}

// Test 10: Verify tetrahedra have unique vertices (no degenerate tetrahedra)
TEST(Hexa2TetraTopologicalMappingTest, NoDegenerateTetrahedra)
{
    auto grid = sofa::core::objectmodel::New<sofa::component::topology::container::grid::GridTopology>(3, 3, 3);

    auto tetraTopo = sofa::core::objectmodel::New<
        sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
    auto mapping = sofa::core::objectmodel::New<
        sofa::component::topology::mapping::Hexa2TetraTopologicalMapping>();

    mapping->setTopologies(grid.get(), tetraTopo.get());
    mapping->init();

    // Check that all tetrahedra have 4 unique vertices
    for (size_t i = 0; i < tetraTopo->getNbTetrahedra(); ++i)
    {
        auto tetra = tetraTopo->getTetrahedron(i);
        std::set<sofa::Index> uniqueVertices(tetra.begin(), tetra.end());

        EXPECT_EQ(uniqueVertices.size(), 4)
            << "Tetrahedron " << i << " has duplicate vertices";
    }
}
