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
#include <sofa/helper/system/FileRepository.h>

#include <sofa/component/io/mesh/MeshGmshLoader.h>

#include <sofa/helper/BackTrace.h>

using namespace sofa::component::io::mesh;
using sofa::testing::BaseTest;
using sofa::helper::BackTrace;

namespace sofa
{
namespace meshgmshloader_test
{

    int initTestEnvironment()
    {
        BackTrace::autodump();
        return 0;
    }
    int s_autodump = initTestEnvironment();


    class MeshGmshLoader_test : public BaseTest, public MeshGmshLoader
    {
    public:

        /**
            * Constructor call for each test
            */
        void SetUp() override {}

        /**
            * Helper function to check mesh loading.
            * Compare basic values from a mesh with given results.
            */
        void loadTest(std::string filename, int nbPositions, int nbEdges, int nbTriangles, int nbQuads, int nbPolygons,
                      int nbTetra, int nbHexa, int nbNormals)
        {
            this->setFilename(sofa::helper::system::DataRepository.getFile(filename));

            EXPECT_TRUE(this->load());
            EXPECT_EQ((size_t)nbPositions, this->d_positions.getValue().size());
            EXPECT_EQ((size_t)nbEdges, this->d_edges.getValue().size());
            EXPECT_EQ((size_t)nbTriangles, this->d_triangles.getValue().size());
            EXPECT_EQ((size_t)nbQuads, this->d_quads.getValue().size());
            EXPECT_EQ((size_t)nbPolygons, this->d_polygons.getValue().size());
            EXPECT_EQ((size_t)nbTetra, this->d_tetrahedra.getValue().size());
            EXPECT_EQ((size_t)nbHexa, this->d_hexahedra.getValue().size());
            EXPECT_EQ((size_t)nbNormals, this->d_normals.getValue().size());
        }

    };

    /** MeshGmshLoader::load()
    * For a given meshes check that imported data are correct
    */
    TEST_F(MeshGmshLoader_test, LoadCall)
    {
        //Check loader high level result
        this->setFilename(sofa::helper::system::DataRepository.getFile("mesh/msh4_cube.msh"));
        EXPECT_TRUE(this->load());

        // Testing number of : nodes/positions, edges, triangles, quads, polygons, tetra, hexa, normals
        loadTest("mesh/msh4_cube.msh", 14, 12, 24, 0, 0, 24, 0, 0); //Data read by Gmsh software
    }

} // namespace meshgmshloader_test
} // namespace sofa
