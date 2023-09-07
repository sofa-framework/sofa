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

#include <sofa/component/io/mesh/MeshSTLLoader.h>
#include <sofa/helper/BackTrace.h>

using namespace sofa::component::io::mesh;
using sofa::testing::BaseTest;
using sofa::helper::BackTrace;

namespace sofa::meshstlloader_test
{

int initTestEnvironment()
{
    BackTrace::autodump();
    return 0;
}
int s_autodump = initTestEnvironment();


class MeshSTLLoaderTest : public BaseTest, public MeshSTLLoader
{
public:

    MeshSTLLoaderTest()
    {}

    /**
    * Helper function to check mesh loading.
    * Compare basic values from a mesh with given results.
    */
    void loadTest(std::string filename, int nbPositions, int nbEdges, int nbTriangles, int nbQuads, int nbPolygons,
                    int nbTetra, int nbHexa, int nbNormals)
    {
        this->setFilename(sofa::helper::system::DataRepository.getFile(filename));

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

/** MeshSTLLoader::load()
* For a given meshes check that imported data are correct
*/
TEST_F(MeshSTLLoaderTest, LoadCall)
{
    //Check loader high level result for one file
    EXPECT_MSG_EMIT(Error);
    EXPECT_FALSE(this->load());
    this->setFilename(sofa::helper::system::DataRepository.getFile("mesh/circle_knot_ascii.stl"));
    EXPECT_TRUE(this->load());

    // Testing number of : nodes/positions, edges, triangles, quads, polygons, tetra, hexa, normals
    // for several STL meshes in share/mesh/
    loadTest("mesh/circle_knot_ascii.stl", 6144, 0, 12288, 0, 0, 0, 0, 12288);
    loadTest("mesh/dragon.stl", 1190, 0, 2526, 0, 0, 0, 0, 2526);
    loadTest("mesh/pliers_binary.stl", 5356, 0, 10708, 0, 0, 0, 0, 10712);
}

} // namespace sofa::meshstlloader_test
