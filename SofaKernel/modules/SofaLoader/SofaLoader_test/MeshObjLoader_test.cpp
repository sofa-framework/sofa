/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaTest/Sofa_test.h>

#include <SofaLoader/MeshObjLoader.h>

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace ;

using namespace sofa::component::loader;

namespace sofa
{
namespace meshobjloader_test
{

int initTestEnvironment()
{
    BackTrace::autodump() ;
    return 0;
}
int s_autodump = initTestEnvironment() ;


class MeshObjLoader_test  : public ::testing::Test, public MeshObjLoader
{
public :

    /**
     * Constructor call for each test
     */
    void SetUp(){}

    /**
     * Helper function to check mesh loading.
     * Compare basic values from a mesh with given results.
     */
    void loadTest(std::string filename, int pointNb, int edgeNb, int triangleNb,  int quadNb, int polygonNb,
                  int tetraNb, int hexaNb, int normalPerVertexNb, int normalListNb, int texCoordListNb, int materialNb)
    {
        SOFA_UNUSED(normalListNb);
        SOFA_UNUSED(texCoordListNb);
        SOFA_UNUSED(materialNb);

        this->setFilename(sofa::helper::system::DataRepository.getFile(filename));

        EXPECT_TRUE(this->load());
        EXPECT_EQ((size_t)pointNb, this->d_positions.getValue().size());
        EXPECT_EQ((size_t)edgeNb, this->d_edges.getValue().size());
        EXPECT_EQ((size_t)triangleNb, this->d_triangles.getValue().size());
        EXPECT_EQ((size_t)quadNb, this->d_quads.getValue().size());
        EXPECT_EQ((size_t)polygonNb, this->d_polygons.getValue().size());
        EXPECT_EQ((size_t)tetraNb, this->d_tetrahedra.getValue().size());
        EXPECT_EQ((size_t)hexaNb, this->d_hexahedra.getValue().size());
        EXPECT_EQ((size_t)normalPerVertexNb, this->d_normals.getValue().size());
    }

};

/** MeshObjLoader::load()
 * For a given meshes check that imported data are correct
 */
TEST_F(MeshObjLoader_test, LoadCall)
{
    //Check loader high level result
    EXPECT_FALSE(this->load());
    this->setFilename(sofa::helper::system::DataRepository.getFile("mesh/square.obj"));
    EXPECT_TRUE(this->load());

    //Use several meshes to test : edges, triangles, quads, normals, materials NUMBER
    loadTest("mesh/square.obj", 4, 4, 0,  0, 0, 0, 0, 0, 0, 0, 0);
    loadTest("mesh/dragon.obj", 1190, 0, 2564,  0, 0, 0, 0, 0, 0, 0, 0);
    loadTest("mesh/box.obj", 4, 0, 4,  0, 0, 0, 0, 4, 2, 0, 0);
    loadTest("mesh/cube.obj", 8, 0, 12,  0, 0, 0, 0, 8, 8, 0, 0);
    loadTest("mesh/caducee_base.obj", 3576, 0, 420,  3350, 0, 0, 0, 0, 0, 0, 8);
    loadTest("mesh/torus.obj", 800, 0, 1600,  0, 0, 0, 0, 0, 0, 861, 0);
}

} // namespace meshobjloader_test
} // namespace sofa
