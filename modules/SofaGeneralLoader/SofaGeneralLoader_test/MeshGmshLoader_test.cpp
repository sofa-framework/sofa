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

#include <sofa/helper/testing/BaseTest.h>
#include <sofa/helper/system/FileRepository.h>

#include <SofaGeneralLoader/MeshGmshLoader.h>
#include <sofa/helper/BackTrace.h>

using namespace sofa::component::loader;
using sofa::helper::testing::BaseTest;
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
        void loadTest(std::string filename, int pointNb, int edgeNb, int triangleNb, int quadNb, int polygonNb,
                      int tetraNb, int hexaNb, int normalPerVertexNb, int normalListNb, int texCoordListNb, int materialNb)
        {
            SOFA_UNUSED(normalPerVertexNb);
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
            EXPECT_EQ((size_t)normalListNb, this->d_normals.getValue().size());
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

        ////Use meshes to test : edges, triangles, quads, normals, materials NUMBER
        loadTest("mesh/msh4_cube.msh", 14, 49, 60, 0, 0, 24, 0, 14, 0, 0, 0);
    }

} // namespace meshgmshloader_test
} // namespace sofa
