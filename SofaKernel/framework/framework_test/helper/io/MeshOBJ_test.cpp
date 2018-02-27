/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/io/MeshOBJ.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

namespace sofa {

using namespace core::loader;

class MeshOBJ_test : public BaseTest
{
protected:
    void SetUp()
    {
        sofa::helper::system::DataRepository.addFirstPath(FRAMEWORK_TEST_RESOURCES_DIR);
    }
    void TearDown()
    {
        sofa::helper::system::DataRepository.removePath(FRAMEWORK_TEST_RESOURCES_DIR);
    }

    struct MeshOBJTestData
    {
        sofa::helper::io::MeshOBJ mesh;

        std::string filename;
        unsigned int nbVertices;
        unsigned int nbLines;
        unsigned int nbTriangles;
        unsigned int nbQuads;
        unsigned int nbFacets;
        unsigned int nbTexcoords;
        unsigned int nbNormals;

        MeshOBJTestData(const std::string& filename, unsigned int nbVertices
            , unsigned int nbLines, unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbFacets
            , unsigned int nbTexcoords, unsigned int nbNormals)
            : mesh(filename)
            , filename(filename), nbVertices(nbVertices)
            , nbLines(nbLines), nbTriangles(nbTriangles), nbQuads(nbQuads), nbFacets(nbFacets)
            , nbTexcoords(nbTexcoords), nbNormals(nbNormals)
        {

        }

        void testBench()
        {
            EXPECT_EQ(nbVertices, mesh.getVertices().size());
            const sofa::helper::vector< helper::vector < helper::vector <int> > > & facets = mesh.getFacets();
            unsigned int fileNbLines, fileNbTriangles, fileNbQuads;
            fileNbLines = fileNbTriangles = fileNbQuads = 0;
            for (unsigned int i = 0; i < facets.size(); i++)
            {
                const helper::vector<helper::vector<int> >& vtnFacet = facets[i];
                if (vtnFacet[0].size() > 0)
                {
                    const helper::vector<int>& facet = vtnFacet[0];
                    switch (facet.size())
                    {
                    case 2:
                        fileNbLines++;
                        break;
                    case 3:
                        fileNbTriangles++;
                        break;
                    case 4:
                        fileNbQuads++;
                        break;
                    default:
                        //?
                        break;
                    }
                }
            }
            EXPECT_EQ(nbLines, fileNbLines);
            EXPECT_EQ(nbTriangles, fileNbTriangles);
            EXPECT_EQ(nbQuads, fileNbQuads);
            EXPECT_EQ(nbFacets, facets.size());
            EXPECT_EQ(nbTexcoords, mesh.getTexCoords().size());
            EXPECT_EQ(nbNormals, mesh.getNormals().size());
        }
    };

};

TEST_F(MeshOBJ_test, MeshOBJ_NoFile)
{
    /// This generate a test failure if no message is generated.
    EXPECT_MSG_EMIT(Error) ;

    MeshOBJTestData meshNoFile("mesh/randomnamewhichdoesnotexist.obj", 0, 0, 0, 0, 0, 0, 0);
    meshNoFile.testBench();
}

TEST_F(MeshOBJ_test, MeshOBJ_NoMesh)
{
    MeshOBJTestData meshNoMesh("mesh/meshtest_nomesh.obj", 0, 0, 0, 0, 0, 0, 0);
    meshNoMesh.testBench();
}

TEST_F(MeshOBJ_test, MeshOBJ_UV_N_MTL_Init)
{
    MeshOBJTestData meshUVNMTL("mesh/meshtest_uv_n_mtl.obj", 10, 1, 4, 5, 10, 27, 9);
    meshUVNMTL.testBench();
}


}// namespace sofa
