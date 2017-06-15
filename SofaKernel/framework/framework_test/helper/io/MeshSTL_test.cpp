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
#include <sofa/helper/io/MeshSTL.h>

#include <gtest/gtest.h>

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::Message;

namespace sofa {

class MeshSTL_test : public ::testing::Test
{
protected:

  MeshSTL_test() {}

  void SetUp()
  {
      sofa::helper::system::DataRepository.addFirstPath(FRAMEWORK_TEST_RESOURCES_DIR);
  }
  void TearDown()
  {
      sofa::helper::system::DataRepository.removePath(FRAMEWORK_TEST_RESOURCES_DIR);
  }

  struct MeshSTLTestData
  {
      sofa::helper::io::MeshSTL mesh;

      std::string filename;
      unsigned int nbVertices;
      unsigned int nbLines;
      unsigned int nbTriangles;
      unsigned int nbQuads;
      unsigned int nbFacets;
      unsigned int nbNormals;

      MeshSTLTestData(const std::string& filename, unsigned int nbVertices
          , unsigned int nbLines, unsigned int nbTriangles, unsigned int nbQuads, unsigned int nbFacets
          , unsigned int nbNormals)
          : mesh(filename)
          , filename(filename), nbVertices(nbVertices)
          , nbLines(nbLines), nbTriangles(nbTriangles), nbQuads(nbQuads), nbFacets(nbFacets)
          , nbNormals(nbNormals)
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
          EXPECT_EQ(nbNormals, mesh.getNormals().size());
      }
  };
};

TEST_F(MeshSTL_test, MeshSTL_NoFile)
{
    /// This generate a test failure if no message is generated.
    EXPECT_MSG_EMIT(Error) ;

    MeshSTLTestData meshNoFile("mesh/randomnamewhichdoesnotexist.obj", 0, 0, 0, 0, 0, 0);
    meshNoFile.testBench();
}

TEST_F(MeshSTL_test, MeshSTL_NoMesh)
{
    MeshSTLTestData meshNoMesh("mesh/meshtest_nomesh.stl", 0, 0, 0, 0, 0, 0);
    meshNoMesh.testBench();
}


}
