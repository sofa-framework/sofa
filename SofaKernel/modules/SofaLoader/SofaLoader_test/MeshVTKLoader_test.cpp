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

#include <gtest/gtest.h>

#include "SofaLoader/MeshVTKLoader.h"
#include "SofaLoader/BaseVTKReader.h"
using sofa::component::loader::MeshVTKLoader ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository ;

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace ;

#include <SofaTest/TestMessageHandler.h>

namespace sofa
{
namespace meshvtkloader_test
{

int initTestEnvironment()
{
    BackTrace::autodump() ;
    return 0;
}
int s_autodump = initTestEnvironment() ;


struct MeshVTKLoaderTest : public ::testing::Test,
                            public MeshVTKLoader
{
    using BaseVTKDataIO = component::loader::BaseVTKReader::BaseVTKDataIO;
    template<typename T>
    using VTKDataIO = component::loader::BaseVTKReader::VTKDataIO<T>;

    MeshVTKLoaderTest()
    {}

    void testLoad(std::string const& filename, unsigned nbPoints, unsigned nbEdges, unsigned nbTriangles, unsigned nbQuads, unsigned nbPolygons, unsigned nbTetrahedra, unsigned nbHexahedra)
    {
        setFilename(filename);
        EXPECT_TRUE(load());
        EXPECT_EQ(nbPoints, d_positions.getValue().size());
        EXPECT_EQ(nbEdges, d_edges.getValue().size());
        EXPECT_EQ(nbTriangles, d_triangles.getValue().size());
        EXPECT_EQ(nbQuads, d_quads.getValue().size());
        EXPECT_EQ(nbPolygons, d_polygons.getValue().size());
        EXPECT_EQ(nbTetrahedra, d_tetrahedra.getValue().size());
        EXPECT_EQ(nbHexahedra, d_hexahedra.getValue().size());
    }

};

TEST_F(MeshVTKLoaderTest, detectFileType)
{
    ASSERT_EQ(MeshVTKLoader::LEGACY, detectFileType(DataRepository.getFile("mesh/liver.vtk").c_str()));
    ASSERT_EQ(MeshVTKLoader::XML, detectFileType(DataRepository.getFile("mesh/Armadillo_Tetra_4406.vtu").c_str()));
}

TEST_F(MeshVTKLoaderTest, loadLegacy)
{
    testLoad(DataRepository.getFile("mesh/liver.vtk"), 5008, 0, 10000, 0, 0, 0, 0);
}

TEST_F(MeshVTKLoaderTest, loadXML)
{
    testLoad(DataRepository.getFile("mesh/Armadillo_Tetra_4406.vtu"), 1446, 0, 0, 0, 0, 4406, 0);
}

TEST_F(MeshVTKLoaderTest, loadLegacy_binary)
{
    using BaseData = core::objectmodel::BaseData;

    testLoad(DataRepository.getFile("mesh/vox8_binary.vtk"), 27, 0, 0, 0, 0, 0, 8);

    BaseData* cRamp1 = this->findData("cRamp1");
    EXPECT_TRUE(cRamp1 != nullptr);
    EXPECT_TRUE(dynamic_cast<Data<helper::vector<float>>*>(cRamp1) != nullptr);

    BaseData* cRamp2 = this->findData("cRamp2");
    EXPECT_TRUE(cRamp2 != nullptr);
    EXPECT_TRUE(dynamic_cast<Data<helper::vector<float>>*>(cRamp2) != nullptr);

    BaseData* cVects = this->findData("cVects");
    EXPECT_TRUE(cVects != nullptr);
    EXPECT_TRUE(dynamic_cast<Data<helper::vector<defaulttype::Vec3f>>*>(cVects) != nullptr);

    BaseData* cv2 = this->findData("cv2");
    EXPECT_TRUE(cv2 != nullptr);
    EXPECT_TRUE(dynamic_cast<Data<helper::vector<defaulttype::Vec3f>>*>(cv2) != nullptr);

    BaseData* mytest = this->findData("mytest");
    EXPECT_TRUE(mytest != nullptr);
    EXPECT_TRUE(dynamic_cast<Data<helper::vector<float>>*>(mytest) != nullptr);

    BaseData* xRamp = this->findData("xRamp");
    EXPECT_TRUE(xRamp != nullptr);
    EXPECT_TRUE(dynamic_cast<Data<helper::vector<float>>*>(xRamp) != nullptr);

    BaseData* yRamp = this->findData("yRamp");
    EXPECT_TRUE(yRamp != nullptr);
    EXPECT_TRUE(dynamic_cast<Data<helper::vector<float>>*>(yRamp) != nullptr);

    BaseData* zRamp = this->findData("zRamp");
    EXPECT_TRUE(zRamp != nullptr);
    EXPECT_TRUE(dynamic_cast<Data<helper::vector<float>>*>(zRamp) != nullptr);

    BaseData* outVect = this->findData("outVect");
    EXPECT_TRUE(outVect != nullptr);
    EXPECT_TRUE(dynamic_cast<Data<helper::vector<defaulttype::Vec3f>>*>(outVect) != nullptr);

    BaseData* vect2 = this->findData("vect2");
    EXPECT_TRUE(vect2 != nullptr);
    EXPECT_TRUE(dynamic_cast<Data<helper::vector<defaulttype::Vec3f>>*>(vect2) != nullptr);
}

TEST_F(MeshVTKLoaderTest, loadInvalidFilenames)
{
    EXPECT_MSG_EMIT(Error) ;

    setFilename("");
    EXPECT_FALSE(load());

    setFilename("/home/test/thisisnotavalidpath");
    EXPECT_FALSE(load());

    setFilename(DataRepository.getFile("test.vtu"));
    EXPECT_FALSE(load());

    setFilename(DataRepository.getFile("test.vtk"));
    EXPECT_FALSE(load());
}

//TODO(dmarchal): Remove this tests until we can fix them.
#if 0
TEST_F(MeshVTKLoaderTest, loadBrokenVtkFile_OpenIssue)
{
    setFilename(DataRepository.getFile("mesh/liver_for_test_broken.vtk"));
    EXPECT_FALSE(load());
}

TEST_F(MeshVTKLoaderTest, loadBrokenVtuFile_OpenIssue)
{
    setFilename(DataRepository.getFile("mesh/Armadillo_Tetra_4406_for_test_broken.vtu"));
    EXPECT_FALSE(load());
}
#endif

}// namespace meshvtkloader_test
}// namespace sofa
