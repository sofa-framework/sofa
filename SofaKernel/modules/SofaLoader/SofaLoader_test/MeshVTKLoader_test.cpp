/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
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
using sofa::helper::logging::ExpectMessage ;
using sofa::helper::logging::Message ;

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
    testLoad(DataRepository.getFile("mesh/vox8_binary.vtk"), 27, 0, 0, 0, 0, 0, 8);

    const auto& cellDataVec = this->reader->inputCellDataVector;
    const auto& pointDataVec = this->reader->inputPointDataVector;

    const auto findCellData = [this,&cellDataVec] (const std::string& name)
    {
        auto res = std::find_if(cellDataVec.begin(), cellDataVec.end(), [&name](const BaseVTKDataIO* data_ptr)
        {
            return data_ptr->name == name;
        });
        return res ;
    };
    auto cRamp1 = findCellData("cRamp1");
    EXPECT_TRUE(cRamp1 != cellDataVec.end());
    EXPECT_TRUE(dynamic_cast<VTKDataIO<float>*>(*cRamp1) != nullptr);

    auto cRamp2 = findCellData("cRamp2");
    EXPECT_TRUE(cRamp2 != cellDataVec.end());
    EXPECT_TRUE(dynamic_cast<VTKDataIO<float>*>(*cRamp2) != nullptr);

    auto cVects = findCellData("cVects");
    EXPECT_TRUE(cVects != cellDataVec.end());
    VTKDataIO<defaulttype::Vec<3, float>>* cVectsCasted = dynamic_cast<VTKDataIO<defaulttype::Vec<3, float>>*>(*cVects);
    EXPECT_TRUE(cVectsCasted != nullptr);

    auto cv2 = findCellData("cv2");
    EXPECT_TRUE(cv2 != cellDataVec.end());
    VTKDataIO<defaulttype::Vec<3, float>>* cv2Casted = dynamic_cast<VTKDataIO<defaulttype::Vec<3, float>>*>(*cv2);
    EXPECT_TRUE(cv2Casted != nullptr);


    const auto findPointData = [this,&pointDataVec] (const std::string& name)
    {
        auto res = std::find_if(pointDataVec.begin(), pointDataVec.end(), [&name](const BaseVTKDataIO* data_ptr)
        {
            return data_ptr->name == name;
        });
        return res;
    };

    auto mytest = findPointData("mytest");
    EXPECT_TRUE(mytest != pointDataVec.end());
    EXPECT_TRUE(dynamic_cast<VTKDataIO<float>*>(*mytest) != nullptr);

    auto xRamp = findPointData("xRamp");
    EXPECT_TRUE(xRamp != pointDataVec.end());
    EXPECT_TRUE(dynamic_cast<VTKDataIO<float>*>(*xRamp) != nullptr);

    auto yRamp = findPointData("yRamp");
    EXPECT_TRUE(yRamp != pointDataVec.end());
    EXPECT_TRUE(dynamic_cast<VTKDataIO<float>*>(*yRamp) != nullptr);

    auto zRamp = findPointData("zRamp");
    EXPECT_TRUE(zRamp != pointDataVec.end());
    EXPECT_TRUE(dynamic_cast<VTKDataIO<float>*>(*zRamp) != nullptr);

    auto outVect = findPointData("outVect");
    EXPECT_TRUE(outVect != pointDataVec.end());
    VTKDataIO<defaulttype::Vec<3, float>>* outVectCasted = dynamic_cast<VTKDataIO<defaulttype::Vec<3, float>>*>(*outVect);
    EXPECT_TRUE(outVectCasted != nullptr);

    auto vect2 = findPointData("vect2");
    EXPECT_TRUE(vect2 != pointDataVec.end());
    VTKDataIO<defaulttype::Vec<3, float>>* vect2Casted = dynamic_cast<VTKDataIO<defaulttype::Vec<3, float>>*>(*vect2);
    EXPECT_TRUE(vect2Casted != nullptr);
}

TEST_F(MeshVTKLoaderTest, loadInvalidFilenames)
{
    ExpectMessage errmsg(Message::Error) ;

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
