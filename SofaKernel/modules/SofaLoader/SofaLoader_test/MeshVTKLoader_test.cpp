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

#include <SofaTest/Sofa_test.h>

#include "SofaLoader/MeshVTKLoader.h"

namespace sofa {

struct MeshVTKLoader_test : public Sofa_test<>, public component::loader::MeshVTKLoader
{

    MeshVTKLoader_test()
    {}

    void test_load(std::string const& filename, unsigned nbPoints, unsigned nbEdges, unsigned nbTriangles, unsigned nbQuads, unsigned nbPolygons, unsigned nbTetrahedra, unsigned nbHexahedra)
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

TEST_F(MeshVTKLoader_test, detectFileType)
{
    ASSERT_EQ(component::loader::MeshVTKLoader::LEGACY, detectFileType(sofa::helper::system::DataRepository.getFile("mesh/liver.vtk").c_str()));
    ASSERT_EQ(component::loader::MeshVTKLoader::XML, detectFileType(sofa::helper::system::DataRepository.getFile("mesh/Armadillo_Tetra_4406.vtu").c_str()));
}

TEST_F(MeshVTKLoader_test, loadLegacy)
{
    test_load(sofa::helper::system::DataRepository.getFile("mesh/liver.vtk"), 5008, 0, 10000, 0, 0, 0, 0);
}

TEST_F(MeshVTKLoader_test, loadXML)
{
    test_load(sofa::helper::system::DataRepository.getFile("mesh/Armadillo_Tetra_4406.vtu"), 1446, 0, 0, 0, 0, 4406, 0);
}

TEST_F(MeshVTKLoader_test, load)
{
    setFilename("");
    EXPECT_FALSE(load());
}

}// namespace sofa
