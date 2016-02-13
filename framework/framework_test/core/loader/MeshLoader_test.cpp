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
#include <sofa/core/loader/MeshLoader.h>

#include <gtest/gtest.h>

namespace sofa {

using namespace core::loader;

class MeshLoader_test;

class MeshTestLoader : public MeshLoader
{
public:
    friend class MeshLoader_test;
    typedef helper::WriteAccessor< Data<helper::vector<sofa::defaulttype::Vector3> > > waPositions;
    typedef helper::WriteAccessor< Data< helper::vector< Triangle > > > waTtriangles;
    typedef helper::WriteAccessor< Data< helper::vector< Tetrahedron > > > waTetrahedra;

    virtual bool load()
    {
        return true;
    }
};

/** Test suite for MeshLoader
 *
 * @author Thomas Lemaire @date 2014
 */
class MeshLoader_test : public ::testing::Test
{
protected:
    MeshLoader_test() {}

    void updateMesh()
    {
        meshLoader.updateMesh();
    }

    void populateMesh_1triangle_1tetra()
    {
        MeshTestLoader::waPositions my_positions(meshLoader.positions);
        meshLoader.addPosition(&(my_positions.wref()), 0.,0.,0.);
        meshLoader.addPosition(&(my_positions.wref()), 1.,0.,0.);
        meshLoader.addPosition(&(my_positions.wref()), 0.,1.,0.);
        meshLoader.addPosition(&(my_positions.wref()), 0.,0.,1.);

        MeshTestLoader::waTtriangles my_triangles(meshLoader.triangles);
        meshLoader.addTriangle(&(my_triangles.wref()), MeshLoader::Triangle(0,1,2));

        MeshTestLoader::waTetrahedra my_tetrahedra(meshLoader.tetrahedra);
        meshLoader.addTetrahedron(&(my_tetrahedra.wref()), MeshLoader::Tetrahedron(0,1,2,3) );

    }

    MeshTestLoader meshLoader;

};

TEST_F(MeshLoader_test, createSubElements)
{
    populateMesh_1triangle_1tetra();
    EXPECT_EQ(4u, meshLoader.positions.getValue().size());
    EXPECT_EQ(1u, meshLoader.triangles.getValue().size());
    EXPECT_EQ(1u, meshLoader.tetrahedra.getValue().size());

    meshLoader.createSubelements.setValue(false);
    updateMesh();
    EXPECT_EQ(4u, meshLoader.positions.getValue().size());
    EXPECT_EQ(1u, meshLoader.triangles.getValue().size());
    EXPECT_EQ(1u, meshLoader.tetrahedra.getValue().size());

    meshLoader.createSubelements.setValue(true);
    updateMesh();
    EXPECT_EQ(4u, meshLoader.positions.getValue().size());
    EXPECT_EQ(4u, meshLoader.triangles.getValue().size());
    EXPECT_EQ(1u, meshLoader.tetrahedra.getValue().size());

}

}// namespace sofa
