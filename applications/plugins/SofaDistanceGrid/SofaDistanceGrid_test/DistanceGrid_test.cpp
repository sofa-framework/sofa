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
#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test ;

#include <sofa/defaulttype/Vec.h>

#include <SofaDistanceGrid/DistanceGrid.h>
using sofa::component::container::DistanceGrid ;

namespace sofa
{
namespace component
{
namespace container
{
namespace _distancegrid_
{
using sofa::defaulttype::Vector3 ;

struct DistanceGrid_test : public Sofa_test<SReal>
{
    void chekcValidConstructorsCube(){
        EXPECT_MSG_NOEMIT(Warning, Error) ;

        DistanceGrid grid(10, 10, 10,
                          DistanceGrid::Coord(-1,-1,-1),
                          DistanceGrid::Coord(1.0,1.0,1.0)) ;

        EXPECT_EQ(grid.getNx(), 10) ;
        EXPECT_EQ(grid.getNy(), 10) ;
        EXPECT_EQ(grid.getNz(), 10) ;

        EXPECT_FALSE(grid.inBBox(Vector3(-2, 0, 0), 0.0f)) ;
        EXPECT_FALSE(grid.inBBox(Vector3( 0,-2, 0), 0.0f)) ;
        EXPECT_FALSE(grid.inBBox(Vector3( 0, 0,-2), 0.0f)) ;
        EXPECT_FALSE(grid.inBBox(Vector3( 2, 0, 0), 0.0f)) ;
        EXPECT_FALSE(grid.inBBox(Vector3( 0, 2, 0), 0.0f)) ;
        EXPECT_FALSE(grid.inBBox(Vector3( 0, 0, 2), 0.0f)) ;

        EXPECT_EQ(grid.size(), 10*10*10);

        //todo(dmarchal:2017-05-02) This "isCube" & "getCubeDim" stuff is ugly as hell !
        EXPECT_FALSE(grid.isCube());
    }

    void checInvalidConstructorsCube(int x, int y, int z,
                                     float mx, float my, float mz,
                                     float ex, float ey, float ez){
        std::cout << "x-y-z:" << x << ", " << y << ", " << z << std::endl ;
        std::cout << "mx-my-mz:" << mx << ", " << my << ", " << mz << std::endl  ;
        std::cout << "ex-ey-ez:" << ex << ", " << ey << ", " << ez << std::endl ;

        DistanceGrid grid(x, y, z,
                          DistanceGrid::Coord(mx,my,mz),
                          DistanceGrid::Coord(ex,ey,ez)) ;
    }
};

TEST_F(DistanceGrid_test, chekcValidConstructorsCube) {
    ASSERT_NO_THROW(this->chekcValidConstructorsCube()) ;
}

TEST_F(DistanceGrid_test, chekcInvalidConstructorsCube) {
    std::vector< std::vector< float >> values = {
        {-10, 10, 10,  -1,-1,-1,  1, 1,1},
        { 10,-10, 10,  -1,-1,-1,  1, 1,1},
        { 10, 10,-10,  -1,-1,-1,  1, 1,1},
        { 10, 10,  0,  -1,-1,-1,  1, 1,1},
        { 10,  0, 10,  -1,-1,-1,  1, 1,1},
        {  0, 10,  0,  -1,-1,-1,  1, 1,1},
        {  0, 10,  0,  -1, 1,-1,  1,-1,1} };
    for(auto& v : values ){
        ASSERT_NO_THROW( this->checInvalidConstructorsCube(
                           (int)v[0], (int)v[1], (int)v[2],
                                v[3],      v[4],      v[5],
                                v[6],      v[7],      v[8])) ;
    }
}


} // __distance_grid__
} // container
} // component
} // sofa
