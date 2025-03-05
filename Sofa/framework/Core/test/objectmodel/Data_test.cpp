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
#include <sofa/testing/config.h>

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/vectorData.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;


namespace sofa {

using namespace core::objectmodel;

class Data_test : public BaseTest
{
public:
    Data<int> dataInt;
    Data<float> dataFloat;
    Data<double> dataDouble;
    Data<sofa::type::vector<sofa::type::RGBAColor>> dataVectorColor;
    Data<bool> dataBool;

    Data<SReal> dataSReal;
    Data<sofa::type::Vec3> dataVec3;
    Data<sofa::type::vector<sofa::type::Vec3>> dataVectorVec3;

    Data<sofa::topology::Element<sofa::geometry::Edge> > dataEdge;
    Data<sofa::topology::Element<sofa::geometry::Hexahedron> > dataHexahedron;
    Data<sofa::topology::Element<sofa::geometry::Pentahedron> > dataPentahedron;
    Data<sofa::topology::Element<sofa::geometry::Point> > dataPoint;
    Data<sofa::topology::Element<sofa::geometry::Pyramid> > dataPyramid;
    Data<sofa::topology::Element<sofa::geometry::Quad> > dataQuad;
    Data<sofa::topology::Element<sofa::geometry::Tetrahedron> > dataTetrahedron;
    Data<sofa::topology::Element<sofa::geometry::Triangle> > dataTriangle;

    Data<sofa::type::vector<sofa::topology::Element<sofa::geometry::Edge> > > dataVecEdge;
    Data<sofa::type::vector<sofa::topology::Element<sofa::geometry::Hexahedron> > > dataVecHexahedron;
    Data<sofa::type::vector<sofa::topology::Element<sofa::geometry::Pentahedron> > > dataVecPentahedron;
    Data<sofa::type::vector<sofa::topology::Element<sofa::geometry::Point> > > dataVecPoint;
    Data<sofa::type::vector<sofa::topology::Element<sofa::geometry::Pyramid> > > dataVecPyramid;
    Data<sofa::type::vector<sofa::topology::Element<sofa::geometry::Quad> > > dataVecQuad;
    Data<sofa::type::vector<sofa::topology::Element<sofa::geometry::Tetrahedron> > > dataVecTetrahedron;
    Data<sofa::type::vector<sofa::topology::Element<sofa::geometry::Triangle> > > dataVecTriangle;

};


TEST_F(Data_test, validInfo)
{
    EXPECT_TRUE(defaulttype::DataTypeInfo<sofa::topology::Element<sofa::geometry::Edge> >::ValidInfo);
    EXPECT_TRUE(defaulttype::DataTypeInfo<sofa::topology::Element<sofa::geometry::Hexahedron> >::ValidInfo);
    EXPECT_TRUE(defaulttype::DataTypeInfo<sofa::topology::Element<sofa::geometry::Pentahedron> >::ValidInfo);
    // EXPECT_TRUE(defaulttype::DataTypeInfo<sofa::topology::Element<sofa::geometry::Point> >::ValidInfo);
    EXPECT_TRUE(defaulttype::DataTypeInfo<sofa::topology::Element<sofa::geometry::Pyramid> >::ValidInfo);
    EXPECT_TRUE(defaulttype::DataTypeInfo<sofa::topology::Element<sofa::geometry::Quad> >::ValidInfo);
    EXPECT_TRUE(defaulttype::DataTypeInfo<sofa::topology::Element<sofa::geometry::Tetrahedron> >::ValidInfo);
    EXPECT_TRUE(defaulttype::DataTypeInfo<sofa::topology::Element<sofa::geometry::Triangle> >::ValidInfo);
}

TEST_F(Data_test, dataTypeName)
{
    EXPECT_EQ(std::string{sofa::geometry::ElementInfo<sofa::geometry::Edge>::name()}, "Edge");
    EXPECT_EQ(std::string{sofa::geometry::ElementInfo<sofa::geometry::Hexahedron>::name()}, "Hexahedron");
    EXPECT_EQ(std::string{sofa::geometry::ElementInfo<sofa::geometry::Pentahedron>::name()}, "Pentahedron");
    // EXPECT_EQ(std::string{sofa::geometry::ElementInfo<sofa::geometry::Point>::name()}, "Point");
    EXPECT_EQ(std::string{sofa::geometry::ElementInfo<sofa::geometry::Pyramid>::name()}, "Pyramid");
    EXPECT_EQ(std::string{sofa::geometry::ElementInfo<sofa::geometry::Quad>::name()}, "Quad");
    EXPECT_EQ(std::string{sofa::geometry::ElementInfo<sofa::geometry::Tetrahedron>::name()}, "Tetrahedron");
    EXPECT_EQ(std::string{sofa::geometry::ElementInfo<sofa::geometry::Triangle>::name()}, "Triangle");


    EXPECT_EQ(defaulttype::DataTypeName<sofa::topology::Element<sofa::geometry::Edge> >::name(), "Edge");
    EXPECT_EQ(defaulttype::DataTypeName<sofa::topology::Element<sofa::geometry::Hexahedron> >::name(), "Hexahedron");
    EXPECT_EQ(defaulttype::DataTypeName<sofa::topology::Element<sofa::geometry::Pentahedron> >::name(), "Pentahedron");
    // EXPECT_EQ(defaulttype::DataTypeName<sofa::topology::Element<sofa::geometry::Point> >::name(), "Point");
    EXPECT_EQ(defaulttype::DataTypeName<sofa::topology::Element<sofa::geometry::Pyramid> >::name(), "Pyramid");
    EXPECT_EQ(defaulttype::DataTypeName<sofa::topology::Element<sofa::geometry::Quad> >::name(), "Quad");
    EXPECT_EQ(defaulttype::DataTypeName<sofa::topology::Element<sofa::geometry::Tetrahedron> >::name(), "Tetrahedron");
    EXPECT_EQ(defaulttype::DataTypeName<sofa::topology::Element<sofa::geometry::Triangle> >::name(), "Triangle");
}

TEST_F(Data_test, getValueTypeString)
{
    EXPECT_EQ(dataInt.getValueTypeString(), "i");
    EXPECT_EQ(dataFloat.getValueTypeString(), "f");
    EXPECT_EQ(dataDouble.getValueTypeString(), "d");
    EXPECT_EQ(dataBool.getValueTypeString(), "bool");
    EXPECT_EQ(dataVectorColor.getValueTypeString(), "vector<RGBAColor>");

    if constexpr (std::is_same_v <SReal, double>)
    {
        EXPECT_EQ(dataSReal.getValueTypeString(), "d");
        EXPECT_EQ(dataVec3.getValueTypeString(), "Vec3d");
        EXPECT_EQ(dataVectorVec3.getValueTypeString(), "vector<Vec3d>");
    }
    else
    {
        EXPECT_EQ(dataSReal.getValueTypeString(), "f");
        EXPECT_EQ(dataVec3.getValueTypeString(), "Vec3f");
        EXPECT_EQ(dataVectorVec3.getValueTypeString(), "vector<Vec3f>");
    }

    EXPECT_EQ(dataEdge.getValueTypeString(), "Edge");
    EXPECT_EQ(dataHexahedron.getValueTypeString(), "Hexahedron");
    EXPECT_EQ(dataPentahedron.getValueTypeString(), "Pentahedron");
    // EXPECT_EQ(dataPoint.getValueTypeString(), "Point");
    EXPECT_EQ(dataPyramid.getValueTypeString(), "Pyramid");
    EXPECT_EQ(dataQuad.getValueTypeString(), "Quad");
    EXPECT_EQ(dataTetrahedron.getValueTypeString(), "Tetrahedron");
    EXPECT_EQ(dataTriangle.getValueTypeString(), "Triangle");

    EXPECT_EQ(dataVecEdge.getValueTypeString(), "vector<Edge>");
    EXPECT_EQ(dataVecHexahedron.getValueTypeString(), "vector<Hexahedron>");
    EXPECT_EQ(dataVecPentahedron.getValueTypeString(), "vector<Pentahedron>");
    // EXPECT_EQ(dataVecPoint.getValueTypeString(), "vector<Point>");
    EXPECT_EQ(dataVecPyramid.getValueTypeString(), "vector<Pyramid>");
    EXPECT_EQ(dataVecQuad.getValueTypeString(), "vector<Quad>");
    EXPECT_EQ(dataVecTetrahedron.getValueTypeString(), "vector<Tetrahedron>");
    EXPECT_EQ(dataVecTriangle.getValueTypeString(), "vector<Triangle>");
}

TEST_F(Data_test, getNameWithValueTypeInfo)
{
    EXPECT_EQ(dataInt.getValueTypeInfo()->name(), "i");
    EXPECT_EQ(dataFloat.getValueTypeInfo()->name(), "f");
    EXPECT_EQ(dataDouble.getValueTypeInfo()->name(), "d");
    EXPECT_EQ(dataBool.getValueTypeInfo()->name(), "bool");
    EXPECT_EQ(dataVectorColor.getValueTypeInfo()->name(), "vector<RGBAColor>");

    if constexpr (std::is_same_v <SReal, double>)
    {
        EXPECT_EQ(dataSReal.getValueTypeInfo()->name(), "d");
        EXPECT_EQ(dataVec3.getValueTypeInfo()->name(), "Vec3d");
        EXPECT_EQ(dataVectorVec3.getValueTypeInfo()->name(), "vector<Vec3d>");
    }
    else
    {
        EXPECT_EQ(dataSReal.getValueTypeInfo()->name(), "f");
        EXPECT_EQ(dataVec3.getValueTypeInfo()->name(), "Vec3f");
        EXPECT_EQ(dataVectorVec3.getValueTypeInfo()->name(), "vector<Vec3f>");
    }
}
}// namespace sofa
