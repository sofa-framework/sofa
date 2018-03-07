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
#include <SceneCreator/SceneCreator.h>

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation;
using sofa::simulation::Simulation ;
using sofa::core::objectmodel::New ;

#include <SofaGeneralEngine/MergePoints.h>
using sofa::component::engine::MergePoints ;

using std::vector;
using std::string;

namespace sofa
{

template <typename _DataTypes>
struct MergePoints_test : public Sofa_test<typename _DataTypes::Real>,
        MergePoints<_DataTypes>
{
    typedef MergePoints<_DataTypes> ThisClass;
    typedef _DataTypes DataTypes;
    typedef typename _DataTypes::Coord Coord;
    typedef typename _DataTypes::VecCoord VecCoord;
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;

    Simulation* m_simu;
    typename ThisClass::SPtr m_thisObject;

    void SetUp()
    {
        setSimulation(m_simu = new DAGSimulation());
        m_thisObject = New<ThisClass >();
    }

    /// It is important to freeze what are the available Data field
    /// of a component and rise warning/errors when some one removed.
    void attrTest()
    {
        m_thisObject->setName("myname") ;
        EXPECT_TRUE(m_thisObject->getName() == "myname") ;

        // List of the supported attributes the user expect to find
        // This list needs to be updated if you add an attribute.
        vector<string> attrnames = {
            "position1", "position2", "mappingX2",
            "indices1","indices2", "points",
            "noUpdate"
        };

        for(auto& attrname : attrnames)
            EXPECT_NE( m_thisObject->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;

        return ;
    }

    /// Shouldn't crash without input data
    void initTest()
    {
        EXPECT_NO_THROW(m_thisObject->init()) << "The component should succeed in being initialized.";
    }

    void additionTest()
    {
        // set addition mode
        m_thisObject->f_X2_mapping.unset();
        // reset output
        m_thisObject->f_indices1.unset();
        m_thisObject->f_indices2.unset();
        m_thisObject->f_points.unset();
        // set input
        VecCoord x1;
        x1.push_back(Coord(0.0f, 0.0f, 0.0f));
        x1.push_back(Coord(1.0f, 0.0f, 0.0f));
        m_thisObject->f_X1 = x1;
        VecCoord x2;
        x2.push_back(Coord(0.0f, 1.0f, 0.0f));
        x2.push_back(Coord(0.0f, 0.0f, 1.0f));
        m_thisObject->f_X2 = x2;
        // update
        m_thisObject->update();
        // check output
        const VecCoord& points = m_thisObject->f_points.getValue();
        const SetIndex& indices1 = m_thisObject->f_indices1.getValue();
        const SetIndex& indices2 = m_thisObject->f_indices2.getValue();
        ASSERT_EQ(points.size(),4);
        ASSERT_EQ(indices1.size(),2);
        ASSERT_EQ(indices2.size(),2);
        ASSERT_TRUE( (Coord(0.0f, 0.0f, 0.0f)-points[0]).norm() < 0.000000001f);
        ASSERT_TRUE( (Coord(1.0f, 0.0f, 0.0f)-points[1]).norm() < 0.000000001f);
        ASSERT_TRUE( (Coord(0.0f, 1.0f, 0.0f)-points[2]).norm() < 0.000000001f);
        ASSERT_TRUE( (Coord(0.0f, 0.0f, 1.0f)-points[3]).norm() < 0.000000001f);
        ASSERT_EQ(indices1[0],0);
        ASSERT_EQ(indices1[1],1);
        ASSERT_EQ(indices2[0],2);
        ASSERT_EQ(indices2[1],3);
    }

    void injectiontest()
    {
        // TODO
    }

};

using testing::Types;
typedef Types<Vec3Types> DataTypes;

TYPED_TEST_CASE(MergePoints_test, DataTypes);

TYPED_TEST(MergePoints_test, initTest ) {
    ASSERT_NO_THROW(this->initTest());
}
TYPED_TEST(MergePoints_test, attrTest ){
    ASSERT_NO_THROW(this->attrTest());
}
TYPED_TEST(MergePoints_test, additionTest ){
    ASSERT_NO_THROW(this->additionTest());
}
//TYPED_TEST(MergePoints_test, injectiontest ){
//    ASSERT_NO_THROW(this->injectiontest());
//}

}


