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

#include <gtest/gtest.h>
#include <sofa/component/topology/container/dynamic/PointSetTopologyContainer.h>

using namespace sofa::component::topology::container::dynamic;


namespace
{

TEST( PointSetTopology_test, checkPointSetTopologyIsEmptyConstructed )
{
    const PointSetTopologyContainer::SPtr pointContainer = sofa::core::objectmodel::New< PointSetTopologyContainer >();
    EXPECT_EQ( 0, pointContainer->getNbPoints() );
}


TEST( PointSetTopology_test, checkPointSetTopologyInitialization )
{
    const PointSetTopologyContainer::SPtr pointContainer = sofa::core::objectmodel::New< PointSetTopologyContainer >();
    sofa::defaulttype::Vec3Types::VecCoord initPos;
    initPos.resize( 50 );
    pointContainer->d_initPoints.setValue( initPos );

    pointContainer->init();

    EXPECT_EQ( 50, pointContainer->getNbPoints() );
}

TEST( PointSetTopology_test, checkAddPoint )
{
    const PointSetTopologyContainer::SPtr pointContainer = sofa::core::objectmodel::New< PointSetTopologyContainer >();
    pointContainer->addPoint();

    EXPECT_EQ( 1, pointContainer->getNbPoints() );
}

TEST( PointSetTopology_test, checkAddPoints )
{
    const PointSetTopologyContainer::SPtr pointContainer = sofa::core::objectmodel::New< PointSetTopologyContainer >();
    pointContainer->addPoints(10);

    EXPECT_EQ( 10, pointContainer->getNbPoints() );

    pointContainer->addPoints(5);

    EXPECT_EQ( 15, pointContainer->getNbPoints() );
}

TEST( PointSetTopology_test, checkRemovePoint )
{
    const PointSetTopologyContainer::SPtr pointContainer = sofa::core::objectmodel::New< PointSetTopologyContainer >();
    pointContainer->addPoint();
    pointContainer->removePoint();
   
    EXPECT_EQ( 0, pointContainer->getNbPoints() );
}

TEST( PointSetTopology_test, checkRemovePoints )
{
    const PointSetTopologyContainer::SPtr pointContainer = sofa::core::objectmodel::New< PointSetTopologyContainer >();
    pointContainer->addPoints(10);
    pointContainer->removePoints(3);
    
    EXPECT_EQ( 7, pointContainer->getNbPoints() );

    pointContainer->removePoints(3);

    EXPECT_EQ( 4, pointContainer->getNbPoints() );

}





}
