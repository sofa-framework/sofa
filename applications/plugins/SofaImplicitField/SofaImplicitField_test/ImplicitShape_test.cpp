/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
* with this program. If not, see <http://www.gnuSceneCreator_test.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/testing/BaseTest.h>

#include <sofa/type/Vec.h>
using sofa::type::Vec3d ;

#include <SofaImplicitField/components/geometry/SphericalField.h>
using sofa::component::geometry::SphericalField ;

namespace
{

class SphericalFieldTest : public sofa::testing::BaseTest
{
public:
    bool checkSphericalField();
    bool checkDiscreteGridField();
};


bool SphericalFieldTest::checkSphericalField()
{
    SphericalField sphere_test;
    Vec3d p(1,1,2);
    sphere_test.getValue(p) ;
    return true;
}


TEST_F(SphericalFieldTest, checkSphericalField) { ASSERT_TRUE( checkSphericalField() ); }

}
