/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include <sstream>
using std::stringstream ;

#include <string>
using std::string ;

#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Data ;

#include <sofa/helper/types/Material.h>
using sofa::helper::types::Material ;

#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test ;

namespace sofa {

class Material_test : public Sofa_test<>
{
public:

    void checkConstructor()
    {
        Material m;
        EXPECT_FALSE( m.activated );
        EXPECT_TRUE( m.useAmbient );
        EXPECT_TRUE( m.useDiffuse );
        EXPECT_FALSE( m.useSpecular );
        EXPECT_FALSE( m.useEmissive );
        EXPECT_FALSE( m.useShininess );
        EXPECT_FALSE( m.useTexture );
        EXPECT_FALSE( m.useBumpMapping );
    }

    void checkDataRead(const std::string& testmat)
    {
        Material m1;
        m1.name = "notdefault" ;
        EXPECT_EQ( m1.name, "notdefault" ) ;

        Data<Material> m;
        m.setValue(m1) ;
        EXPECT_EQ( m.getValue().name, "notdefault" ) ;

        m.read( testmat );
        EXPECT_EQ( m.getValue().name, "sofa_logo" ) ;
        EXPECT_TRUE( m.getValue().useAmbient ) ;
        EXPECT_TRUE( m.getValue().useDiffuse ) ;
        EXPECT_TRUE( m.getValue().useSpecular ) ;
        EXPECT_TRUE( m.getValue().useShininess ) ;
        EXPECT_FALSE( m.getValue().useEmissive ) ;
        EXPECT_EQ( m.getValueString(), testmat ) ;
    }
};

TEST_F(Material_test, checkConstructor)
{
        checkConstructor();
}

TEST_F(Material_test, checkDataRead)
{
        checkDataRead("sofa_logo Diffuse 1 0.3 0.18 0.05 1 Ambient 1 0.05 0.02 0 1 Specular 1 1 1 1 1 Emissive 0 0 0 0 0 Shininess 1 1000 ");
}



}// namespace sofa
