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

#include <sofa/component/collision/response/contact/PenalityContactForceField.h>
#include <sofa/component/solidmechanics/testing/ForceFieldTestCreation.h>

using sofa::component::collision::response::contact::PenalityContactForceField;

struct PenalityContactForceField_test : public sofa::ForceField_test<PenalityContactForceField<sofa::defaulttype::Vec3Types> >
{
    void test_2particles( Real stiffness, Real contactDistance,
                          sofa::type::Vec3 x0,sofa::type:: Vec3 v0,
                          sofa::type::Vec3 x1, sofa::type:: Vec3 v1,
                          sofa::type::Vec3 f0)
    {
        // potential energy is not implemented and won't be tested
        this->flags &= ~TEST_POTENTIAL_ENERGY;

        VecCoord x(2);
        DataTypes::set( x[0], x0[0],x0[1],x0[2]);
        DataTypes::set( x[1], x1[0],x1[1],x1[2]);
        VecDeriv v(2);
        DataTypes::set( v[0], v0[0],v0[1],v0[2]);
        DataTypes::set( v[1], v1[0],v1[1],v1[2]);
        VecDeriv f(2);
        DataTypes::set( f[0],  f0[0], f0[1], f0[2]);
        DataTypes::set( f[1], -f0[0],-f0[1],-f0[2]);

        this->force->addContact(0, 1, 0, 0, (x1 - x0).normalized(), contactDistance, stiffness);

        this->run_test( x, v, f );
    }
};

TEST_F(PenalityContactForceField_test, twoParticles)
{
    sofa::type::Vec3
            x0(0,0,0), // position of the first particle
            v0(0,0,0), // velocity of the first particle
            x1(2,0,0), // position of the second particle
            v1(0,0,0), // velocity of the second particle
            f0(-1,0,0); // expected force on the first particleá

    SReal k = 1.0;  // stiffness
    SReal contactDistance = 3.;

    this->test_2particles(k, contactDistance, x0, v0, x1,v1, f0);
    this->force->clear();

    k = 2.0;
    f0 = {-2., 0., 0.};

    this->test_2particles(k, contactDistance, x0, v0, x1,v1, f0);
    this->force->clear();

    x1 = { 4.8, -0.4, -2.3};
    contactDistance = 6.;
    f0 = {-1.19136193691474900901994260494, 0.0992801614095624312961163582258, 0.57086092810498378913308670235};

    this->test_2particles(k, contactDistance, x0, v0, x1,v1, f0);
    this->force->clear();
}
