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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/mass/testing/MassTestCreation.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::mass::testing
{

/***************************************************************************************************
 * UniformMass
 **************************************************************************************************/

template <typename DataTypes>
struct UniformMass_template_test : public Mass_test<UniformMass<DataTypes>>
{
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;

    void run()
    {
        this->m_mass->setTotalMass(10.0_sreal);

        VecCoord x(2);
        DataTypes::set(x[0], 9.5, -49.2, 5.32);
        DataTypes::set(x[1], 0.8, 17.6, -7.3);

        VecDeriv v(2);
        DataTypes::set(v[0], 3.54, -0.87, 12.09);
        DataTypes::set(v[1], 0.048, -8.7, -0.12);

        this->run_test(x, v);
    }
};

typedef ::testing::Types<
    defaulttype::Vec1Types,
    defaulttype::Vec2Types,
    defaulttype::Vec3Types,
    defaulttype::Vec6Types
> UniformMassDataTypes;

TYPED_TEST_SUITE(UniformMass_template_test, UniformMassDataTypes);

TYPED_TEST(UniformMass_template_test, test)
{
    this->run();
}

} // namespace sofa::component::mass::testing
