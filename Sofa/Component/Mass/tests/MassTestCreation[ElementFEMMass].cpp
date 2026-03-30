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
#include <sofa/component/mass/ElementFEMMass.h>
#include <sofa/component/mass/testing/MassTestCreation.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/testing/LinearCongruentialRandomGenerator.h>

namespace sofa::component::mass::testing
{

using MeshTopology = sofa::component::topology::container::constant::MeshTopology;


/***************************************************************************************************
 * ElementFEMMass
 **************************************************************************************************/

template <typename MassParam>
struct ElementMass_template_test : public Mass_test<ElementFEMMass<typename MassParam::DataTypes, typename MassParam::ElementType>>
{
    using DataTypes = typename MassParam::DataTypes;

    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;

    MeshTopology::SPtr m_topology;

    ElementMass_template_test()
    {
        this->m_testAccFromF = false;
        this->m_testKineticEnergy = false;
        this->m_testAddMToMatrix = false;

        m_topology = sofa::core::objectmodel::New<MeshTopology>();
        this->m_node->addObject(m_topology);

        m_topology->addEdge(0, 1);
        m_topology->addTriangle(0, 1, 2);
        m_topology->addQuad(0, 1, 2, 3);
        m_topology->addTetra(0, 1, 2, 3);
        m_topology->addHexa(0, 1, 2, 3, 4, 5, 6, 7);

        auto nodalDensity = sofa::core::objectmodel::New<NodalMassDensity<sofa::Real_t<DataTypes>>>();
        this->m_node->addObject(nodalDensity);
    }

    void run()
    {
        VecCoord x(8);
        VecDeriv v(8);

        sofa::testing::LinearCongruentialRandomGenerator lcg(96547);

        for (std::size_t i = 0; i < 8; ++i)
        {
            DataTypes::set(x[i], lcg.generateInRange(-10., 10.), lcg.generateInRange(-10., 10.), lcg.generateInRange(-10., 10.));
            DataTypes::set(v[i], lcg.generateInRange(-10., 10.), lcg.generateInRange(-10., 10.), lcg.generateInRange(-10., 10.));
        }

        this->run_test(x, v);
    }
};

template<class TDataTypes, class TElementType>
struct MassParam
{
    using DataTypes = TDataTypes;
    using ElementType = TElementType;
};

typedef ::testing::Types<
    MassParam<defaulttype::Vec1Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec2Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec2Types, sofa::geometry::Triangle>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Triangle>,
    MassParam<defaulttype::Vec2Types, sofa::geometry::Quad>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Quad>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Tetrahedron>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Hexahedron>
> ElementMassDataTypes;

TYPED_TEST_SUITE(ElementMass_template_test, ElementMassDataTypes);

TYPED_TEST(ElementMass_template_test, test)
{
    this->run();
}

} // namespace sofa::component::mass::testing
