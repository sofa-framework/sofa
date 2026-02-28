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
#include <sofa/component/mass/MeshMatrixMass.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::mass::testing
{

using MeshTopology = sofa::component::topology::container::constant::MeshTopology;

/***************************************************************************************************
 * MeshMatrixMass
 **************************************************************************************************/

template <typename DataTypes>
struct MeshMatrixMass_template_test : public Mass_test<MeshMatrixMass<DataTypes>>
{
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;
    using Real = sofa::Real_t<DataTypes>;

    MeshTopology::SPtr m_topology;

    MeshMatrixMass_template_test()
    {
        m_topology = sofa::core::objectmodel::New<MeshTopology>();
        this->m_node->addObject(m_topology);
    }

    void run(const std::vector<std::vector<Real>>& coords, bool lumped = false)
    {
        this->m_mass->setTotalMass(10.0_sreal);

        VecCoord x(static_cast<sofa::Size>(coords.size()));
        VecDeriv v(static_cast<sofa::Size>(coords.size()));

        for (size_t i = 0; i < coords.size(); ++i)
        {
            DataTypes::set(x[i], coords[i][0], coords[i][1], coords[i][2]);
            DataTypes::set(v[i], (coords[i][0] * 0.1), (coords[i][1] * 0.1), (coords[i][2] * 0.1));
            m_topology->addPoint(static_cast<SReal>(coords[i][0]), static_cast<SReal>(coords[i][1]), static_cast<SReal>(coords[i][2]));
        }

        m_topology->computeCrossElementBuffers();

        this->m_testAccFromF = lumped;
        this->m_mass->d_lumping.setValue(lumped);

        this->run_test(x, v);
    }

    void runTriangle(bool lumped)
    {
        this->m_topology->addTriangle(0, 1, 2);
        this->m_topology->addEdge(0, 1);
        this->m_topology->addEdge(1, 2);
        this->m_topology->addEdge(2, 0);
        this->run({{0,0,0}, {1,0,0}, {0,1,0}}, lumped);
    }

    void runQuad(bool lumped)
    {
        this->m_topology->addQuad(0, 1, 2, 3);
        this->m_topology->addEdge(0, 1);
        this->m_topology->addEdge(1, 2);
        this->m_topology->addEdge(2, 3);
        this->m_topology->addEdge(3, 0);
        this->run({{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}}, lumped);
    }

    void runTetrahedron(bool lumped)
    {
        this->m_topology->addTetra(0, 1, 2, 3);

        for (const auto edge : sofa::core::topology::edgesInTetrahedronArray)
        {
            this->m_topology->addEdge(edge[0], edge[1]);
        }

        this->run({{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}}, lumped);
    }

    void runHexahedron(bool lumped)
    {
        this->m_topology->addHexa(0, 1, 2, 3, 4, 5, 6, 7);

        for (const auto edge : sofa::core::topology::edgesInHexahedronArray)
        {
            this->m_topology->addEdge(edge[0], edge[1]);
        }

        this->run({{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
               {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}}, lumped);
    }
};

typedef ::testing::Types<
    defaulttype::Vec3Types
> MeshMatrixMassDataTypes;

TYPED_TEST_SUITE(MeshMatrixMass_template_test, MeshMatrixMassDataTypes);

TYPED_TEST(MeshMatrixMass_template_test, Triangle)
{
    this->runTriangle(false);
}

TYPED_TEST(MeshMatrixMass_template_test, TriangleLumped)
{
    this->runTriangle(true);
}

TYPED_TEST(MeshMatrixMass_template_test, Tetrahedron)
{
    this->runTetrahedron(false);
}

TYPED_TEST(MeshMatrixMass_template_test, TetrahedronLumped)
{
    this->runTetrahedron(true);
}

TYPED_TEST(MeshMatrixMass_template_test, Quad)
{
    this->runQuad(false);
}

TYPED_TEST(MeshMatrixMass_template_test, QuadLumped)
{
    this->runQuad(true);
}

TYPED_TEST(MeshMatrixMass_template_test, Hexahedron)
{
    this->runHexahedron(false);
}

TYPED_TEST(MeshMatrixMass_template_test, HexahedronLumped)
{
    this->runHexahedron(true);
}

} // namespace sofa::component::mass::testing
