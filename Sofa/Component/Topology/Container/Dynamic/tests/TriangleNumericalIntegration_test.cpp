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

//Including Simulation
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/set.h>
// Including constraint, force and mass
#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/NumericalIntegrationDescriptor.h>
#include <sofa/component/topology/container/dynamic/CommonAlgorithms.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest;


namespace sofa {

using namespace component;
using namespace type;
using namespace defaulttype;
/**  Patch test in 2D and 3D.
A movement is applied to the borders of a mesh. The points within should have a bilinear movement relative to the border movements.*/

template <typename _DataTypes>
struct TriangleNumericalIntegration_test : public NumericTest<typename _DataTypes::Real>
{
    using Inherit = NumericTest<typename _DataTypes::Real>;

    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<DataTypes> TriangleSetGeometryAlgorithms;
    typedef typename sofa::component::topology::container::dynamic::NumericalIntegrationDescriptor<Real, 3> NumericalIntegrationDescriptor;
    //  typedef typename sofa::component::topology::lfactorial lfactorial;
    /// Root of the scene graph
    simulation::Node::SPtr root;
    /// Simulation
    simulation::Simulation* simulation;
    // the geometry algorithm algorithm
    typename sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<DataTypes>::SPtr geo;

    // Create the context for the scene
    void SetUp() override
    {
        // Init simulation
        simulation = sofa::simulation::getSimulation();

        root = simulation::getSimulation()->createNewGraph("root");
    }
    // create the TriangleSetGeometryAlgorithms object
    void createScene()
    {
        geo = core::objectmodel::New<TriangleSetGeometryAlgorithms>();
        root->addObject(geo);
    }
    bool testNumericalIntegration()
    {
        size_t k;
        Real integral, weight;
        typename NumericalIntegrationDescriptor::BarycentricCoordinatesType bc;
        Vec<3, unsigned short> randomPolynomial;

        // get the descriptor of numerical integration on tetrahedra
        NumericalIntegrationDescriptor &nid = geo->getTriangleNumericalIntegrationDescriptor();
        // get all quadrature method
        std::set<typename NumericalIntegrationDescriptor::QuadratureMethod> qmArray = nid.getQuadratureMethods();
        typename std::set<typename NumericalIntegrationDescriptor::QuadratureMethod>::iterator itqm;
        for (itqm = qmArray.begin(); itqm != qmArray.end(); itqm++) {
            // get all the integration orders for this integration method
            std::set<typename NumericalIntegrationDescriptor::IntegrationOrder> ioArray = nid.getIntegrationOrders(*itqm);
            typename std::set<typename NumericalIntegrationDescriptor::IntegrationOrder>::iterator itio;
            // go through the integration orders of this method
            for (itio = ioArray.begin(); itio != ioArray.end(); itio++) {
                // get the integration point
                typename NumericalIntegrationDescriptor::QuadraturePointArray qpa = nid.getQuadratureMethod(*itqm, *itio);
                /// set the random polynomial defined as u^i*v^j*w^k where [i,j,k] is a random vector given by randomPolynomial
                /// such that i+j+k+l= degree
                for (k = 0; k < 3; ++k) randomPolynomial[k] = 0;
                for (k = 0; k < (*itio); ++k) {
                    randomPolynomial[helper::irand() % 3]++;
                }
                // compute the integral over the triangle through numerical integration
                integral = (Real)0;
                for (k = 0; k < qpa.size(); ++k) {
                    typename NumericalIntegrationDescriptor::QuadraturePoint qp = qpa[k];
                    // the barycentric coordinate
                    bc = qp.first;
                    // the weight of the integration point
                    weight = qp.second;
                    integral += pow(bc[0], randomPolynomial[0])*pow(bc[1], randomPolynomial[1])*
                        pow(bc[2], randomPolynomial[2])*weight;

                }
                /// real integral value
                /// use the classical integration formula on the tetrahedron with barycentric coordinates i.e.
                /// int_{\tetrahedron} L_1^a  L_2^b  L_3^c   dV= (a! b! c!) *2V / (a+b+c+2)!
                /// where L1 , L2 , L3  are the 3 barycentric coordinates.
                using sofa::component::topology::container::dynamic::lfactorial;

                Real realIntegral = (Real)lfactorial(randomPolynomial[0])*
                    lfactorial(randomPolynomial[1])*lfactorial(randomPolynomial[2]) / (lfactorial((*itio) + 2));
                if (fabs(realIntegral - integral) > 1e-8) {
                    ADD_FAILURE() << "Error in numerical integration on triangle for integration method " << (*itio) <<
                        "  and integration order " << (*itio) << " for polynomial defined by " << randomPolynomial << std::endl
                        << "Got  " << integral << " instead of " << realIntegral << std::endl << "Failed seed number = " << Inherit::seed << std::endl;
                    return false;
                }
            }
        }
        return(true);
    }


    void TearDown() override
    {
        if (root != nullptr)
            sofa::simulation::node::unload(root);
    }

};

// Define the list of DataTypes to instanciate
using ::testing::Types;
typedef Types<
    Vec2Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_SUITE(TriangleNumericalIntegration_test, DataTypes);

// first test topology
TYPED_TEST(TriangleNumericalIntegration_test, testNumericalIntegration)
{
    this->createScene();
    ASSERT_TRUE(this->testNumericalIntegration());

}



} // namespace sofa
