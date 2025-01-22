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

//Including Simulation
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Node.h>
// Including constraint, force and mass
#include <sofa/component/topology/container/dynamic/TetrahedronSetGeometryAlgorithms.h>
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
struct TetrahedronNumericalIntegration_test : public NumericTest<typename _DataTypes::Real>
{
    using Inherit = NumericTest<typename _DataTypes::Real>;

    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename sofa::component::topology::container::dynamic::TetrahedronSetGeometryAlgorithms<DataTypes> TetrahedronSetGeometryAlgorithms;
    typedef typename sofa::component::topology::container::dynamic::NumericalIntegrationDescriptor<Real,4> NumericalIntegrationDescriptor;
//	typedef typename sofa::component::topology::lfactorial lfactorial;
    /// Root of the scene graph
    simulation::Node::SPtr root;
    /// Simulation
    simulation::Simulation* simulation;
    // the geometry algorithm algorithm
    typename sofa::component::topology::container::dynamic::TetrahedronSetGeometryAlgorithms<DataTypes>::SPtr geo;

    // Create the context for the scene
    void doSetUp() override
    {
        // Init simulation
        simulation = sofa::simulation::getSimulation();

         root = simulation::getSimulation()->createNewGraph("root");
    }
     // create the TetrahedronSetGeometryAlgorithms object
    void createScene()
    {
        geo = core::objectmodel::New<TetrahedronSetGeometryAlgorithms>();
        root->addObject(geo);
    }
    bool testNumericalIntegration()
    {
        size_t k;
        Real integral,weight;
        typename NumericalIntegrationDescriptor::BarycentricCoordinatesType bc;
        Vec<4,unsigned short> randomPolynomial;

        // get the descriptor of numerical integration on tetrahedra
        NumericalIntegrationDescriptor &nid=geo->getTetrahedronNumericalIntegrationDescriptor();
        // get all quadrature method
        std::set<typename NumericalIntegrationDescriptor::QuadratureMethod> qmArray=nid.getQuadratureMethods();
        typename std::set<typename NumericalIntegrationDescriptor::QuadratureMethod>::iterator itqm;
        for (itqm=qmArray.begin();itqm!=qmArray.end();itqm++) {
            // get all the integration orders for this integration method
            std::set<typename NumericalIntegrationDescriptor::IntegrationOrder> ioArray=nid.getIntegrationOrders(*itqm);
            typename std::set<typename NumericalIntegrationDescriptor::IntegrationOrder>::iterator itio;
            // go through the integration orders of this method
            for (itio=ioArray.begin();itio!=ioArray.end();itio++) {
                // get the integration point
                typename NumericalIntegrationDescriptor::QuadraturePointArray qpa=nid.getQuadratureMethod(*itqm, *itio);
                /// set the random polynomial defined as u^i*v^j*w^k*r^l where [i,j,k,l] is a random vector given by randomPolynomial
                /// such that i+j+k+l= degree
                for(k=0;k<4;++k) randomPolynomial[k]=0;
                for (k=0;k<(*itio);++k) {
                    randomPolynomial[helper::irand()%4]++;
                }
                // compute the integral over the tetrahedron through numerical integration
                integral=(Real)0;
                for(k=0;k<qpa.size();++k) {
                    typename NumericalIntegrationDescriptor::QuadraturePoint qp=qpa[k];
                    // the barycentric coordinate
                    bc=qp.first;
                    // the weight of the integration point
                    weight=qp.second;
                    integral+=pow(bc[0],randomPolynomial[0])*pow(bc[1],randomPolynomial[1])*
                        pow(bc[2],randomPolynomial[2])*pow(bc[3],randomPolynomial[3])*weight;

                }
                /// real integral value
                /// use the classical integration formula on the tetrahedron with barycentric coordinates i.e.
                /// int_{\tetrahedron} L_1^a  L_2^b  L_3^c  L_4^d dV= (a! b! c! d!) *6V / (a+b+c+d+3)!
                /// where L1 , L2 , L3 , L4 are the 4 barycentric coordinates.
                using sofa::component::topology::container::dynamic::lfactorial;

                Real realIntegral=(Real)lfactorial(randomPolynomial[0])*
                    lfactorial(randomPolynomial[1])*lfactorial(randomPolynomial[2])*
                    lfactorial(randomPolynomial[3])/(lfactorial((*itio)+3));
                if (fabs(realIntegral-integral)>1e-8) {
                    ADD_FAILURE() << "Error in numerical integration on tetrahedron for integration method " <<(*itio)<<
                        "  and integration order " <<(*itio)  << " for polynomial defined by "<< randomPolynomial<< std::endl
                     << "Got  " <<integral<<" instead of " <<realIntegral  << std::endl << "Failed seed number = " << Inherit::seed << std::endl;
                    return false;
                }
            }
        }
        return(true);
    }


    void doTearDown() override
    {
        if (root!=nullptr)
            sofa::simulation::node::unload(root);
    }

};

// Define the list of DataTypes to instantiate
using ::testing::Types;
typedef Types<
    Vec3Types
> DataTypes; // the types to instantiate.

// Test suite for all the instantiations
TYPED_TEST_SUITE(TetrahedronNumericalIntegration_test, DataTypes);

// first test topology
TYPED_TEST( TetrahedronNumericalIntegration_test , testNumericalIntegration )
{
    EXPECT_MSG_NOEMIT(Error, Warning);

    this->createScene();
    ASSERT_TRUE( this->testNumericalIntegration());

}



} // namespace sofa
