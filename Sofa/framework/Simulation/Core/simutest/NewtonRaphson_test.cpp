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
#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;
using sofa::simulation::Node ;


#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <sofa/simpleapi/SimpleApi.h>

#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/objectmodel/SPtr.h>

class Quadratic1DField: public sofa::core::behavior::ForceField<sofa::defaulttype::Vec1dTypes>
{
public:
    SOFA_CLASS(Quadratic1DField, sofa::core::behavior::ForceField<sofa::defaulttype::Vec1dTypes>);
    Quadratic1DField()
    : d_k1(initData(&d_k1, "k1", ""))
    , d_k2(initData(&d_k2, "k2", ""))
    {}

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
    {
        sofa::helper::WriteOnlyAccessor< DataVecDeriv > _f = f;
        sofa::helper::ReadAccessor< DataVecCoord > _x = x;
        const auto x_0 = _x[0][0];
        _f[0] += sofa::type::Vec1d(x_0*x_0 + d_k1.getValue()*x_0 + d_k2.getValue());
    }

    virtual void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
    {
        auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                           .withRespectToPositionsIn(this->mstate);


        sofa::helper::ReadAccessor< DataVecCoord > _x = this->mstate->readPositions();
        const auto x_0 = _x[0][0];
        dfdx(0,0) += 2.0 * x_0 + d_k1.getValue();

    }

    virtual void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx )
    { }

    virtual SReal getPotentialEnergy(const sofa::core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
    { }

    sofa::core::objectmodel::Data<SReal> d_k1;
    sofa::core::objectmodel::Data<SReal> d_k2;

};



class Antagonists2DLinearSprings: public sofa::core::behavior::ForceField<sofa::defaulttype::Vec2dTypes>
{
public:
    SOFA_CLASS(Antagonists2DLinearSprings, sofa::core::behavior::ForceField<sofa::defaulttype::Vec2dTypes>);
    Antagonists2DLinearSprings()
    : d_k1(initData(&d_k1, "k1", ""))
    , d_k2(initData(&d_k2, "k2", ""))
    {}

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
    {
        // sofa::helper::WriteOnlyAccessor< DataVecDeriv > _f = f;
        // sofa::helper::ReadAccessor< DataVecCoord > _x = x;
        // const auto x_0 = _x[0][0];
        // _f[0] += sofa::type::Vec1d(x_0*x_0 + d_k1.getValue()*x_0 + d_k2.getValue());
    }

    virtual void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
    {
        // auto dfdx = matrix->getForceDerivativeIn(this->mstate)
        //                    .withRespectToPositionsIn(this->mstate);
        //
        //
        // sofa::helper::ReadAccessor< DataVecCoord > _x = this->mstate->readPositions();
        // const auto x_0 = _x[0][0];
        // dfdx(0,0) += 2.0 * x_0 + d_k1.getValue();

    }

    virtual void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx )
    { }

    virtual SReal getPotentialEnergy(const sofa::core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
    { }

    sofa::core::objectmodel::Data<SReal> d_k1;
    sofa::core::objectmodel::Data<SReal> d_k2;

};

class NewtonRaphson_test: public BaseSimulationTest
{
public:


    void doSetUp() override
    {
        this->loadPlugins({
            Sofa.Component.StateContainer,
            Sofa.Component.SceneUtility,
            Sofa.Component.IntegrationScheme,
            Sofa.Component.LinearSolver.Direct,
            Sofa.Component.Topology.Container.Dynamic,
            Sofa.Component.Mass,
        });
    }

    void test1D_quadraticFunction_static(bool lineSearch)
    {
        const SReal m = 1.0;
        const SReal g = 9.81;
        const SReal k2 = 10;
        const SReal k1 = 10 * sqrt(m*g + k2);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='" << g  <<" 0 0' time='0' animate='0' > \n";
        scene << (lineSearch ? "<NewtonRaphson maxNbIterationsLineSearch='20' maxNbIterationsNewton='3' residueThreshold='1e-10' lineSearchReductionRate='0.2' /> "
                             : "<NewtonRaphson maxNbIterationsLineSearch='0' maxNbIterationsNewton='3' residueThreshold='1e-10' /> ") << " \n";
        scene << "   <StaticEquilibriumIntegrationScheme />                                                                     \n"
                 "   <SparseLDLSolver template='CompressedRowSparseMatrixd' />                                                                                        \n"
                 "   <PointSetTopologyContainer position='0'/>                                             \n"
                 "   <MechanicalObject template='Vec1d' name='state' position='0'/>                                             \n"
                 "   <UniformMass totalMass='"<< m <<"'/>                                                                             \n"
                 "</Node>                                                                                                       \n" ;

        SceneInstance c("xml", scene.str()) ;

        Node* m_root = c.root.get() ;
        ASSERT_NE(m_root, nullptr) ;

        auto quadraticField = sofa::core::objectmodel::New<Quadratic1DField>();
        quadraticField->d_k2.setValue(k2);
        quadraticField->d_k1.setValue(k1);

        m_root->addObject(quadraticField);

        c.initScene() ;

        SReal expectedSolution = (-k1 + sqrt(k1 * k1 - 4*(m*g + k2)))/2.0;

        sofa::simulation::node::animate(m_root, 1);

        EXPECT_NEAR(m_root->getMechanicalState()->getPX(0),expectedSolution, 1e-10);

    }


    void test1D_quadraticFunction_dynamic_step_by_step(unsigned nbSteps = 10)
    {
        const SReal m = 1.0;
        const SReal g = 9.81;
        const SReal k2 = 10;
        const SReal k1 = 10 * sqrt(m*g + k2);

        const SReal dt1 = 1;



        for (unsigned j=0; j<nbSteps; ++j)
        {
            std::stringstream scene ;
            scene << "<?xml version='1.0'?>"
                     "<Node name='Root' gravity='" << g  <<" 0 0' time='0' dt='"<< dt1 <<"' animate='0' > \n"
                     "<NewtonRaphson maxNbIterationsLineSearch='0' maxNbIterationsNewton='" << j+1 << "' residueThreshold='1e-10' printLog='true' />   \n"
                     "   <EulerImplicitIntegrationScheme />                                                                     \n"
                     "   <SparseLDLSolver template='CompressedRowSparseMatrixd' />                                                                                        \n"
                     "   <PointSetTopologyContainer position='0'/>                                             \n"
                     "   <MechanicalObject template='Vec1d' name='state' position='0'/>                                             \n"
                     "   <UniformMass totalMass='"<< m <<"'/>                                                                             \n"
                     "</Node>                                                                                                       \n" ;

            SceneInstance c1("xml", scene.str()) ;

            Node* m_root1 = c1.root.get() ;
            ASSERT_NE(m_root1, nullptr) ;

            auto quadraticField1 = sofa::core::objectmodel::New<Quadratic1DField>();
            quadraticField1->d_k2.setValue(k2);
            quadraticField1->d_k1.setValue(k1);

            m_root1->addObject(quadraticField1);

            c1.initScene() ;

            sofa::simulation::node::animate(m_root1, dt1);
            std::cout<<"Step #"<< j + 1 <<" "<<m_root1->getMechanicalState()->getPX(0)<<std::endl;

        }



    }

    void test1D_quadraticFunction_dynamic(bool lineSearch)
    {
        const SReal m = 1.0;
        const SReal g = 9.81;
        const SReal k2 = 10;
        const SReal k1 = 10 * sqrt(m*g + k2);

        const unsigned dtRatio = 10;
        const SReal dt1 = 0.01;
        const SReal dt2 = dtRatio * dt1;

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='" << g  <<" 0 0' time='0' animate='0' > \n";
        scene << (lineSearch ? "<NewtonRaphson maxNbIterationsLineSearch='20' maxNbIterationsNewton='200' residueThreshold='1e-10' lineSearchReductionRate='0.2' /> "
                             : "<NewtonRaphson maxNbIterationsLineSearch='0' maxNbIterationsNewton='200' residueThreshold='1e-10' /> ") << " \n";
        scene << "   <EulerImplicitIntegrationScheme />                                                                     \n"
                 "   <SparseLDLSolver template='CompressedRowSparseMatrixd' />                                                                                        \n"
                 "   <PointSetTopologyContainer position='0'/>                                             \n"
                 "   <MechanicalObject template='Vec1d' name='state' position='0'/>                                             \n"
                 "   <UniformMass totalMass='"<< m <<"'/>                                                                             \n"
                 "</Node>                                                                                                       \n" ;

        SceneInstance c1("xml", scene.str()) ;
        SceneInstance c2("xml", scene.str()) ;

        Node* m_root1 = c1.root.get() ;
        ASSERT_NE(m_root1, nullptr) ;

        auto quadraticField1 = sofa::core::objectmodel::New<Quadratic1DField>();
        quadraticField1->d_k2.setValue(k2);
        quadraticField1->d_k1.setValue(k1);

        m_root1->addObject(quadraticField1);

        c1.initScene() ;


        Node* m_root2 = c2.root.get() ;
        ASSERT_NE(m_root2, nullptr) ;

        auto quadraticField2 = sofa::core::objectmodel::New<Quadratic1DField>();
        quadraticField2->d_k2.setValue(k2);
        quadraticField2->d_k1.setValue(k1);

        m_root2->addObject(quadraticField2);

        c2.initScene() ;

        for (unsigned i=0; i<dtRatio; ++i)
        {
            sofa::simulation::node::animate(m_root1, dt1);
        }
        sofa::simulation::node::animate(m_root2, dt2);
        EXPECT_NEAR(m_root1->getMechanicalState()->getPX(0),m_root2->getMechanicalState()->getPX(0),1e-5);


    }


    void test2D_twoLinearSpringsPerpendicularToGravity()
    {

    }

};

TEST_F(NewtonRaphson_test , Quadratic1DSpring_Static_NoLineSearch )
{
    this->test1D_quadraticFunction_static(false);
}

TEST_F(NewtonRaphson_test , Quadratic1DSpring_Static_LineSearch )
{
    this->test1D_quadraticFunction_static(true);
}

TEST_F(NewtonRaphson_test , Quadratic1DSpring_Dynamic_StepByStep )
{
    this->test1D_quadraticFunction_dynamic_step_by_step();
}


TEST_F(NewtonRaphson_test , Quadratic1DSpring_Dynamic_NoLineSearch )
{
    this->test1D_quadraticFunction_dynamic(false);
}

TEST_F(NewtonRaphson_test , Quadratic1DSpring_Dynamic_LineSearch )
{
    this->test1D_quadraticFunction_dynamic(true);
}

// TEST_F(NewtonRaphson_test , TwoLinearSpringsPerpendicularToGravity )
// {
//     this->test2D_twoLinearSpringsPerpendicularToGravity();
// }
