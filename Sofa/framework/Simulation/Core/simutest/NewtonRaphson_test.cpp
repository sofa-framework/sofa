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


#include <cmath>
#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <sofa/simpleapi/SimpleApi.h>

#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/objectmodel/SPtr.h>

class Poliynomial1DField: public sofa::core::behavior::ForceField<sofa::defaulttype::Vec1dTypes>
{
public:
    SOFA_CLASS(Poliynomial1DField, sofa::core::behavior::ForceField<sofa::defaulttype::Vec1dTypes>);
    Poliynomial1DField()
    : d_factors(initData(&d_factors, "factors", ""))
    {}

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
    {
        sofa::helper::WriteOnlyAccessor< DataVecDeriv > _f = f;
        sofa::helper::ReadAccessor< DataVecCoord > _x = x;
        const auto x_0 = _x[0][0];
        _f[0] += sofa::type::Vec1d(eval(x_0));
    }

    virtual void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
    {
        auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                           .withRespectToPositionsIn(this->mstate);


        sofa::helper::ReadAccessor< DataVecCoord > _x = this->mstate->readPositions();
        const auto x_0 = _x[0][0];
        dfdx(0,0) += evalDeriv(x_0);

    }

    SReal eval(SReal x)
    {
        const unsigned order = d_factors.getValue().size() - 1;
        SReal evaluation = 0.0;
        for (unsigned i= 0 ; i<=order; ++i)
        {
            if (i==0)
                evaluation += d_factors.getValue()[i];
            else
                evaluation += pow(x, i) * d_factors.getValue()[i];
        }
        return evaluation;
    }

    SReal evalDeriv(SReal x)
    {
        const unsigned order = d_factors.getValue().size() - 1;
        SReal evaluation = 0.0;
        for (unsigned i= 1 ; i<=order; ++i)
        {
            if (i==1)
                evaluation += d_factors.getValue()[i];
            else
                evaluation += pow(x, i - 1 ) * d_factors.getValue()[i] * i;
        }
        return evaluation;
    }

    virtual void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx )
    { }

    virtual SReal getPotentialEnergy(const sofa::core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
    { }

    sofa::core::objectmodel::Data<sofa::type::vector<SReal>> d_factors;

};

/***
 * Equation of the form \frac{\partial^2 x}{\partial t ^2} + e^{2x}=0
 * The analytical solution is of the form
 *      x(t)                             = log(A / \cosh(A*(t - t_0)))
 *      frac{\partial x}{\partial t}     = -A*\tanh(A*(t-t_0))
 *      frac{\partial^2 x}{\partial t^2} = -A^2*(1-\tanh^2(A*(t-t_0)))
 * Using this function one can check the capability of the Newton-Raphson solver to follow the right trajectory
 ***/
class LiouvilleForceField: public sofa::core::behavior::ForceField<sofa::defaulttype::Vec1dTypes>
{
public:
    SOFA_CLASS(LiouvilleForceField, sofa::core::behavior::ForceField<sofa::defaulttype::Vec1dTypes>);
    LiouvilleForceField()
    : d_A(initData(&d_A, "A", ""))
    , d_t0(initData(&d_t0, "t0", ""))
    , d_M(initData(&d_M, "M", ""))
    {}

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
    {
        sofa::helper::WriteOnlyAccessor< DataVecDeriv > _f = f;
        sofa::helper::ReadAccessor< DataVecCoord > _x = x;
        const auto x_0 = _x[0][0];
        _f[0] += sofa::type::Vec1d(eval(x_0));
    }


    virtual void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
    {
        auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                           .withRespectToPositionsIn(this->mstate);


        sofa::helper::ReadAccessor< DataVecCoord > _x = this->mstate->readPositions();
        const auto x_0 = _x[0][0];
        dfdx(0,0) += evalDeriv(x_0);
    }

    SReal eval(SReal x)
    {
        return -d_M.getValue()*exp(2 * x);
    }

    SReal evalDeriv(SReal x)
    {
        return  2*eval(x);
    }

    static SReal analyticalSolution(const SReal A, const SReal t0, const SReal t)
    {
        return log(A / cosh(A*(t - t0)));
    }

    static SReal analyticalSolutionDerivative(const SReal A, const SReal t0, const SReal t)
    {
        return -A * tanh(A*(t - t0));
    }

    virtual void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx )
    { }

    virtual SReal getPotentialEnergy(const sofa::core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
    { }

    sofa::core::objectmodel::Data<SReal> d_A;
    sofa::core::objectmodel::Data<SReal> d_t0;
    sofa::core::objectmodel::Data<SReal> d_M;

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
        const SReal k0 = 10;
        const SReal k1 = 10 * sqrt(m*g + k1);
        const SReal k2 = 1.0;

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

        auto quadraticField = sofa::core::objectmodel::New<Poliynomial1DField>();
        quadraticField->d_factors.setValue({k0, k1, k2});

        m_root->addObject(quadraticField);

        c.initScene() ;

        SReal expectedSolution = (-k1 + sqrt(k1 * k1 - 4*k2*(m*g + k0)))/(2.0*k2);

        sofa::simulation::node::animate(m_root, 1);

        EXPECT_NEAR(m_root->getMechanicalState()->getPX(0),expectedSolution, 1e-10);

    }



    void test1D_liouville_dynamic_compare_analytical(unsigned nSteps, bool lineSearch)
    {

        const SReal t0 = 2;
        const SReal A = 10;
        const SReal m = 1000;
        const SReal startTime = 1.7;

        const SReal dt1 = 0.002;


        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='0 0 0' time='"<< startTime <<"' animate='0' dt='"<< dt1 << "' > \n";
        scene << (lineSearch ? "<NewtonRaphson maxNbIterationsLineSearch='0' maxNbIterationsNewton='2' residueThreshold='0' lineSearchReductionRate='0.2' printLog='true'/> "
                             : "<NewtonRaphson maxNbIterationsLineSearch='0' maxNbIterationsNewton='200' residueThreshold='0'  /> ") << " \n";
        scene << "   <EulerImplicitIntegrationScheme/>                                                                     \n"
                 "   <SparseLDLSolver template='CompressedRowSparseMatrixd' />                                                                                        \n"
                 "   <PointSetTopologyContainer position='"<<LiouvilleForceField::analyticalSolution(A, t0, startTime)<<"' />                                             \n"
                 "   <MechanicalObject template='Vec1d' name='state' position='"<<LiouvilleForceField::analyticalSolution(A, t0, startTime)<<"' velocity='"<<LiouvilleForceField::analyticalSolutionDerivative(A, t0, startTime)<<"' />                                             \n"
                 "   <UniformMass totalMass='"<< m <<"'/>                                                                             \n"
                 "</Node>                                                                                                       \n" ;

        SceneInstance c1("xml", scene.str()) ;
        SceneInstance c2("xml", scene.str()) ;

        Node* m_root1 = c1.root.get() ;
        ASSERT_NE(m_root1, nullptr) ;

        auto liouville = sofa::core::objectmodel::New<LiouvilleForceField>();
        liouville->d_A.setValue(A);
        liouville->d_t0.setValue(t0);
        liouville->d_M.setValue(m);

        m_root1->addObject(liouville);


        c1.initScene() ;

        std::vector<SReal> pos;
        std::vector<SReal> vel;
        pos.resize(nSteps);
        vel.resize(nSteps);

        auto * realState = static_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes>*>(m_root1->getMechanicalState());

        SReal mechaVel;
        realState->copyToBuffer(&mechaVel, sofa::core::vec_id::write_access::velocity, 1);

        EXPECT_NEAR(realState->getPX(0),LiouvilleForceField::analyticalSolution(A, t0, m_root1->getTime()),1e-5);
        EXPECT_NEAR(mechaVel,LiouvilleForceField::analyticalSolutionDerivative(A, t0, m_root1->getTime()),1e-5);
        pos[0] = realState->getPX(0);
        vel[0] = mechaVel;

        for (unsigned i=0; i<nSteps; ++i)
        {
            sofa::simulation::node::animate(m_root1, dt1);
            realState->copyToBuffer(&mechaVel, sofa::core::vec_id::write_access::velocity, 1);

            EXPECT_NEAR(realState->getPX(0),LiouvilleForceField::analyticalSolution(A, t0, m_root1->getTime()),1e-5);
            EXPECT_NEAR(mechaVel,LiouvilleForceField::analyticalSolutionDerivative(A, t0, m_root1->getTime()),1e-5);

            pos[i+1] = realState->getPX(0);
            vel[i+1] = mechaVel;
        }

        std::cout<<"NR_pos = np.array([";
        for (unsigned i=0; i<nSteps-1; ++i)
        {
            std::cout << pos[i] << ", ";
        }
        std::cout << pos[nSteps]<<"])"<<std::endl;
        std::cout<<"NR_vel = np.array([";
        for (unsigned i=0; i<nSteps-1; ++i)
        {
            std::cout << vel[i] << ", ";
        }
        std::cout << vel[nSteps]<<"])"<<std::endl;


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

TEST_F(NewtonRaphson_test , Liouville_Dynamic_Analytical )
{
    this->test1D_liouville_dynamic_compare_analytical(300, true);
}



