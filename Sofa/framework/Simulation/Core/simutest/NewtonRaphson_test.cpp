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


class Simple2DSpring: public sofa::core::behavior::ForceField<sofa::defaulttype::Vec2dTypes>
{
public:
    SOFA_CLASS(Simple2DSpring, sofa::core::behavior::ForceField<sofa::defaulttype::Vec2dTypes>);
    Simple2DSpring()
    : d_k(initData(&d_k, "k", ""))
    , d_l0(initData(&d_l0, "l0", ""))
    {}

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
    {
        sofa::helper::WriteOnlyAccessor< DataVecDeriv > _f = f;
        sofa::helper::ReadAccessor< DataVecCoord > _x = x;
        const auto x_0 = _x[0];
        const auto x_1 = _x[1];
        _f[0] += eval(x_0,x_1);
        _f[1] += -eval(x_0,x_1);
    }

    virtual void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
    {
        auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                           .withRespectToPositionsIn(this->mstate);

        sofa::helper::ReadAccessor< DataVecCoord > _x = this->mstate->readPositions();
        const auto x_0 = _x[0];
        const auto x_1 = _x[1];
        const auto dir = x_0 - x_1;

        const auto k = d_k.getValue();
        const auto l0 = d_l0.getValue();

        for (unsigned i=0; i<2; ++i)
        {
            for (unsigned j=0; j<2; ++j)
            {
                sofa::type::Mat2x2d temp{};
                temp [0][0] = (i!=j ? -1 : 1) * (k * ( l0/(dir.norm()) - 1 ) - k * l0 * pow(dir[0],2.0)/pow(dot(dir, dir), 3.0/2.0)) ;
                temp [0][1] = (i!=j ? -1 : 1) * (- k * l0 * dir[0] * dir[1]/pow(dot(dir, dir), 3.0/2.0)) ;
                temp [1][0] = (i!=j ? -1 : 1) * (- k * l0 * dir[0] * dir[1]/pow(dot(dir, dir), 3.0/2.0)) ;
                temp [1][1] = (i!=j ? -1 : 1) * (k * ( l0/(dir.norm()) - 1 ) - k * l0 * pow(dir[1],2.0)/pow(dot(dir, dir), 3.0/2.0)) ;
                dfdx(Deriv::total_size * i, Deriv::total_size * j) += temp;
            }
        }
    }

    sofa::type::Vec2d eval(sofa::type::Vec2d x_0, sofa::type::Vec2d x_1)
    {
        const auto dir = x_0 - x_1;
        return d_k.getValue() * dir/dir.norm() * (d_l0.getValue() - dir.norm());
    }


    sofa::type::Mat2x2d stiffness(const sofa::type::Vec2d& dir) const
    {
        const SReal k = d_k.getValue();
        const SReal l0 = d_l0.getValue();
        const SReal n = dir.norm();
        sofa::type::Mat2x2d K;
        for (unsigned a=0; a<2; ++a)
            for (unsigned b=0; b<2; ++b)
                K[a][b] = (a==b ? k * (l0/n - 1) : 0.0) - k * l0 * dir[a] * dir[b] / (n*n*n);
        return K;
    }

    // Needed because the integration scheme uses addMBKv (thus addDForce) in its RHS for the
    // Rayleigh stiffness damping and the first-iteration K*r1 term
    virtual void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx )
    {
        sofa::helper::WriteAccessor< DataVecDeriv > _df = df;
        sofa::helper::ReadAccessor< DataVecDeriv > _dx = dx;
        sofa::helper::ReadAccessor< DataVecCoord > _x = this->mstate->readPositions();
        const SReal kFactor = mparams->kFactor();
        const sofa::type::Vec2d dforce = stiffness(_x[0] - _x[1]) * (_dx[0] - _dx[1]) * kFactor;
        _df[0] += dforce;
        _df[1] -= dforce;
    }

    virtual SReal getPotentialEnergy(const sofa::core::MechanicalParams* /*mparams*/, const DataVecCoord& x) const
    { }

    sofa::core::objectmodel::Data<SReal> d_k;
    sofa::core::objectmodel::Data<SReal> d_l0;

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

static std::string create1DScene(const SReal position, const SReal velocity, const SReal m, const SReal g,
                                        const SReal startTime, const SReal dt, const bool lineSearch, const bool staticSolve)
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>"
             "<Node name='Root' gravity='" << g <<" 0 0' time='"<< startTime <<"' animate='0' dt='"<< dt << "' > \n";
    scene << (lineSearch ? "<NewtonRaphson maxNbIterationsLineSearch='20' maxNbIterationsNewton='20' residualThreshold='1e-30' lineSearchReductionRate='0.2'/> "
                         : "<NewtonRaphson maxNbIterationsLineSearch='0' maxNbIterationsNewton='20' residualThreshold='1e-30'  /> ") << " \n";
    scene << (staticSolve ? "   <StaticEquilibriumIntegrationScheme/> \n"  : "   <EulerImplicitIntegrationScheme/> \n" );
    scene << "   <SparseLDLSolver template='CompressedRowSparseMatrixd' />                                                                                        \n"
             "   <PointSetTopologyContainer position='"<<position<<"' />                                             \n"
             "   <MechanicalObject template='Vec1d' name='state' position='"<<position<<"' velocity='"<<velocity<<"' />                                             \n"
             "   <UniformMass totalMass='"<< m <<"'/>                                                                             \n"
             "</Node>                                                                                                       \n" ;
    return scene.str();
}


static std::string create2DScene(const sofa::type::Vec2d position, const sofa::type::Vec2d velocity, const SReal m, const sofa::type::Vec2d  g,
                                        const SReal startTime, const SReal dt, const bool lineSearch, const bool staticSolve)
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>"
             "<Node name='Root' gravity='" << g[0] << " " << g[1] <<" 0' time='"<< startTime <<"' animate='0' dt='"<< dt << "' > \n";
    scene << (lineSearch ? "<NewtonRaphson maxNbIterationsLineSearch='20' maxNbIterationsNewton='20' residualThreshold='1e-30' lineSearchReductionRate='0.2'/> "
                         : "<NewtonRaphson maxNbIterationsLineSearch='0' maxNbIterationsNewton='20' residualThreshold='1e-30'  /> ") << " \n";
    scene << (staticSolve ? "   <StaticEquilibriumIntegrationScheme/> \n"  : "   <EulerImplicitIntegrationScheme rayleighMass='2' rayleighStiffness='0.1'/> \n" );
    scene << "   <SparseLDLSolver template='CompressedRowSparseMatrixd' />                                                                                        \n"
             "   <PointSetTopologyContainer position='"<<position<<" 0 0' />                                             \n"
             "   <MechanicalObject template='Vec2d' name='state' position='"<<position<<" 0 0' velocity='"<<velocity<<" 0 0' />                                             \n"
             "   <UniformMass totalMass='"<< 2*m <<"'/>                                                                             \n"
             "   <FixedProjectiveConstraint indices='1'/>                                                                             \n"
             "</Node>                                                                                                       \n" ;
    return scene.str();
}


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
            Sofa.Component.Constraint.Projective
        });
    }

    void test1D_quadraticFunction_static(bool lineSearch)
    {
        const SReal m = 1.0;
        const SReal g = 9.81;
        const SReal k0 = 10;
        const SReal k1 = 10 * sqrt(m*g + k0);
        const SReal k2 = 1.0;

        const std::string scene = create1DScene(0, 0, m, g, 0, 0 , lineSearch, true);
        SceneInstance c("xml", scene.c_str()) ;

        Node* m_root = c.root.get() ;
        ASSERT_NE(m_root, nullptr) ;

        auto quadraticField = sofa::core::objectmodel::New<Poliynomial1DField>();
        quadraticField->d_factors.setValue({k0, k1, k2});

        m_root->addObject(quadraticField);

        c.initScene() ;

        SReal expectedSolution = (-k1 + sqrt(k1 * k1 - 4*k2*(m*g + k0)))/(2.0*k2);

        sofa::simulation::node::animate(m_root);

        EXPECT_NEAR(m_root->getMechanicalState()->getPX(0),expectedSolution, 1e-10);

    }


    // Without line search, the starting point must be in a region where the stiffness is definite
    // (spring stretched): from a compressed configuration such as {-1, -0.1}, the first full Newton step
    // increases the residual and is rejected by the NewtonRaphson (which then does not move at all).
    void test_2D_linear_spring(bool lineSearch, sofa::type::Vec2d startPosition)
    {
        const SReal m = 2.0;
        const SReal g = 9.81;
        const SReal l0 = 1.5;
        const SReal k0 = 8;

        const std::string scene = create2DScene(startPosition, {0, 0}, m, {0, -g}, 0, 0 , lineSearch, true);
        SceneInstance c("xml", scene.c_str()) ;

        Node* m_root = c.root.get() ;
        ASSERT_NE(m_root, nullptr) ;

        auto simple2DSpring = sofa::core::objectmodel::New<Simple2DSpring>();
        simple2DSpring->d_k.setValue(k0);
        simple2DSpring->d_l0.setValue(l0);

        m_root->addObject(simple2DSpring);

        c.initScene() ;

        sofa::type::Vec2d expectedSolution{0, -(l0 + m*g/k0)};

        sofa::simulation::node::animate(m_root);

        EXPECT_NEAR(m_root->getMechanicalState()->getPX(0),expectedSolution[0], 1e-10);
        EXPECT_NEAR(m_root->getMechanicalState()->getPY(0),expectedSolution[1], 1e-10);
    }


    void test2D_linear_spring_dynamic_step_by_step(bool lineSearch)
    {
        const SReal m = 2.0;
        const SReal g = 9.81;
        const SReal l0 = 1.5;
        const SReal k0 = 8;

        const std::string scene = create2DScene({-1, -0.1}, {0, 0}, m, {0, -g}, 0, 0.01 , lineSearch, false);
        SceneInstance c("xml", scene.c_str()) ;

        Node* m_root = c.root.get() ;
        ASSERT_NE(m_root, nullptr) ;

        auto simple2DSpring = sofa::core::objectmodel::New<Simple2DSpring>();
        simple2DSpring->d_k.setValue(k0);
        simple2DSpring->d_l0.setValue(l0);

        m_root->addObject(simple2DSpring);

        c.initScene() ;

        sofa::type::Vec2d expectedSolution{0, -(l0 + m*g/k0)};

        std::vector<sofa::type::Vec2d> expectedValue{{-1.00019171247996, -0.100982980399615},
                                                  {-1.00056981305708, -0.102931691796746},
                                                  {-1.00112894962203, -0.105829333403968},
                                                  {-1.00186372930715, -0.109659529084676},
                                                  {-1.00276870578739, -0.114406303132725},
                                                  {-1.00383836709484, -0.120054055708613},
                                                  {-1.00506712412284, -0.126587537869115},
                                                  {-1.00644929998999, -0.133991826153435},
                                                  {-1.00797912042896, -0.142252296717049},
                                                  {-1.00965070535706, -0.151354599033696},
                                                  {-1.01145806177669, -0.161284629215909},
                                                  {-1.01339507814142, -0.172028503034245},
                                                  {-1.01545552030903, -0.183572528744312},
                                                  {-1.01763302918493, -0.195903179857918},
                                                  {-1.0199211201383, -0.209007068019343},
                                                  {-1.02231318424942, -0.222870916169143},
                                                  {-1.02480249141989, -0.237481532195052},
                                                  {-1.02738219534862, -0.252825783281977},
                                                  {-1.03004534034588, -0.268890571179989},
                                                  {-1.03278486992674, -0.285662808610343}};

        for(unsigned i=0; i<expectedValue.size(); ++i)
        {
            sofa::simulation::node::animate(m_root);
            EXPECT_NEAR(m_root->getMechanicalState()->getPX(0),expectedValue[i][0], 1e-10);
            EXPECT_NEAR(m_root->getMechanicalState()->getPY(0),expectedValue[i][1], 1e-10);
        }
    }


    void test2D_linear_spring_dynamic(bool lineSearch, unsigned nbStep)
    {
        const SReal m = 2.0;
        const SReal g = 9.81;
        const SReal l0 = 1.5;
        const SReal k0 = 4;

        const std::string scene = create2DScene({-1, -0.1}, {0, 0}, m, {0, -g}, 0, 0.1 , lineSearch, false);
        SceneInstance c("xml", scene.c_str()) ;

        Node* m_root = c.root.get() ;
        ASSERT_NE(m_root, nullptr) ;

        auto simple2DSpring = sofa::core::objectmodel::New<Simple2DSpring>();
        simple2DSpring->d_k.setValue(k0);
        simple2DSpring->d_l0.setValue(l0);

        m_root->addObject(simple2DSpring);

        c.initScene() ;

        sofa::type::Vec2d expectedSolution{0, -(l0 + m*g/k0)};

        for(unsigned i=0; i<nbStep; ++i)
            sofa::simulation::node::animate(m_root);

        EXPECT_NEAR(m_root->getMechanicalState()->getPX(0),expectedSolution[0], 1e-10);
        EXPECT_NEAR(m_root->getMechanicalState()->getPY(0),expectedSolution[1], 1e-10);
    }

    void test1D_liouville_dynamic_compare_analytical( bool lineSearch)
    {

        const SReal t0 = 2;
        const SReal A = 10;
        const SReal m = 1000;
        const SReal startTime = 1.8;
        const SReal finalTime = 2.2;
        const SReal dt1 = 0.002;
        const unsigned nbHDivision = 6;


        std::vector<SReal> pos;
        std::vector<SReal> vel;
        pos.resize(nbHDivision);
        vel.resize(nbHDivision);


        for (unsigned i=0; i<nbHDivision; ++i)
        {
            const SReal newDt=dt1/pow(2,i);
            const std::string scene = create1DScene(LiouvilleForceField::analyticalSolution(A, t0, startTime), LiouvilleForceField::analyticalSolutionDerivative(A, t0, startTime), m, 0, startTime, newDt , lineSearch, false);

            SceneInstance c1("xml", scene) ;

            const unsigned nbStep = round((finalTime - startTime)/newDt);

            Node* m_root1 = c1.root.get() ;
            ASSERT_NE(m_root1, nullptr) ;

            auto liouville = sofa::core::objectmodel::New<LiouvilleForceField>();
            liouville->d_A.setValue(A);
            liouville->d_t0.setValue(t0);
            liouville->d_M.setValue(m);

            m_root1->addObject(liouville);
            c1.initScene();

            auto * realState = static_cast<sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec1dTypes>*>(m_root1->getMechanicalState());
            SReal mechaVel;

            pos[i] = 0;
            vel[i] = 0;

            for (unsigned j=0; j<nbStep; ++j)
            {
                sofa::simulation::node::animate(m_root1, newDt);
                realState->copyToBuffer(&mechaVel, sofa::core::vec_id::write_access::velocity, 1);
                pos[i] += fabs(realState->getPX(0) -LiouvilleForceField::analyticalSolution(A, t0, startTime + newDt * (j+1)));
                vel[i] += fabs(mechaVel - LiouvilleForceField::analyticalSolutionDerivative(A, t0,  startTime + newDt * (j+1)));
            }
            pos[i] /= nbStep;
            vel[i] /= nbStep;

        }

        const std::vector<SReal> expectedSlop{1e-2, 1e-2, 1e-2, 5e-3, 3e-3, 3e-4};

        for (unsigned i=1; i<nbHDivision; ++i)
        {
            EXPECT_NEAR(std::log2(pos[i-1]/pos[i]),1.0,expectedSlop[i]);
            EXPECT_NEAR(std::log2(vel[i-1]/vel[i]),1.0,expectedSlop[i]);
        }

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

TEST_F(NewtonRaphson_test , Linear2DSpring_Static_NoLineSearch )
{
    this->test_2D_linear_spring(false, {-1, -2});
}

TEST_F(NewtonRaphson_test , Linear2DSpring_Static_LineSearch )
{
    this->test_2D_linear_spring(true, {-1, -0.1});
}

TEST_F(NewtonRaphson_test , Linear2DSpring_Dynamic_NoLineSearch )
{
    this->test2D_linear_spring_dynamic(false, 2600);
}

TEST_F(NewtonRaphson_test , Linear2DSpring_Dynamic_LineSearch )
{
    this->test2D_linear_spring_dynamic(true, 2600);
}

TEST_F(NewtonRaphson_test , Linear2DSpring_StepByStep )
{
    this->test2D_linear_spring_dynamic_step_by_step(true);
}

TEST_F(NewtonRaphson_test , Liouville_Dynamic_Analytical )
{
    this->test1D_liouville_dynamic_compare_analytical(false);
}



