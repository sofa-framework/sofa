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
#include <sofa/testing/BaseSimulationTest.h>
#include <sofa/testing/NumericTest.h>

#include <SceneCreator/SceneCreator.h>
#include <sofa/defaulttype/VecTypes.h>

//Including Simulation
#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/component/solidmechanics/tensormass/TetrahedralTensorMassForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedralCorotationalFEMForceField.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/mechanicalload/TrianglePressureForceField.h>
#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>
#include <sofa/component/engine/select/BoxROI.h>
#include <sofa/component/engine/generate/GenerateCylinder.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/component/mass/MeshMatrixMass.h>
#include <sofa/component/odesolver/backward/StaticSolver.h>
#include <sofa/component/constraint/projective/FixedProjectiveConstraint.h>
#include <sofa/component/constraint/projective/FixedPlaneProjectiveConstraint.h>
#include <sofa/component/constraint/projective/LineProjectiveConstraint.h>
#include <sofa/simulation/DefaultAnimationLoop.h>

namespace sofa {

using namespace component;
using namespace defaulttype;
using namespace modeling;

const double pressureArray[] = {0.6, 0.2, -0.2, -0.6};
const size_t sizePressureArray = sizeof(pressureArray)/sizeof(pressureArray[0]);

const double youngModulusArray[] = {1.0,2.0};
const size_t sizeYoungModulusArray = sizeof(youngModulusArray)/sizeof(youngModulusArray[0]);

const double poissonRatioArray[] = {0.0,0.3,0.49};
const size_t sizePoissonRatioArray = sizeof(poissonRatioArray)/sizeof(poissonRatioArray[0]);

/// Structure which contains the nodes and the pointers useful for the cylindertraction test
template<class T>
struct CylinderTractionStruct
{
    simulation::Node::SPtr root;
    typename component::statecontainer::MechanicalObject<T>::SPtr dofs;
    typename component::mechanicalload::TrianglePressureForceField<T>::SPtr forceField;
};

template <typename DataTypes>
CylinderTractionStruct<DataTypes>  createCylinderTractionScene(
        int resolutionCircumferential,
        int resolutionRadial,
        int resolutionHeight,
        int maxIter)
{
    // Definitions
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename component::statecontainer::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename component::engine::select::BoxROI<DataTypes> BoxRoi;
    typedef component::linearsolver::iterative::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;
    typename simulation::Node::SPtr root;
    CylinderTractionStruct<DataTypes> tractionStruct;

    // Root node
    root = sofa::simulation::getSimulation()->createNewGraph("root");
    tractionStruct.root=root;

    root->setGravity( Coord(0,0,0) );
    root->setAnimate(false);
    root->setDt(0.05);

    sofa::modeling::addNew<sofa::simulation::DefaultAnimationLoop>(root, "animationLoop");

    // GenerateCylinder object
    typename sofa::component::engine::generate::GenerateCylinder<DataTypes>::SPtr eng= sofa::modeling::addNew<sofa::component::engine::generate::GenerateCylinder<DataTypes> >(root,"cylinder");
    eng->f_radius=0.2;
    eng->f_height=1.0;
    eng->f_resolutionCircumferential=resolutionCircumferential;
    eng->f_resolutionRadial=resolutionRadial;
    eng->f_resolutionHeight=resolutionHeight;
    // TetrahedronSetTopologyContainer object
    const typename sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer::SPtr container1= sofa::modeling::addNew<sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>(root,"Container1");
    sofa::modeling::setDataLink(&eng->f_tetrahedra,&container1->d_tetrahedron);
    sofa::modeling::setDataLink(&eng->f_outputTetrahedraPositions,&container1->d_initPoints);
    container1->d_createTriangleArray=true;
    // TetrahedronSetGeometryAlgorithms object
    typename sofa::component::topology::container::dynamic::TetrahedronSetGeometryAlgorithms<DataTypes>::SPtr geo1= sofa::modeling::addNew<sofa::component::topology::container::dynamic::TetrahedronSetGeometryAlgorithms<DataTypes> >(root);

    // CGLinearSolver
    const typename CGLinearSolver::SPtr cgLinearSolver = modeling::addNew< CGLinearSolver >(root,"linearSolver");
    cgLinearSolver->d_maxIter.setValue(maxIter);
    cgLinearSolver->d_tolerance.setValue(1e-9);
    cgLinearSolver->d_smallDenominatorThreshold.setValue(1e-9);
    // StaticSolver
    typename component::odesolver::backward::StaticSolver::SPtr solver = modeling::addNew<component::odesolver::backward::StaticSolver>(root,"StaticSolver");
    // mechanicalObject object
    typename MechanicalObject::SPtr meca1= sofa::modeling::addNew<MechanicalObject>(root);
    sofa::modeling::setDataLink(&eng->f_outputTetrahedraPositions,&meca1->x);
    tractionStruct.dofs=meca1;
    // MeshMatrixMass
    typename sofa::component::mass::MeshMatrixMass<DataTypes>::SPtr mass= sofa::modeling::addNew<sofa::component::mass::MeshMatrixMass<DataTypes> >(root,"BezierMass");
    sofa::type::vector< Real > massDensity;
    massDensity.clear();
    massDensity.resize(1);
    massDensity[0] = 1.0;
    mass->d_massDensity.setValue(massDensity);
    mass->d_lumping=false;
    /// box fixed
    type::vector< type::Vec<6,Real> > vecBox;
    type::Vec<6,Real> box;
    box[0]= -0.01;box[1]= -0.01;box[2]= -0.01;box[3]= 0.01;box[4]= 0.01;box[5]= 0.01;
    vecBox.push_back(box);
    typename BoxRoi::SPtr boxRoi1 = modeling::addNew<BoxRoi>(root,"boxRoiFix");
    boxRoi1->d_alignedBoxes.setValue(vecBox);
    boxRoi1->d_strict.setValue(false);
    // FixedProjectiveConstraint
    typename component::constraint::projective::FixedProjectiveConstraint<DataTypes>::SPtr fc=
        modeling::addNew<typename component::constraint::projective::FixedProjectiveConstraint<DataTypes> >(root);
    sofa::modeling::setDataLink(&boxRoi1->d_indices,&fc->d_indices);
    // FixedPlaneProjectiveConstraint
    typename component::constraint::projective::FixedPlaneProjectiveConstraint<DataTypes>::SPtr fpc=
            modeling::addNew<typename component::constraint::projective::FixedPlaneProjectiveConstraint<DataTypes> >(root);
    fpc->d_dmin= -0.01;
    fpc->d_dmax= 0.01;
    fpc->d_direction=Coord(0,0,1);
    /// box pressure
    box[0]= -0.2;box[1]= -0.2;box[2]= 0.99;box[3]= 0.2;box[4]= 0.2;box[5]= 1.01;
    vecBox[0]=box;
    typename BoxRoi::SPtr boxRoi2 = modeling::addNew<BoxRoi>(root,"boxRoiPressure");
    boxRoi2->d_alignedBoxes.setValue(vecBox);
    boxRoi2->d_computeTriangles=true;
    boxRoi2->d_strict.setValue(false);
    /// TrianglePressureForceField
    typename component::mechanicalload::TrianglePressureForceField<DataTypes>::SPtr tpff=
            modeling::addNew<typename component::mechanicalload::TrianglePressureForceField<DataTypes> >(root);
    tractionStruct.forceField=tpff;
    sofa::modeling::setDataLink(&boxRoi2->d_triangleIndices,&tpff->triangleList);
    // LineProjectiveConstraint
    typename component::constraint::projective::LineProjectiveConstraint<DataTypes>::SPtr ptlc=
            modeling::addNew<typename component::constraint::projective::LineProjectiveConstraint<DataTypes> >(root);
    ptlc->f_direction=Coord(1,0,0);
    ptlc->f_origin=Coord(0,0,0);
    sofa::type::vector<sofa::Index> vArray;
    vArray.push_back(resolutionCircumferential*(resolutionRadial-1)+1);
    ptlc->f_indices.setValue(vArray);

    return tractionStruct;
}

/**  Test force fields implementing linear elasticity on tetrahedral mesh.
Implement traction applied on the top part of a cylinder and test that the deformation
is simply related with the Young Modulus and Poisson Ratio of the isotropc linear elastic material */

template <typename _DataTypes>
struct LinearElasticity_test : public sofa::testing::BaseSimulationTest, sofa::testing::NumericTest<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename statecontainer::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename sofa::component::solidmechanics::tensormass::TetrahedralTensorMassForceField<DataTypes> TetrahedralTensorMassForceField;
    typedef typename sofa::core::behavior::ForceField<DataTypes>::SPtr ForceFieldSPtr;
    typedef ForceFieldSPtr (LinearElasticity_test<_DataTypes>::*LinearElasticityFF)(simulation::Node::SPtr,double,double);
    /// Simulation
    simulation::Simulation* simulation;
    /// struct with the pointer of the main components
    CylinderTractionStruct<DataTypes> tractionStruct;
    /// index of the vertex used to compute the compute the deformation
    size_t vIndex;


    // Create the context for the scene
    void SetUp()
    {
        // Init simulation
        simulation = sofa::simulation::getSimulation();
        const size_t resolutionCircumferential=7;
        const size_t  resolutionRadial=3;
        const size_t  resolutionHeight=7;
        const size_t maxIteration=3000; // maximum iteration for the CG.

        tractionStruct= createCylinderTractionScene<_DataTypes>(resolutionCircumferential,resolutionRadial,
            resolutionHeight,maxIteration);
        /// take the vertex at mid height and on the surface of the cylinder
        vIndex=(resolutionCircumferential*(resolutionRadial-1)+1)*resolutionHeight/2;
    }
    ForceFieldSPtr addTetrahedralLinearElastic(simulation::Node::SPtr root,
        double youngModulus,double poissonRatio)
    {
        typename TetrahedralTensorMassForceField::SPtr ff=addNew<TetrahedralTensorMassForceField>(root);
        ff->setYoungModulus(youngModulus);
        ff->setPoissonRatio(poissonRatio);
        return (ForceFieldSPtr )ff;
    }
    ForceFieldSPtr addTetrahedralCorotationalFEMLinearElastic(simulation::Node::SPtr root,
        double youngModulus,double poissonRatio)
    {
        typename sofa::component::solidmechanics::fem::elastic::TetrahedralCorotationalFEMForceField<DataTypes>::SPtr ff=addNew<sofa::component::solidmechanics::fem::elastic::TetrahedralCorotationalFEMForceField<DataTypes> >(root);
        ff->setYoungModulus(youngModulus);
        ff->setPoissonRatio(poissonRatio);
        ff->setMethod(0); // small method
        return (ForceFieldSPtr )ff;
    }
    bool testLinearElasticityInTraction(LinearElasticityFF createForceField){

        sofa::simulation::node::initRoot(tractionStruct.root.get());

        size_t i,j,k;
        for (k=0;k<sizeYoungModulusArray;++k) {
            Real youngModulus=youngModulusArray[k];
            for (j=0;j<sizePoissonRatioArray;++j) {
                Real poissonRatio=poissonRatioArray[j];
                // create the linear elasticity force field
                ForceFieldSPtr ff=(this->*createForceField)(tractionStruct.root,youngModulus,poissonRatio);
                ff->init();

                for (i=0;i<sizePressureArray;++i) {
                    // set the pressure on the top part
                    Real pressure= pressureArray[i];

                    tractionStruct.forceField.get()->setPressure(Coord(0, 0, pressure));
                    sofa::simulation::node::reset(tractionStruct.root.get());
                    
                    // record the initial point of a given vertex
                    Coord p0=tractionStruct.dofs.get()->read(sofa::core::ConstVecCoordId::position())->getValue()[vIndex];

                    //  do one step of the static solver
                    sofa::simulation::node::animate(tractionStruct.root.get(), 0.5_sreal);

                    // Get the simulated final position of that vertex
                    Coord p1=tractionStruct.dofs.get()->read(sofa::core::ConstVecCoordId::position())->getValue()[vIndex];
                    // test the young modulus
                    Real longitudinalDeformation=(p1[2]-p0[2])/p0[2];
                    if (fabs(longitudinalDeformation-pressure/youngModulus)>1e-4) {
                        ADD_FAILURE() << "Wrong longitudinal deformation for Young Modulus = " << youngModulus << " Poisson Ratio = "<<
                            poissonRatio << " pressure= "<<pressure<< std::endl <<
                            "Got "<<longitudinalDeformation<< " instead of "<< pressure/youngModulus<< std::endl;
                        return false;
                    }
                    // compute radial deformation
                    p0[2]=0;
                    p1[2]=0;
                    Real radius=p0.norm2();
                    Real radialDeformation= dot(p0,p1)/radius-1 ;
                    // test the Poisson Ratio
                    if (fabs(radialDeformation+pressure*poissonRatio/youngModulus)>2e-4) {
                        ADD_FAILURE() << "Wrong radial deformation for Young Modulus = " << youngModulus << " Poisson Ratio = "<<
                            poissonRatio << " pressure= "<<pressure<< std::endl <<
                            "Got "<<radialDeformation<< " instead of "<< -pressure*poissonRatio/youngModulus<< std::endl;
                        return false;
                    }
                }
                tractionStruct.root->removeObject(ff);
            }
        }
        return true;
    }
    void TearDown()
    {
        if (tractionStruct.root!=nullptr)
            sofa::simulation::node::unload(tractionStruct.root);
    }

};

// Define the list of DataTypes to instanciate
using ::testing::Types;
typedef Types<
    Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_SUITE(LinearElasticity_test, DataTypes);

// first test topology
TYPED_TEST( LinearElasticity_test , testTractionTensorMass )
{
    //	this->loadScene( "tests/SofaTest/LinearElasticity.scn");
    ASSERT_TRUE( this->testLinearElasticityInTraction(&sofa::LinearElasticity_test<TypeParam>::addTetrahedralLinearElastic));
}

TYPED_TEST( LinearElasticity_test , testTractionCorotational )
{
//	this->loadScene( "tests/SofaTest/LinearElasticity.scn");
    ASSERT_TRUE( this->testLinearElasticityInTraction(&sofa::LinearElasticity_test<TypeParam>::addTetrahedralCorotationalFEMLinearElastic));
}

} // namespace sofa
