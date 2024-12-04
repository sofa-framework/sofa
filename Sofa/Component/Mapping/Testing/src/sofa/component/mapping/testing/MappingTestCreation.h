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
#pragma once

#include <sofa/core/BaseLocalMappingMatrix.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest;

#include <sofa/simulation/VectorOperations.h>

#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/core/Mapping.h>

#include <sofa/helper/logging/Messaging.h>

namespace sofa::mapping_test
{


/** @brief Base class for the Mapping tests, with helpers to automatically test applyJ, applyJT, applyDJT and getJs using finite differences.

  Specific test cases can be created using a derived class instantiated on the mapping class to test,
  and calling function runTest( const VecCoord_t<In>& parentInit,
                  const VecCoord_t<Out>& childInit,
                  const VecCoord_t<In> parentNew,
                  const VecCoord_t<Out> expectedChildNew);


  This function compares the actual output positions with the expected ones, then automatically tests the methods related to
  the Jacobian using finite differences.
  - A small change of the input positions dxIn is randomly chosen and added to the current position. The same is set as velocity.
  - mapping->apply is called, and the difference dXout between the new output positions and the previous positions is computed
  - to validate mapping->applyJ, dXin is converted to input velocity vIn and mapping->applyJ is called. dXout and the output velocity vOut must be the same (up to linear approximations errors, thus we apply a very small change of position).
  - to validate mapping->getJs, we use it to get the Jacobian, then we check that J.vIn = vOut
  - to validate mapping->applyJT, we apply it after setting the child force fc=vOut, then we check that parent force fp = J^T.fc
  - to validate mapping->applyDJT, we set the child force, and we compare the parent force before and after a small displacement

  The magnitude of the small random changes applied in finite differences is between deltaRange.first*epsilon and deltaRange.second*epsilon,
  and a failure is issued if the error is greater than errorMax*epsilon,
  where epsilon=std::numeric_limits<Real>::epsilon() is 1.19209e-07 for float and 2.22045e-16 for double.

  @author François Faure @date 2013
  */
template< class _Mapping>
struct Mapping_test: public BaseSimulationTest, NumericTest<typename _Mapping::In::Real>
{
    using Mapping = _Mapping;

    using In = typename Mapping::In;
    using InDOFs = component::statecontainer::MechanicalObject<In>;

    using Out = typename Mapping::Out;
    using OutDOFs = component::statecontainer::MechanicalObject<Out>;

    typedef linearalgebra::EigenSparseMatrix<In,Out> EigenSparseMatrix;

    core::Mapping<In,Out>* mapping; ///< the mapping to be tested
    typename InDOFs::SPtr  inDofs;  ///< mapping input
    typename OutDOFs::SPtr outDofs; ///< mapping output
    simulation::Node::SPtr root;         ///< Root of the scene graph, created by the constructor an re-used in the tests
    std::pair<Real_t<In>, Real_t<In>> deltaRange; ///< The minimum and maximum magnitudes of the change of each scalar value of the small displacement is perturbation * numeric_limits<Real>::epsilon. This epsilon is 1.19209e-07 for float and 2.22045e-16 for double.
    Real_t<In> errorMax;     ///< The test is successful if the (infinite norm of the) difference is less than  errorMax * numeric_limits<Real>::epsilon
    Real_t<In> errorFactorDJ;     ///< The test for geometric stiffness is successful if the (infinite norm of the) difference is less than  errorFactorDJ * errorMax * numeric_limits<Real>::epsilon


    static constexpr unsigned char TEST_getJs = 1; ///< testing getJs used in assembly API
    static constexpr unsigned char TEST_getK = 2; ///< testing getK used in assembly API
    static constexpr unsigned char TEST_applyJT_matrix = 4; ///< testing applyJT on matrices
    static constexpr unsigned char TEST_applyDJT = 8; ///< testing applyDJT
    static constexpr unsigned char TEST_buildGeometricStiffnessMatrix = 16; ///< testing buildGeometricStiffnessMatrix
    static constexpr unsigned char TEST_ASSEMBLY_API = TEST_getJs | TEST_getK; ///< testing functions used in assembly API getJS getKS
    static constexpr unsigned char TEST_GEOMETRIC_STIFFNESS = TEST_applyDJT | TEST_getK | TEST_buildGeometricStiffnessMatrix; ///< testing functions used in assembly API getJS getKS
    unsigned char flags; ///< testing options. (all by default). To be used with precaution. Please implement the missing API in the mapping rather than not testing it.

    /// Returns true if the test of the method corresponding to the flag parameter must be executed, false otherwise
    bool isTestExecuted(const unsigned char testFlag) const
    {
        return flags & testFlag;
    }

    /// Defines if a test of the method corresponding to the flag parameter must be executed or not.
    void setTestExecution(const unsigned char testFlag, const bool doTheTest)
    {
        if (doTheTest)
        {
            flags |= testFlag;  // Set the flag
        }
        else
        {
            flags &= ~testFlag; // Unset the flag
        }
    }


    Mapping_test()
        : deltaRange(1,1000)
        , errorMax(10)
        , errorFactorDJ(1)
        , flags(TEST_ASSEMBLY_API | TEST_GEOMETRIC_STIFFNESS)
    {
        /// Parent node
        root = simpleapi::createRootNode(simulation::getSimulation(), "root");

        inDofs = core::objectmodel::New<InDOFs>();
        root->addObject(inDofs);

        /// Child node
        simulation::Node::SPtr childNode = root->createChild("childNode");
        outDofs = core::objectmodel::New<OutDOFs>();
        childNode->addObject(outDofs);
        auto mappingSptr = core::objectmodel::New<Mapping>();
        mapping = mappingSptr.get();
        childNode->addObject(mapping);

        mapping->setModels(inDofs.get(),outDofs.get());
    }

    Mapping_test(std::string fileName)
        : deltaRange(1, 1000),
          errorMax(100),
          errorFactorDJ(1),
          flags(TEST_ASSEMBLY_API|TEST_GEOMETRIC_STIFFNESS)
    {
        assert(simulation::getSimulation());

        /// Load the scene
        root = simpleapi::createRootNode(simulation::getSimulation(), "root");
        root = sofa::simulation::node::load(fileName.c_str(), false);

        // InDofs
        inDofs = root->get<InDOFs>(root->SearchDown);

        // Get child nodes
        const simulation::Node::SPtr patchNode = root->getChild("Patch");
        simulation::Node::SPtr elasticityNode = patchNode->getChild("Elasticity");

        // Add OutDofs
        outDofs = core::objectmodel::New<OutDOFs>();
        elasticityNode->addObject(outDofs);

        // Add mapping to the scene
        auto mappingSptr = core::objectmodel::New<Mapping>();
        mapping = mappingSptr.get();
        elasticityNode->addObject(mapping);
        mapping->setModels(inDofs.get(),outDofs.get());
        
    }

    /**
     * Test the mapping using the given values and small changes.
     * Return true in case of success, if all errors are below maxError*epsilon.
     * The mapping is initialized using the two first parameters,
     * then a new parent position is applied,
     * and the new child position is compared with the expected one.
     * Additionally, the Jacobian-related methods are tested using finite differences.
     *
     * The initialization values can used when the mapping is an embedding, e.g. to attach a mesh to a rigid object we compute the local coordinates of the vertices based on their world coordinates and the frame coordinates.
     * In other cases, such as mapping from pairs of points to distances, no initialization values are necessary, an one can use the same values as for testing, i.e. runTest( xp, expected_xc, xp, expected_xc).
     *
     *\param parentInit initial parent position
     *\param childInit initial child position
     *\param parentNew new parent position
     *\param expectedChildNew expected position of the child corresponding to the new parent position
     */
    virtual bool runTest( const VecCoord_t<In>& parentInit,
                          const VecCoord_t<Out>& childInit,
                          const VecCoord_t<In>& parentNew,
                          const VecCoord_t<Out>& expectedChildNew)
    {
        checkComparisonThreshold();
        warnMissingTests();

        const auto errorThreshold = this->epsilon() * errorMax;

        using EigenSparseMatrix = linearalgebra::EigenSparseMatrix<In, Out>;

        core::MechanicalParams mparams;
        mparams.setKFactor(1.0);
        mparams.setSupportOnlySymmetricMatrix(false);

        /// test apply: check if the child positions are the expected ones
        bool succeed = testMappingPositionVelocity(
            parentInit, childInit, parentNew, expectedChildNew, errorThreshold,
            mparams);

        const std::size_t sizeIn = inDofs->getSize();
        const std::size_t sizeOut = outDofs->getSize();

        // get position data
        VecCoord_t<Out> positionOut = outDofs->readPositions().ref();

        // set random child forces and propagate them to the parent
        VecDeriv_t<Out> forceOut = generateRandomVecDeriv<Out>(sizeOut, 0.1, 1.);

        VecDeriv_t<In> forceIn;
        computeForceInFromForceOut(mparams, forceIn, forceOut);

        // set small parent velocities and use them to update the child
        const VecDeriv_t<In> velocityIn = generateRandomVecDeriv<In>(sizeIn,
            this->epsilon() * deltaRange.first,
            this->epsilon() * deltaRange.second);

        const VecCoord_t<In> perturbedPositionIn = computePerturbedPositions(sizeIn, velocityIn);

        VecDeriv_t<Out> velocityOut;
        computeVelocityOutFromVelocityIn(mparams, velocityOut, velocityIn);

        // apply geometric stiffness
        inDofs->vRealloc( &mparams, core::vec_id::write_access::dx ); // dx is not allocated by default
        inDofs->writeDx().wref() = velocityIn;

        const VecDeriv_t<In> dfp_withoutUpdateK = applyDJT(mparams, false);
        const VecDeriv_t<In> dfp_withUpdateK = applyDJT(mparams, true);

        // Jacobian will be obsolete after applying new positions
        if( isTestExecuted(TEST_getJs) )
        {
            EigenSparseMatrix* J = this->getMatrix<EigenSparseMatrix>(mapping->getJs());

            // forceIn has been computed using applyJT
            // The following tests that forceIn is also the result of applying
            // the transposed jacobian matrix
            succeed &= checkJacobianMatrixTranspose(J, forceOut, forceIn, errorThreshold);

            // velocityOut has been computed using applyJ
            // The following tests that velocityOut is also the result of applying
            // the jacobian matrix
            succeed &= checkJacobianMatrix(J, velocityIn, velocityOut, errorThreshold);
        }

        if( isTestExecuted(TEST_applyJT_matrix) )
        {
            // TODO test applyJT on matrices
            // basic idea build a out random matrix  e.g. with helper::drand(100.0)
            // call applyJT on both this matrice and on all its lines (oe cols ?) one by one
            // then compare results

//            OutMatrixDeriv outMatrices(  ); // how to build that, what size?
//            /*WriteInMatrixDeriv min = */inDofs->write( vec_id::write_access::constraintJacobian );
//            WriteOutMatrixDeriv mout = outDofs->write( vec_id::write_access::constraintJacobian );
//            copyToData(mout,outMatrices);

//            mapping->applyJt(  ConstraintParams*, vec_id::write_access::constraintJacobian, vec_id::write_access::constraintJacobian );


        }

        // compute parent forces from pre-treated child forces (in most cases, the pre-treatment does nothing)
        // the pre-treatment can be useful to be able to compute 2 comparable results of applyJT with a small displacement to test applyDJT
        computeForceInFromForceOut(mparams, forceIn, preTreatment(forceOut));

        ///////////////////////////////

        // propagate small displacement
        inDofs->writePositions().wref() = perturbedPositionIn;
        succeed &= testApplyJonPosition(mparams, positionOut, velocityOut, errorThreshold);

        const VecDeriv_t<In> forceChange = computeForceChange(mparams, sizeIn, forceOut, forceIn);

        if( isTestExecuted(TEST_applyDJT) )
        {
            succeed &= checkApplyDJT(dfp_withoutUpdateK, forceChange, errorThreshold, false);
            succeed &= checkApplyDJT(dfp_withUpdateK, forceChange, errorThreshold, true);
        }

        if( isTestExecuted(TEST_getK))
        {
            succeed &= testGetK(sizeIn, velocityIn, forceChange, errorThreshold);
        }

        if( isTestExecuted(TEST_buildGeometricStiffnessMatrix) )
        {
            succeed &= testBuildGeometricStiffnessMatrix(sizeIn, velocityIn, forceChange, errorThreshold);
        }

        if(!succeed)
        {
            ADD_FAILURE() << "Failed Seed number = " << this->seed << std::endl;
        }
        return succeed;
    }


    /** Test the mapping using the given values and small changes.
     * Return true in case of success, if all errors are below maxError*epsilon.
     * The mapping is initialized using the first parameter,
     * @warning this version supposes the mapping initialization does not depend on child positions
     * otherwise, use the runTest functions with 4 parameters
     * the child position is computed from parent position and compared with the expected one.
     * Additionally, the Jacobian-related methods are tested using finite differences.
     *
     *\param parent parent position
     *\param expectedChild expected position of the child corresponding to the parent position
     */
    virtual bool runTest( const VecCoord_t<In>& parent,
                          const VecCoord_t<Out> expectedChild)
    {
        VecCoord_t<Out> childInit( expectedChild.size() ); // start with null child
        return runTest( parent, childInit, parent, expectedChild );
    }

    ~Mapping_test() override
    {
        if (root != nullptr)
        {
            sofa::simulation::node::unload(root);
        }
    }

protected:



    /** Returns OutCoord substraction a-b */
    virtual Deriv_t<Out> difference( const Coord_t<Out>& a, const Coord_t<Out>& b )
    {
        return Out::coordDifference(a,b);
    }

    virtual VecDeriv_t<Out> difference( const VecDeriv_t<Out>& a, const VecDeriv_t<Out>& b )
    {
        if (a.size() != b.size())
        {
            ADD_FAILURE() << "VecDeriv_t<Out> have different sizes";
            return {};
        }

        VecDeriv_t<Out> c;
        c.reserve(a.size());
        std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(c), std::minus());

        return c;
    }

    /** Possible child force pre-treatment, does nothing by default
      */
    virtual VecDeriv_t<Out> preTreatment( const VecDeriv_t<Out>& f )
    {
        return f;
    }


    void checkComparisonThreshold()
    {
        if( deltaRange.second / errorMax <= sofa::testing::g_minDeltaErrorRatio )
        {
            ADD_FAILURE() << "The comparison threshold is too large for the finite difference delta";
        }
    }

    void warnMissingTests() const
    {
        msg_warning_when(!isTestExecuted(TEST_getJs), "MappingTest") << "getJs is not tested";
        msg_warning_when(!isTestExecuted(TEST_getK)           , "MappingTest") << "getK is not tested";
        msg_warning_when(!isTestExecuted(TEST_applyJT_matrix) , "MappingTest") << "applyJT on matrices is not tested";
        msg_warning_when(!isTestExecuted(TEST_applyDJT)       , "MappingTest") << "applyDJT is not tested";
        msg_warning_when(!isTestExecuted(TEST_buildGeometricStiffnessMatrix) , "MappingTest") << "buildGeometricStiffnessMatrix is not tested";
    }

    bool testMappingPositionVelocity(const VecCoord_t<In>& parentInit,
                                     const VecCoord_t<Out>& childInit,
                                     const VecCoord_t<In>& parentNew,
                                     const VecCoord_t<Out>& expectedChildNew,
                                     const Real_t<In> errorThreshold,
                                     core::MechanicalParams mparams)
    {
        helper::WriteAccessor positionAccessorIn = inDofs->writePositions();
        positionAccessorIn.wref() = parentInit;

        helper::WriteAccessor positionAccessorOut = outDofs->writePositions();
        positionAccessorOut.wref() = childInit;

        /// Init based on parentInit
        sofa::simulation::node::initRoot(root.get());

        /// Updated to parentNew
        positionAccessorIn.wref() = parentNew;
        mapping->apply(&mparams, core::vec_id::write_access::position, core::vec_id::write_access::position);
        mapping->applyJ(&mparams, core::vec_id::write_access::velocity, core::vec_id::write_access::velocity);

        bool succeed = true;

        if (expectedChildNew.size() != positionAccessorOut.size())
        {
            ADD_FAILURE() << "Size of output dofs is wrong: " << positionAccessorOut.size() << " expected: " << expectedChildNew.size();
            succeed = false;
        }

        for (unsigned i = 0; i < positionAccessorOut.size(); ++i)
        {
            if (!this->isSmall(difference(positionAccessorOut[i], expectedChildNew[i]).norm(), errorMax))
            {
                ADD_FAILURE() << "Position of mapped particle " << i << " is wrong: \n" << positionAccessorOut[i] <<"\nexpected: \n" << expectedChildNew[i]
                        <<  "\ndifference should be less than " << errorThreshold << " (" << difference(positionAccessorOut[i],expectedChildNew[i]).norm() << ")" << std::endl;
                succeed = false;
            }
        }

        return succeed;
    }

    template<class DataTypes>
    VecDeriv_t<DataTypes> generateRandomVecDeriv(const std::size_t size, const Real_t<DataTypes> minMagnitude, const Real_t<DataTypes> maxMagnitude)
    {
        VecDeriv_t<DataTypes> randomForce;
        randomForce.reserve(size);
        for (std::size_t i = 0; i < size; i++)
        {
            randomForce.push_back(DataTypes::randomDeriv(minMagnitude, maxMagnitude));
        }
        return randomForce;
    }

    void computeForceInFromForceOut(core::MechanicalParams mparams, VecDeriv_t<In>& forceIn, const VecDeriv_t<Out>& forceOut)
    {
        inDofs->writeForces()->fill(Deriv_t<In>());  // reset parent forces before accumulating child forces

        outDofs->writeForces().wref() = forceOut;
        mapping->applyJT( &mparams, core::vec_id::write_access::force, core::vec_id::write_access::force );
        forceIn = inDofs->readForces().ref();
    }

    void computeVelocityOutFromVelocityIn(core::MechanicalParams mparams, VecDeriv_t<Out>& velocityOut, const VecDeriv_t<In>& velocityIn)
    {
        inDofs->writeVelocities().wref() = velocityIn;
        mapping->applyJ( &mparams, core::vec_id::write_access::velocity, core::vec_id::write_access::velocity );
        velocityOut = outDofs->readVelocities().ref();
    }

    const VecDeriv_t<In>& applyDJT(core::MechanicalParams mparams, bool updateK)
    {
        inDofs->writeForces()->fill(Deriv_t<In>()); //reset force

        if (updateK)
        {
            mapping->updateK( &mparams, core::vec_id::read_access::force ); // updating stiffness matrix for the current state and force
        }
        mapping->applyDJT( &mparams, core::vec_id::write_access::force, core::vec_id::write_access::force );
        return inDofs->readForces().ref();
    }

    [[nodiscard]] bool checkApplyDJT(const VecDeriv_t<In>& dfp, const VecDeriv_t<In>& fp12, Real_t<In> errorThreshold, bool updateK)
    {
        if (this->vectorMaxDiff(dfp, fp12) > errorThreshold * errorFactorDJ)
        {
            const std::string updateKString = updateK ? "after call to updateK" : "no call to updateK";
            ADD_FAILURE() << "applyDJT (" << updateKString << ") test failed" << std::endl
                << "dfp    = " << dfp << std::endl
                << "fp2-fp = " << fp12 << std::endl
                << "error threshold = " << errorThreshold * errorFactorDJ << std::endl;
            return false;
        }
        return true;
    }

    VecCoord_t<In> computePerturbedPositions(const std::size_t sizeIn, const VecDeriv_t<In> velocityIn)
    {
        VecCoord_t<In> perturbedPositionIn;
        perturbedPositionIn.reserve(sizeIn);
        const helper::ReadAccessor positionIn = inDofs->readPositions();
        for (std::size_t i = 0; i < sizeIn; ++i)
        {
            perturbedPositionIn.push_back(positionIn[i] + velocityIn[i]);
        }
        return perturbedPositionIn;
    }


    /// Get one EigenSparseMatrix out of a list. Error if not one single matrix in the list.
    template<class EigenSparseMatrixType>
    static EigenSparseMatrixType* getMatrix(const type::vector<sofa::linearalgebra::BaseMatrix*>* matrices)
    {
        if( !matrices )
        {
            ADD_FAILURE()<< "Matrix list is nullptr (API for assembly is not implemented)";
        }

        if( matrices->empty() )
        {
            ADD_FAILURE()<< "Matrix list is empty";
            return nullptr;
        }

        if( matrices->size() != 1 )
        {
            ADD_FAILURE()<< "Matrix list should have size == 1 in simple mappings (current size = " << matrices->size() << ")";
        }
        EigenSparseMatrixType* ei = dynamic_cast<EigenSparseMatrixType*>((*matrices)[0] );
        if( ei == nullptr )
        {
            ADD_FAILURE() << "getJs returns a matrix of non-EigenSparseMatrix type";
            // TODO perform a slow conversion with a big warning rather than a failure?
        }
        return ei;
    }

    bool checkJacobianMatrixTranspose(
        EigenSparseMatrix* jacobianMatrix,
        const VecDeriv_t<Out>& forceOut,
        const VecDeriv_t<In>& expectedForceIn,
        Real_t<In> errorThreshold)
    {
        VecDeriv_t<In> computedForceIn(expectedForceIn.size(), Deriv_t<In>());

        //computedForceIn += J^T * forceOut
        jacobianMatrix->addMultTranspose(computedForceIn, forceOut);

        const auto diff = this->vectorMaxDiff(computedForceIn, expectedForceIn);
        if (diff > errorThreshold)
        {
            ADD_FAILURE() <<
                "getJs is not consistent with applyJT, difference should be "
                "less than " << errorThreshold << " (" << diff << ")" << std::endl
                << "computedForceIn = " << computedForceIn << std::endl
                << "expectedForceIn = " << expectedForceIn << std::endl;
            return false;
        }
        return true;
    }

    bool checkJacobianMatrix(
        EigenSparseMatrix* jacobianMatrix,
        const VecDeriv_t<In>& velocityIn,
        const VecDeriv_t<Out>& expectedVelocityOut,
        Real_t<In> errorThreshold)
    {
        VecDeriv_t<Out> computedVelocityOut(expectedVelocityOut.size());

        jacobianMatrix->mult(computedVelocityOut, velocityIn);

        const auto diff = this->vectorMaxDiff(computedVelocityOut, expectedVelocityOut);
        if (diff > errorThreshold)
        {
            ADD_FAILURE() <<
                "getJs is not consistent with applyJ, difference should be "
                "less than " << errorThreshold << " (" << diff << ")" << std::endl
                << "velocityIn = " << velocityIn << std::endl
                << "computedVelocityOut = " << computedVelocityOut << std::endl
                << "expectedVelocityOut = " << expectedVelocityOut << std::endl;
            return false;
        }

        return true;
    }

    bool testApplyJonPosition(
        core::MechanicalParams mparams,
        const VecCoord_t<Out>& positionOut,
        const VecDeriv_t<Out>& expectedVelocityOut,
        Real_t<In> errorThreshold
        )
    {
        mapping->apply ( &mparams, core::vec_id::write_access::position, core::vec_id::write_access::position );
        const VecCoord_t<Out>& positionOut1 = outDofs->readPositions();

        const auto sizeOut = positionOut.size();
        VecDeriv_t<Out> dxOut(sizeOut);
        for (unsigned i = 0; i < sizeOut; i++)
        {
            dxOut[i] = difference(positionOut1[i], positionOut[i]);
        }

        const auto diff = this->vectorMaxAbs(difference(dxOut, expectedVelocityOut));
        if (diff > errorThreshold)
        {
            ADD_FAILURE() << "applyJ test failed: the difference between child "
                "position change and child velocity (dt=1) " << diff <<
                " should be less than  " << errorThreshold << std::endl
                << "position change = " << dxOut << std::endl
                << "expectedVelocityOut = " << expectedVelocityOut << std::endl;
            return false;
        }

        return true;
    }

    VecDeriv_t<In> computeForceChange(core::MechanicalParams mparams, const std::size_t sizeIn, VecDeriv_t<Out> forceOut, VecDeriv_t<In> forceIn)
    {
        VecDeriv_t<In> forceChange;
        forceChange.reserve(sizeIn);

        // apply has been called, therefore parent force must be updated
        // based on the same child forces
        VecDeriv_t<In> forceIn2;
        computeForceInFromForceOut(mparams, forceIn2, preTreatment(forceOut));

        for (unsigned i = 0; i < sizeIn; ++i)
        {
            forceChange.push_back(forceIn2[i] - forceIn[i]);
        }

        return forceChange;
    }

    bool testGetK(const std::size_t& sizeIn,
        const VecDeriv_t<In>& velocityIn,
        const VecDeriv_t<In>& forceChange,
        Real_t<In> errorThreshold)
    {
        VecDeriv_t<In> Kv(sizeIn);

        const linearalgebra::BaseMatrix* bk = mapping->getK();

        // K can be null or empty for linear mappings
        // still performing the test with a null Kv vector to check if the mapping is really linear

        if( bk != nullptr )
        {
            typedef linearalgebra::EigenSparseMatrix<In,In> EigenSparseKMatrix;
            const EigenSparseKMatrix* K = dynamic_cast<const EigenSparseKMatrix*>(bk);
            if( K == nullptr )
            {
                ADD_FAILURE() << "getK returns a matrix of non-EigenSparseMatrix type";
                // TODO perform a slow conversion with a big warning rather than a failure?
                return false;
            }

            if( K->compressedMatrix.nonZeros() )
            {
                K->mult(Kv,velocityIn);
            }
        }

        // check that K.vp = dfp
        if (this->vectorMaxDiff(Kv, forceChange) > errorThreshold * errorFactorDJ)
        {
            ADD_FAILURE() << "K test failed, difference should be less than " << errorThreshold*errorFactorDJ  << std::endl
                          << "Kv    = " << Kv << std::endl
                          << "dfp = " << forceChange << std::endl;
            return false;
        }

        return true;
    }

    bool testBuildGeometricStiffnessMatrix(
        std::size_t sizeIn,
        const VecDeriv_t<In>& velocityIn,
        const VecDeriv_t<In>& forceChange,
        Real_t<In> errorThreshold
        )
    {
        core::GeometricStiffnessMatrix testGeometricStiffness;

        struct GeometricStiffnessAccumulator : core::MappingMatrixAccumulator
        {
            void add(sofa::SignedIndex row, sofa::SignedIndex col, float value) override
            {
                assembledMatrix.add(row, col, value);
            }
            void add(sofa::SignedIndex row, sofa::SignedIndex col, double value) override
            {
                assembledMatrix.add(row, col, value);
            }

            linearalgebra::EigenSparseMatrix<In,In> assembledMatrix;
        } accumulator;

        testGeometricStiffness.setMatrixAccumulator(&accumulator, mapping->getFromModel(), mapping->getFromModel());

        accumulator.assembledMatrix.resize(mapping->getFromModel()->getSize() * In::deriv_total_size, mapping->getFromModel()->getSize() * In::deriv_total_size);
        mapping->buildGeometricStiffnessMatrix(&testGeometricStiffness);
        accumulator.assembledMatrix.compress();

        VecDeriv_t<In> Kv(sizeIn);
        if( accumulator.assembledMatrix.compressedMatrix.nonZeros() )
        {
            accumulator.assembledMatrix.mult(Kv,velocityIn);
        }

        // check that K.vp = dfp
        if (this->vectorMaxDiff(Kv,forceChange) > errorThreshold*errorFactorDJ )
        {
            ADD_FAILURE() << "buildGeometricStiffnessMatrix test failed, difference should be less than " << errorThreshold*errorFactorDJ  << std::endl
                          << "Kv    = " << Kv << std::endl
                          << "dfp = " << forceChange << std::endl;
            return false;
        }

        return true;
    }

};

} // namespace sofa::mapping_test
