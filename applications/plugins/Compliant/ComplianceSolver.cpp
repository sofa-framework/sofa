#include "ComplianceSolver.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>
#include <sofa/component/linearsolver/EigenSparseSquareMatrix.h>
#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/SingleMatrixAccessor.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;

SOFA_DECL_CLASS(ComplianceSolver);
int ComplianceSolverClass = core::RegisterObject("A simple explicit time integrator").add< ComplianceSolver >();



ComplianceSolver::ComplianceSolver()
    : implicitVelocity( initData(&implicitVelocity,(SReal)0.5,"implicitVelocity","Weight of the next forces in the average forces used to update the velocities. 1 is implicit, 0 is explicit."))
    , implicitPosition( initData(&implicitPosition,(SReal)0.5,"implicitPosition","Weight of the next velocities in the average velocities used to update the positions. 1 is implicit, 0 is explicit."))
    , verbose( initData(&verbose,false,"verbose","Print a lot of info for debug"))
{
}

//void ComplianceSolver::bwdInit()
//{
//    core::ExecParams params;
//}

void ComplianceSolver::solveEquation()
{
    inverseDiagonalMatrix( invM, matM, 1.0e-6 ); //cerr<<"ComplianceSolver::solve, Minv has " << Minv.nonZeros() << "non-null entries " << endl;
    SMatrix Minv = matP * invM * matP;
    SMatrix schur( matJ * Minv * matJ.transpose() + matC );
    SparseLDLT schurDcmp(schur);
    VectorEigen x = vecPhi.getVectorEigen() - matJ * ( Minv * vecF.getVectorEigen() );
    if( verbose.getValue() )
    {
        //        cerr<<"ComplianceSolver::solve, Minv = " << endl << Eigen::MatrixXd(Minv) << endl;
        cerr<<"ComplianceSolver::solve, schur complement = " << endl << Eigen::MatrixXd(schur) << endl;
        cerr<<"ComplianceSolver::solve,  Minv * vecF.getVectorEigen() = " << (Minv * vecF.getVectorEigen()).transpose() << endl;
        cerr<<"ComplianceSolver::solve, matJ * ( Minv * vecF.getVectorEigen())  = " << ( matJ * ( Minv * vecF.getVectorEigen())).transpose() << endl;
        cerr<<"ComplianceSolver::solve,  vecPhi.getVectorEigen()  = " <<  vecPhi.getVectorEigen().transpose() << endl;
        cerr<<"ComplianceSolver::solve, right-hand term = " << x.transpose() << endl;
    }
    schurDcmp.solveInPlace( x ); // solve (J.M^{-1}.J^T + C).x = c - J.M^{-1}.f
    vecF.getVectorEigen() = vecF.getVectorEigen() + matJ.transpose() * x ; // f = f_ext + J^T.lambda
    if( verbose.getValue() )
    {
        cerr<<"ComplianceSolver::solve, constraint forces = " << x.transpose() << endl;
        cerr<<"ComplianceSolver::solve, net forces = " << vecF << endl;
        //        cerr<<"ComplianceSolver::solve, net forces = " << f << endl;
    }
}

void ComplianceSolver::solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    // tune parameters
    core::MechanicalParams cparams(*params);
    cparams.setMFactor(1.0);
    cparams.setDt(dt);
    cparams.setImplicitVelocity( implicitVelocity.getValue() );
    cparams.setImplicitPosition( implicitPosition.getValue() );

    //  State vectors and operations
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecDeriv dv(&vop, core::VecDerivId::dx() );
    MultiVecDeriv f  (&vop, core::VecDerivId::force() );
    MultiVecCoord nextPos(&vop, xResult );
    MultiVecDeriv nextVel(&vop, vResult );



    // Compute right-hand term
    mop.computeForce(f);
    mop.projectResponse(f);
    if( verbose.getValue() )
    {
        cerr<<"ComplianceSolver::solve, filtered external forces = " << f << endl;
    }

    // Matrix size
    MatrixAssemblyVisitor assembly(&cparams,this);
    this->getContext()->executeVisitor(&assembly(COMPUTE_SIZE));
    //    cerr<<"ComplianceSolver::solve, sizeM = " << assembly.sizeM <<", sizeC = "<< assembly.sizeC << endl;

    if( assembly.sizeC > 0 )
    {


        matM.resize(assembly.sizeM,assembly.sizeM);
        matP.resize(assembly.sizeM,assembly.sizeM);
        matC.resize(assembly.sizeC,assembly.sizeC);
        matJ.resize(assembly.sizeC,assembly.sizeM);
        vecF.resize(assembly.sizeM);
        vecPhi.resize(assembly.sizeC);
        vecF.clear();
        vecPhi.clear();

        // Matrix assembly
        this->getContext()->executeVisitor(&assembly(DO_SYSTEM_ASSEMBLY));

        if( verbose.getValue() )
        {
            cerr<<"ComplianceSolver::solve, final M = " << endl << matM << endl;
            cerr<<"ComplianceSolver::solve, final P = " << endl << matP << endl;
            cerr<<"ComplianceSolver::solve, final C = " << endl << matC << endl;
            cerr<<"ComplianceSolver::solve, final J = " << endl << matJ << endl;
            cerr<<"ComplianceSolver::solve, final f = " << vecF << endl;
            cerr<<"ComplianceSolver::solve, final phi = " << vecPhi << endl;
        }

        // Solve equation system
        solveEquation();

        this->getContext()->executeVisitor(&assembly(DISTRIBUTE_SOLUTION));  // set dv in each MechanicalState
    }

    mop.accFromF(dv, f);
    mop.projectResponse(dv);


    // Apply integration scheme

    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp ops;
    ops.resize(2);
    ops[0].first = nextPos; // p = p + v*h + dv*h*beta
    ops[0].second.push_back(std::make_pair(pos.id(),1.0));
    ops[0].second.push_back(std::make_pair(vel.id(),dt));
    ops[0].second.push_back(std::make_pair(  dv.id(),dt*implicitPosition.getValue()));
    ops[1].first = nextVel; // v = v + dv
    ops[1].second.push_back(std::make_pair(vel.id(),1.0));
    ops[1].second.push_back(std::make_pair(  dv.id(),1.0));
    vop.v_multiop(ops);

    if( verbose.getValue() )
    {
        mop.propagateX(nextPos);
        mop.propagateDx(nextVel);
        serr<<"EulerImplicitSolver, final x = "<< nextPos <<sendl;
        serr<<"EulerImplicitSolver, final v = "<< nextVel <<sendl;
    }

}



simulation::Visitor::Result ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown(simulation::Node* node)
{
    if( pass== COMPUTE_SIZE )
    {
        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            //        cerr<<"node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
            m_offset[node->mechanicalState] = sizeM;
            sizeM += node->mechanicalState->getMatrixSize();
        }


        // ==== process compliances
        if( node->forceField.size()>0  && node->forceField[0]->getComplianceMatrix(mparams) )
        {
            assert(node->mechanicalState);
            c_offset[node->forceField[0]] = sizeC;
            sizeC += node->mechanicalState->getMatrixSize();
        }
        assert(node->forceField.size()<2);

    }
    else if (pass== DO_SYSTEM_ASSEMBLY)
    {
        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            //            cerr<<"pass "<< pass << ", node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
            ComplianceSolver::SMatrix shiftMatrix = createShiftMatrix( node->mechanicalState->getMatrixSize(), sizeM, m_offset[node->mechanicalState] );
            jMap[node->mechanicalState]= shiftMatrix ;

            // projections applied to the independent DOFs. The projection applied to mapped DOFs are ignored.
            if(node->projectiveConstraintSet.size()>0 )
            {
                SMatrix projMat = createIdentityMatrix(node->mechanicalState->getMatrixSize());
                for(unsigned i=0; i<node->projectiveConstraintSet.size(); i++)
                {
                    if( const defaulttype::BaseMatrix* mat = node->projectiveConstraintSet[i]->getJ(mparams) )
                    {
                        //                    cerr<<"MatrixAssemblyVisitor::processNodeTopDown, current projMat = " << projMat << endl;
                        projMat = projMat * getSMatrix(mat);  // multiply with the projection matrix
                        //                    cerr<<"MatrixAssemblyVisitor::processNodeTopDown, constraint matrix = " << *mat << endl;
                        //                    cerr<<"MatrixAssemblyVisitor::processNodeTopDown, constraint matrix = " << toMatrix(mat) << endl;
                        //                    cerr<<"MatrixAssemblyVisitor::processNodeTopDown, new projMat = " << projMat << endl;
                    }
                }
                solver->matP += shiftMatrix.transpose() * projMat * shiftMatrix;
            }

            // Right-hand term
            unsigned offset = m_offset[node->mechanicalState]; // use a copy, because the parameter is modified by addToBaseVector
            //            cerr<<"MatrixAssemblyVisitor::processNodeTopDown, mechanical state "<< node->mechanicalState->getName() <<",  force before = " << solver->vecF << endl;
            node->mechanicalState->addToBaseVector(&solver->vecF,core::VecDerivId::force(),offset);
            //            cerr<<"MatrixAssemblyVisitor::processNodeTopDown, mechanical state "<< node->mechanicalState->getName() <<", cumulated force = " << solver->vecF << endl;



        }

        // ==== mechanical mapping
        if ( node->mechanicalMapping != NULL )
        {
            const vector<sofa::defaulttype::BaseMatrix*>* pJs = node->mechanicalMapping->getJs();
            vector<core::BaseState*> pStates = node->mechanicalMapping->getFrom();
            assert( pJs->size() == pStates.size());
            MechanicalState* mtarget = dynamic_cast<MechanicalState*>(  node->mechanicalMapping->getTo()[0] ); // Only N-to-1 mappings are handled yet
            //            cerr<<"MatrixAssemblyVisitor::processNodeTopDown, mapping  " << node->mechanicalMapping->getName()<< endl;
            for( unsigned i=0; i<pStates.size(); i++ )
            {
                MechanicalState* mstate = dynamic_cast<MechanicalState*>(pStates[i]);
                assert(mstate);
                SMatrix J = getSMatrix( (*pJs)[i] );
                SMatrix contribution;
                //                cerr<<"MatrixAssemblyVisitor::processNodeTopDown, J = "<< endl << J << endl;
                //                cerr<<"MatrixAssemblyVisitor::processNodeTopDown, jMap[ mstate ] = "<< endl << jMap[ mstate ] << endl;
                contribution = J * jMap[ mstate ];
                if( jMap[mtarget].rows()!=contribution.rows() || jMap[mtarget].cols()!=contribution.cols() )
                    jMap[mtarget].resize(contribution.rows(),contribution.cols());
                jMap[mtarget] += contribution;
            }
        }

        // ==== mass
        if (node->mass != NULL  )
        {
            //            cerr<<"pass "<< pass << ", node " << node->getName() << ", mass: " << node->mass->getName() << endl;
            assert( node->mechanicalState != NULL );
            linearsolver::EigenSparseSquareMatrix<SReal> sqmat(node->mechanicalState->getMatrixSize(),node->mechanicalState->getMatrixSize());
            linearsolver::SingleMatrixAccessor accessor( &sqmat );
            node->mass->addMToMatrix( mparams, &accessor );
            //                    cerr<<"eigen matrix of the mass: " << sqmat  << endl;
            SMatrix JtMJtop;
            JtMJtop = jMap[node->mechanicalState].transpose() * sqmat.eigenMatrix * jMap[node->mechanicalState];
            //                    cerr<<"contribution to the mass matrix: " << endl << JtMJtop << endl;
            solver->matM += JtMJtop;  // add J^T M J to the assembled mass matrix
        }

        // ==== compliance
        if( node->forceField.size()>0 && node->forceField[0]->getComplianceMatrix(mparams) )
        {
            SMatrix compOffset = createShiftMatrix( node->mechanicalState->getMatrixSize(), sizeC, c_offset[node->forceField[0]] );

            SMatrix J = SMatrix( compOffset.transpose() * jMap[node->mechanicalState] ); // shift J
            solver->matJ += J;                                          // assemble

            SReal alpha = cparams.implicitVelocity(); // implicit velocity factor in the integration scheme
            SReal beta  = cparams.implicitPosition(); // implicit position factor in the integration scheme
            SReal l = alpha * (beta * mparams->dt() + node->forceField[0]->getDampingRatio() );
            if( fabs(l)<1.0e-10 ) solver->serr << node->forceField[0]->getName() << ", l is not invertible in ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown" << solver->sendl;
            SReal invl = 1.0/l;
            solver->matC += SMatrix( compOffset.transpose() * getSMatrix(node->forceField[0]->getComplianceMatrix(mparams)) * compOffset ) * invl;  // assemble

            // Right-hand term
            node->forceField[0]->writeConstraintValue( &cparams, core::VecDerivId::force() );
            unsigned offset = c_offset[node->forceField[0]]; // use a copy, because the parameter is modified by addToBaseVector
            node->mechanicalState->addToBaseVector(&solver->vecPhi, core::VecDerivId::force(), offset );

        }

    }
    else if (pass== DISTRIBUTE_SOLUTION)
    {
        typedef defaulttype::BaseVector  SofaVector;

        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            //            cerr<<"pass "<< pass << ", node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
            unsigned offset = m_offset[node->mechanicalState]; // use a copy, because the parameter is modified by addToBaseVector
            node->mechanicalState->copyFromBaseVector(core::VecDerivId::force(), &solver->vecF, offset );
        }

    }
    else
    {
        cerr<<"ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown, unknown pass " << pass << endl;
    }

    return RESULT_CONTINUE;

}

void ComplianceSolver::MatrixAssemblyVisitor::processNodeBottomUp(simulation::Node* /*node*/)
{
    if( pass==COMPUTE_SIZE )
    {

    }
    else if (pass==DO_SYSTEM_ASSEMBLY)
    {
        // ==== independent DOFs
        //        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        //        {
        //            jStack.pop();
        //        }

        // ==== mechanical mapping
        //        if ( node->mechanicalMapping != NULL )
        //        {
        //            jStack.pop();
        //        }
    }
    else if (pass==DISTRIBUTE_SOLUTION )
    {
    }
    else
    {
        cerr<<"ComplianceSolver::MatrixAssemblyVisitor::processNodeBottomUp, unknown pass " << pass << endl;
    }

}

/// Return a rectangular matrix (cols>rows), with (offset-1) null columns, then the (rows*rows) identity, then null columns.
/// This is used to shift a "local" matrix to the global indices of an assembly matrix.
ComplianceSolver::SMatrix ComplianceSolver::MatrixAssemblyVisitor::createShiftMatrix( unsigned rows, unsigned cols, unsigned offset )
{
    SMatrix m(rows,cols);
    for(unsigned i=0; i<rows; i++ )
    {
        m.startVec(i);
        m.insertBack( i, offset+i) =1;
        //        m.coeffRef(i,offset+i)=1;
    }
    m.finalize();
    return m;
}

ComplianceSolver::SMatrix ComplianceSolver::MatrixAssemblyVisitor::createIdentityMatrix( unsigned size )
{
    SMatrix m(size,size);
    for(unsigned i=0; i<size; i++ )
    {
        m.startVec(i);
        m.insertBack(i,i) =1;
    }
    m.finalize();
    return m;
}


const ComplianceSolver::SMatrix& ComplianceSolver::MatrixAssemblyVisitor::getSMatrix( const defaulttype::BaseMatrix* m)
{
    const linearsolver::EigenBaseSparseMatrix<SReal>* sm = dynamic_cast<const linearsolver::EigenBaseSparseMatrix<SReal>*>(m);
    assert(sm);
    return sm->eigenMatrix;
}


/// Converts a BaseMatrix to the matrix type used here.
void ComplianceSolver::inverseDiagonalMatrix( SMatrix& Minv, const SMatrix& M, SReal /*threshold*/)
{
    assert(M.rows()==M.cols());
    Minv.resize(M.rows(),M.rows());
    for (int i=0; i<M.outerSize(); ++i)
        for (SMatrix::InnerIterator it(M,i); it; ++it)
        {
            Minv.startVec(i);
            Minv.insertBack(i,i) = 1.0/it.value();
            assert(it.row()==it.col()); // only diagonal matrices
        }
    Minv.finalize();


    //    assert(M.rows()==M.cols());
    //    SparseLDLT Mdcmp(M);
    //    Minv.resize(M.rows(),M.rows());
    //    for( int i=0; i<M.rows(); i++ ){
    //        VectorEigen v(M.rows());
    //        v.fill(0);
    //        v(i)=1;
    //        Mdcmp.solveInPlace(v);
    //        Minv.startVec(i);
    //        for(int j=0; j<M.rows(); j++){
    //            if( fabs(v(j))>=threshold )
    //                Minv.insertBack(j,i) = v(j);
    //        }
    //    }
    //    Minv.finalize();

}



}
}
}
