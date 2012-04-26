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
    :implicitVelocity( initData(&implicitVelocity,(SReal)0.5,"implicitVelocity","Weight of the next forces in the average forces used to update the velocities. 1 is implicit, 0 is explicit."))
    , implicitPosition( initData(&implicitPosition,(SReal)0.5,"implicitPosition","Weight of the next velocities in the average velocities used to update the positions. 1 is implicit, 0 is explicit."))
    , verbose( initData(&verbose,false,"verbose","Print a lot of info for debug"))
//    , _PMinvP( invprojMatrix.eigenMatrix )
{
}


void ComplianceSolver::solveEquation()
{
    SMatrix schur( _matJ * PMinvP() * _matJ.transpose() + _matC );
    SparseLDLT schurDcmp(schur);
    VectorEigen& lambda = _vecLambda.getVectorEigen();
    lambda = _vecPhi.getVectorEigen() - _matJ * ( PMinvP() * _vecF.getVectorEigen() );
    if( verbose.getValue() )
    {
        //        cerr<<"ComplianceSolver::solve, Minv = " << endl << Eigen::MatrixXd(Minv) << endl;
        cerr<<"ComplianceSolver::solve, schur complement = " << endl << Eigen::MatrixXd(schur) << endl;
        cerr<<"ComplianceSolver::solve,  Minv * vecF.getVectorEigen() = " << (PMinvP() * _vecF.getVectorEigen()).transpose() << endl;
        cerr<<"ComplianceSolver::solve, matJ * ( Minv * vecF.getVectorEigen())  = " << ( _matJ * ( PMinvP() * _vecF.getVectorEigen())).transpose() << endl;
        cerr<<"ComplianceSolver::solve,  vecPhi.getVectorEigen()  = " <<  _vecPhi.getVectorEigen().transpose() << endl;
        cerr<<"ComplianceSolver::solve, right-hand term = " << lambda.transpose() << endl;
    }
    schurDcmp.solveInPlace( lambda ); // solve (J.M^{-1}.J^T + C).x = c - J.M^{-1}.f
    _vecF.getVectorEigen() = _vecF.getVectorEigen() + _matJ.transpose() * lambda ; // f = f_ext + J^T.lambda
    _vecDv.getVectorEigen() = PMinvP() * _vecF.getVectorEigen();
    if( verbose.getValue() )
    {
        cerr<<"ComplianceSolver::solve, constraint forces = " << lambda.transpose() << endl;
        cerr<<"ComplianceSolver::solve, net forces = " << _vecF << endl;
        cerr<<"ComplianceSolver::solve, vecDv = " << _vecDv << endl;
    }


}

void ComplianceSolver::solve(const core::ExecParams* params, double h, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    // tune parameters
    core::MechanicalParams cparams(*params);
    cparams.setMFactor(1.0);
    cparams.setDt(h);
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

    // Compute system matrices and vectors
    _PMinvP_isDirty = true;
    MatrixAssemblyVisitor assembly(&cparams,this);
    this->getContext()->executeVisitor(&assembly(COMPUTE_SIZE)); // first the size
    //    cerr<<"ComplianceSolver::solve, sizeM = " << assembly.sizeM <<", sizeC = "<< assembly.sizeC << endl;

//    if( assembly.sizeC > 0 ){


    _matM.resize(assembly.sizeM,assembly.sizeM);
    _projMatrix.eigenMatrix = createIdentityMatrix(assembly.sizeM);
    _matC.resize(assembly.sizeC,assembly.sizeC);
    _matJ.resize(assembly.sizeC,assembly.sizeM);
    _vecF.resize(assembly.sizeM);
    _vecF.clear();
    _vecDv.resize(assembly.sizeM);
    _vecDv.clear();
    _vecPhi.resize(assembly.sizeC);
    _vecPhi.clear();
    _vecLambda.resize(assembly.sizeC);
    _vecLambda.clear();

    // Matrix assembly
    this->getContext()->executeVisitor(&assembly(DO_SYSTEM_ASSEMBLY));

    if( verbose.getValue() )
    {
        cerr<<"ComplianceSolver::solve, final M = " << endl << _matM << endl;
        cerr<<"ComplianceSolver::solve, final P = " << endl << P() << endl;
        cerr<<"ComplianceSolver::solve, final C = " << endl << _matC << endl;
        cerr<<"ComplianceSolver::solve, final J = " << endl << _matJ << endl;
        cerr<<"ComplianceSolver::solve, final f = " << _vecF << endl;
        cerr<<"ComplianceSolver::solve, final phi = " << _vecPhi << endl;
    }

    _matM *= 1.0/h;

    // Solve equation system
    solveEquation();

    this->getContext()->executeVisitor(&assembly(DISTRIBUTE_SOLUTION));  // set dv in each MechanicalState
//    }



    // Apply integration scheme

    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp vmOp;
    vmOp.resize(2);
    vmOp[0].first = nextPos; // p = p + v*h + dv*h*beta
    vmOp[0].second.push_back(std::make_pair( pos.id(), 1.0));
    vmOp[0].second.push_back(std::make_pair( vel.id(), h  ));
    vmOp[0].second.push_back(std::make_pair(  dv.id(), h*implicitPosition.getValue()));
    vmOp[1].first = nextVel; // v = v + ha
    vmOp[1].second.push_back(std::make_pair(vel.id(),1.0));
    vmOp[1].second.push_back(std::make_pair(  dv.id(),1.));
    vop.v_multiop(vmOp);

    if( verbose.getValue() )
    {
        mop.propagateX(nextPos);
        mop.propagateDx(nextVel);
        serr<<"ComplianceSolver, final x = "<< nextPos <<sendl;
        serr<<"ComplianceSolver, final v = "<< nextVel <<sendl;
        serr<<"-----------------------------"<<sendl;
    }

}



simulation::Visitor::Result ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown(simulation::Node* node)
{
    //    cerr<<"ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown visit Node "<<node->getName()<<endl;
    if( pass== COMPUTE_SIZE )
    {
        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            //        cerr<<"node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
            m_offset[node->mechanicalState] = sizeM;
            sizeM += node->mechanicalState->getMatrixSize();
        }

        // ==== Compliances require a block in the global compliance matrix
        if( node->mechanicalState  )
        {
            //            unsigned numCompliances=0;
            for(unsigned i=0; i<node->forceField.size(); i++)
            {
                if(node->forceField[i]->getComplianceMatrix(mparams))
                {
                    c_offset[node->forceField[i]] = sizeC;
                    sizeC += node->mechanicalState->getMatrixSize();
//                    numCompliances++;
                }
                else      // stiffness does not contribute to matrix size, since the stiffness matrix is added to the mass matrix of the state.
                {
                }
            }
            //            assert(numCompliances<2); // more than 1 compliance in a node makes no sense (or do they, if they are not all null ??? It would probably be better to replace them with their harmonic sum, however)
        }
        assert(node->forceField.size()<2);

        // ==== register all the DOFs
        if (node->mechanicalState != NULL)
            localDOFs.insert(node->mechanicalState);

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
            for(unsigned i=0; i<node->projectiveConstraintSet.size(); i++)
            {
                node->projectiveConstraintSet[i]->projectMatrix(&solver->_projMatrix,m_offset[node->mechanicalState]);
            }

            // Right-hand term
            unsigned offset = m_offset[node->mechanicalState]; // use a copy, because the parameter is modified by addToBaseVector
            //            cerr<<"MatrixAssemblyVisitor::processNodeTopDown, mechanical state "<< node->mechanicalState->getName() <<",  force before = " << solver->vecF << endl;
            node->mechanicalState->addToBaseVector(&solver->_vecF,core::VecDerivId::force(),offset);
            //            cerr<<"MatrixAssemblyVisitor::processNodeTopDown, mechanical state "<< node->mechanicalState->getName() <<", cumulated force = " << solver->vecF << endl;



        }

        // ==== mechanical mapping
        if ( node->mechanicalMapping != NULL )
        {
            //            cerr<<"MatrixAssemblyVisitor::processNodeTopDown, mapping  " << node->mechanicalMapping->getName()<< endl;
            const vector<sofa::defaulttype::BaseMatrix*>* pJs = node->mechanicalMapping->getJs();
            vector<core::BaseState*> pStates = node->mechanicalMapping->getFrom();
            assert( pJs->size() == pStates.size());
            MechanicalState* mtarget = dynamic_cast<MechanicalState*>(  node->mechanicalMapping->getTo()[0] ); // Only N-to-1 mappings are handled yet
            for( unsigned i=0; i<pStates.size(); i++ )
            {
                MechanicalState* mstate = dynamic_cast<MechanicalState*>(pStates[i]);
                assert(mstate);
                if( localDOFs.find(mstate)==localDOFs.end() ) // skip states which are not in the scope of the solver, such as mouse DOFs
                    continue;
                SMatrix J = getSMatrix( (*pJs)[i] );
                SMatrix contribution;
                //                cerr<<"MatrixAssemblyVisitor::processNodeTopDown, contribution of parent state  "<< mstate->getName() << endl;
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
            solver->_matM += JtMJtop;  // add J^T M J to the assembled mass matrix
        }

        // ==== compliance
        for(unsigned i=0; i<node->forceField.size(); i++ )
        {
            BaseForceField* ffield = node->forceField[i];
            if( ffield->getComplianceMatrix(mparams)!=NULL )
            {
                SMatrix compOffset = createShiftMatrix( node->mechanicalState->getMatrixSize(), sizeC, c_offset[ffield] );

                SMatrix J = SMatrix( compOffset.transpose() * jMap[node->mechanicalState] ); // shift J
                solver->_matJ += J;                                          // assemble

                SReal alpha = cparams.implicitVelocity(); // implicit velocity factor in the integration scheme
                SReal beta  = cparams.implicitPosition(); // implicit position factor in the integration scheme
                SReal l = alpha * (beta * mparams->dt() + ffield->getDampingRatio() );
                if( fabs(l)<1.0e-10 ) solver->serr << ffield->getName() << ", l is not invertible in ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown" << solver->sendl;
                SReal invl = 1.0/l;
                //            SMatrix complianceMatrix = getSMatrix(ffield->getComplianceMatrix(mparams));
                //            cerr<<"assembly, compliance " << ffield->getName() <<", compliance matrix = " << endl << complianceMatrix << endl;
                solver->_matC += SMatrix( compOffset.transpose() * getSMatrix(ffield->getComplianceMatrix(mparams)) * compOffset ) * invl;  // assemble

                // Right-hand term
                ffield->writeConstraintValue( &cparams, core::VecDerivId::force() );
                unsigned offset = c_offset[ffield]; // use a copy, because the parameter is modified by addToBaseVector
                node->mechanicalState->addToBaseVector(&solver->_vecPhi, core::VecDerivId::force(), offset );

            }
            else if (ffield->getStiffnessMatrix(mparams)) // accumulate the stiffness if the matrix is not null. TODO: Rayleigh damping
            {
                solver->_matM += SMatrix( jMap[node->mechanicalState].transpose() * getSMatrix(ffield->getStiffnessMatrix(mparams)) * jMap[node->mechanicalState] ) * mparams->dt();  // assemble
//                solver->_vecF += jMap[node->mechanicalState].transpose() * getSMatrix(ffield->getStiffnessMatrix(mparams)) *
            }
        }

    }
//    else if (pass== PROJECT_MATRICES)
//    {
//        // ==== independent DOFs
//        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
//        {
//            //            cerr<<"pass "<< pass << ", node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
//            // projections applied to the independent DOFs. The projection applied to mapped DOFs are ignored.
//            for(unsigned i=0; i<node->projectiveConstraintSet.size(); i++){
//                  node->projectiveConstraintSet[i]->projectMatrix(&solver->_projMatrix,m_offset[node->mechanicalState]);
//            }
//        }

//    }
    else if (pass== DISTRIBUTE_SOLUTION)
    {
        typedef defaulttype::BaseVector  SofaVector;

        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            //            cerr<<"pass "<< pass << ", node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
            unsigned offset = m_offset[node->mechanicalState]; // use a copy, because the parameter is modified by addToBaseVector
            node->mechanicalState->copyFromBaseVector(core::VecDerivId::force(), &solver->_vecF, offset );

            offset = m_offset[node->mechanicalState]; // use a copy, because the parameter is modified by addToBaseVector
            node->mechanicalState->copyFromBaseVector(core::VecDerivId::dx(), &solver->_vecDv, offset );
        }

    }
    else
    {
        cerr<<"ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown, unknown pass " << pass << endl;
    }

    return RESULT_CONTINUE;
}

//void ComplianceSolver::MatrixAssemblyVisitor::processNodeBottomUp(simulation::Node* /*node*/)
//{
//    if( pass==COMPUTE_SIZE )
//    {

//    }
//    else if (pass==DO_SYSTEM_ASSEMBLY)
//    {
//    }
////    else if (pass==PROJECT_MATRICES)
////    {
////    }
//    else if (pass==DISTRIBUTE_SOLUTION ) {
//    }
//    else {
//        cerr<<"ComplianceSolver::MatrixAssemblyVisitor::processNodeBottomUp, unknown pass " << pass << endl;
//    }

//}

const ComplianceSolver::SMatrix& ComplianceSolver::PMinvP()
{
    if( _PMinvP_isDirty ) // update it
    {
        inverseDiagonalMatrix( _PMinvP_Matrix.eigenMatrix, _matM, 1.0e-6 );
        _PMinvP_Matrix.eigenMatrix = P().transpose() * _PMinvP_Matrix.eigenMatrix * P();
        _PMinvP_isDirty = false;
    }
    return _PMinvP_Matrix.eigenMatrix;
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

ComplianceSolver::SMatrix ComplianceSolver::createIdentityMatrix( unsigned size )
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


void ComplianceSolver::inverseDiagonalMatrix( SMatrix& Minv, const SMatrix& M, SReal /*threshold*/)
{
    assert(M.rows()==M.cols() && "ComplianceSolver::inverseDiagonalMatrix needs a square matrix");
    Minv.resize(M.rows(),M.rows());
    for (int i=0; i<M.outerSize(); ++i)
    {
        Minv.startVec(i);
        for (SMatrix::InnerIterator it(M,i); it; ++it)
        {
            assert(i==it.col() && "ComplianceSolver::inverseDiagonalMatrix needs a diagonal matrix");
            Minv.insertBack(i,i) = 1.0/it.value();
        }
    }
    Minv.finalize();

}



}
}
}
