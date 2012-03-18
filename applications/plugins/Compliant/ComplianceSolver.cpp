#include "ComplianceSolver.h"
#include "BaseCompliance.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>
#include <sofa/component/linearsolver/EigenSparseSquareMatrix.h>
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
{
}

void ComplianceSolver::bwdInit()
{
    core::ExecParams params;
}

void ComplianceSolver::solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    // tune parameters
    core::ComplianceParams cparams(*params);
    cparams.setMFactor(1.0);
    cparams.setDt(dt);
    cparams.setImplicitVelocity( implicitVelocity.getValue() );
    cparams.setImplicitPosition( implicitPosition.getValue() );

    //  State vectors and operations
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecDeriv f  (&vop, core::VecDerivId::force() );
    MultiVecCoord pos2(&vop, xResult );
    MultiVecDeriv vel2(&vop, vResult );



    // Compute right-hand term
    mop.computeForce(f);

    // Matrix size
    MatrixAssemblyVisitor assembly(&cparams,this);
    this->getContext()->executeVisitor(&assembly(COMPUTE_SIZE));
    cerr<<"ComplianceSolver::solve, sizeM = " << assembly.sizeM <<", sizeC = "<< assembly.sizeC << endl;

    matM.resize(assembly.sizeM,assembly.sizeM);
    matP.resize(assembly.sizeM,assembly.sizeM);
    matC.resize(assembly.sizeC,assembly.sizeC);
    matJ.resize(assembly.sizeC,assembly.sizeM);
    vecF.resize(assembly.sizeM);
    vecPhi.resize(assembly.sizeC);

    // Matrix assembly
    this->getContext()->executeVisitor(&assembly(MATRIX_ASSEMBLY));

    cerr<<"ComplianceSolver::solve, final M = " << endl << matM << endl;
    cerr<<"ComplianceSolver::solve, final P = " << endl << matP << endl;
    cerr<<"ComplianceSolver::solve, final C = " << endl << matC << endl;
    cerr<<"ComplianceSolver::solve, final J = " << endl << matJ << endl;

    // Vector assembly  (do we need a separate pass ?)
    vecF.clear();
    vecPhi.clear();
    this->getContext()->executeVisitor(&assembly(VECTOR_ASSEMBLY));

    cerr<<"ComplianceSolver::solve, final f = " << vecF << endl;
    cerr<<"ComplianceSolver::solve, final phi = " << vecPhi << endl;

    // Solve equation system

    SMatrix Minv = inverseMatrix( matM, 1.0e-6 ); //cerr<<"ComplianceSolver::solve, Minv has " << Minv.nonZeros() << "non-null entries " << endl;
    Minv = matP * Minv * matP;
    SMatrix schur( matJ * Minv * matJ.transpose() + matC );
    SparseLDLT schurDcmp(schur);
    VectorEigen x = vecPhi.getVectorEigen() - matJ * ( Minv * vecF.getVectorEigen() );
    {
//        cerr<<"ComplianceSolver::solve, Minv = " << endl << Eigen::MatrixXd(Minv) << endl;
        cerr<<"ComplianceSolver::solve, schur complement = " << endl << Eigen::MatrixXd(schur) << endl;
        cerr<<"ComplianceSolver::solve,  Minv * vecF.getVectorEigen() = " << (Minv * vecF.getVectorEigen()).transpose() << endl;
        cerr<<"ComplianceSolver::solve, matJ * ( Minv * vecF.getVectorEigen())  = " << ( matJ * ( Minv * vecF.getVectorEigen())).transpose() << endl;
        cerr<<"ComplianceSolver::solve,  vecPhi.getVectorEigen()  = " <<  vecPhi.getVectorEigen().transpose() << endl;
        cerr<<"ComplianceSolver::solve, right-hand term = " << x << endl;
    }
    schurDcmp.solveInPlace( x ); // solve (J.M^{-1}.J^T + C).x = c - J.M^{-1}.f
    cerr<<"ComplianceSolver::solve, constraint forces = " << x << endl;
    vecF.getVectorEigen() = Minv * ( vecF.getVectorEigen() + matJ.transpose() * x ); // dv = M^{-1}.( f_ext + J^T.lambda )   stored in vector f
    cerr<<"ComplianceSolver::solve, velocity change = " << vecF << endl;

    this->getContext()->executeVisitor(&assembly(VECTOR_DISTRIBUTE));  // set dv in each MechanicalState


    // Apply integration scheme

    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp ops;
    ops.resize(2);
    ops[0].first = pos2; // p = p + v*h + dv*h*beta
    ops[0].second.push_back(std::make_pair(pos.id(),1.0));
    ops[0].second.push_back(std::make_pair(vel.id(),dt));
    ops[0].second.push_back(std::make_pair(  f.id(),dt*implicitPosition.getValue()));
    ops[1].first = vel2; // v = v + dv
    ops[1].second.push_back(std::make_pair(vel.id(),1.0));
    ops[1].second.push_back(std::make_pair(  f.id(),1.0));

    vop.v_multiop(ops);


}



simulation::Visitor::Result ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown(simulation::Node* node)
{
    if( pass==COMPUTE_SIZE )
    {
        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            //        cerr<<"node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
            m_offset[node->mechanicalState] = sizeM;
            sizeM += node->mechanicalState->getMatrixSize();
        }


        // ==== process compliances
        vector<BaseCompliance*> compliances;
        node->getNodeObjects<BaseCompliance>(&compliances);
        //    cerr<<"node " << node->getName() << ", compliances: " << endl;
        //    for(unsigned i=0; i<compliances.size(); i++ ){
        //        cerr<< compliances[i]->getName() <<", ";
        //    }
        //    cerr<<endl;
        if( compliances.size()>0 )
        {
            assert(node->mechanicalState);
            c_offset[compliances[0]] = sizeC;
            sizeC += node->mechanicalState->getMatrixSize();
        }
        assert(compliances.size()<2);

    }
    else if (pass==MATRIX_ASSEMBLY)
    {
        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            cerr<<"pass "<< pass << ", node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
            ComplianceSolver::DMatrix shiftMatrix = createShiftMatrix( node->mechanicalState->getMatrixSize(), sizeM, m_offset[node->mechanicalState] );
            jStack.push( shiftMatrix );

            // projections applied to the independent DOFs. The projection applied to mapped DOFs are ignored.
            DMatrix projMat = createIdentityMatrix(node->mechanicalState->getMatrixSize());
            vector<BaseProjectiveConstraintSet*> projections;
            node->getNodeObjects<BaseProjectiveConstraintSet>(&projections);
            for(unsigned i=0; i<projections.size(); i++)
            {
                if( const defaulttype::BaseMatrix* mat = projections[i]->getJ(mparams) )
                {
//                    cerr<<"MatrixAssemblyVisitor::processNodeTopDown, current projMat = " << projMat << endl;
                    projMat = projMat * toMatrix(mat);  // multiply with the projection matrix
//                    cerr<<"MatrixAssemblyVisitor::processNodeTopDown, constraint matrix = " << *mat << endl;
//                    cerr<<"MatrixAssemblyVisitor::processNodeTopDown, constraint matrix = " << toMatrix(mat) << endl;
//                    cerr<<"MatrixAssemblyVisitor::processNodeTopDown, new projMat = " << projMat << endl;
                }
            }
            solver->matP += shiftMatrix.transpose() * projMat * shiftMatrix;


        }

        // ==== mechanical mapping
        if ( node->mechanicalMapping != NULL )
        {
            DMatrix J = toMatrix( node->mechanicalMapping->getJ() );
            DMatrix newJtop;
            newJtop = J * jStack.top();
            jStack.push( newJtop );
            //                    cerr<<"pass "<< pass << ", node " << node->getName() << ", mechanical mapping: " << node->mechanicalMapping->getName() << ", matrix J = " << J << endl;
            //                    cerr<<"pass "<< pass << ", node " << node->getName() << ", mechanical mapping: " << node->mechanicalMapping->getName() << ", new Jtop = " << jStack.top() << endl;
        }

        // ==== mass
        if (node->mass != NULL  )
        {
            cerr<<"pass "<< pass << ", node " << node->getName() << ", mass: " << node->mass->getName() << endl;
            assert( node->mechanicalState != NULL );
            linearsolver::EigenSparseSquareMatrix<SReal> sqmat(node->mechanicalState->getMatrixSize(),node->mechanicalState->getMatrixSize());
            linearsolver::SingleMatrixAccessor accessor( &sqmat );
            node->mass->addMToMatrix( mparams, &accessor );
            //                    cerr<<"eigen matrix of the mass: " << sqmat  << endl;
            DMatrix JtMJtop;
            JtMJtop = jStack.top().transpose() * sqmat.eigenMatrix * jStack.top();
            //                    cerr<<"contribution to the mass matrix: " << endl << JtMJtop << endl;
            solver->matM += JtMJtop;  // add J^T M J to the assembled mass matrix
        }

        // ==== compliance
        vector<BaseCompliance*> compliances;
        node->getNodeObjects<BaseCompliance>(&compliances);
        if( compliances.size()>0 )
        {
            DMatrix compOffset = createShiftMatrix( node->mechanicalState->getMatrixSize(), sizeC, c_offset[compliances[0]] );

            DMatrix J = DMatrix( compOffset.transpose() * jStack.top() ); // shift J
            solver->matJ += J;                                          // assemble

            DMatrix C = DMatrix( compOffset.transpose() * toMatrix(compliances[0]->getMatrix(mparams)) * compOffset ); // shift C
            SReal alpha = cparams.implicitVelocity(); // implicit velocity factor in the integration scheme
            SReal beta  = cparams.implicitPosition(); // implicit position factor in the integration scheme
            SReal l = alpha * (beta * mparams->dt() + compliances[0]->dampingRatio.getValue() );
            if( fabs(l)<1.0e-10 ) solver->serr << compliances[0]->getName() << ", l is not invertible in ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown" << solver->sendl;
            SReal invl = 1.0*l;
            solver->matC += C * invl;                                                                                // assemble
        }

    }
    else if (pass==VECTOR_ASSEMBLY)
    {
        typedef defaulttype::BaseVector  SofaVector;

        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            cerr<<"pass "<< pass << ", node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;

            node->mechanicalState->addToBaseVector(&solver->vecF,core::VecDerivId::force(),m_offset[node->mechanicalState]);
        }

        // ==== compliance
        vector<BaseCompliance*> compliances;
        node->getNodeObjects<BaseCompliance>(&compliances);
        if( compliances.size()>0 )
        {
            cerr<<"pass "<< pass << ", node " << node->getName() << ", compliance: " << compliances[0]->getName() << endl;

            compliances[0]->setConstraint( &cparams, core::VecDerivId::force() );
            node->mechanicalState->addToBaseVector(&solver->vecPhi, core::VecDerivId::force(), c_offset[compliances[0]]);
        }

    }
    else if (pass==VECTOR_DISTRIBUTE)
    {
        typedef defaulttype::BaseVector  SofaVector;

        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            cerr<<"pass "<< pass << ", node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;

            node->mechanicalState->copyFromBaseVector(core::VecDerivId::force(), &solver->vecF, m_offset[node->mechanicalState]);
        }

//        // ==== compliance
//        vector<BaseCompliance*> compliances;
//        node->getNodeObjects<BaseCompliance>(&compliances);
//        if( compliances.size()>0 )
//        {
//            cerr<<"pass "<< pass << ", node " << node->getName() << ", compliance: " << compliances[0]->getName() << endl;

//            compliances[0]->setConstraint( &cparams, core::VecDerivId::force() );
//            node->mechanicalState->copyFromBaseVector(core::VecDerivId::force(), &solver->vecPhi, c_offset[compliances[0]]);
//        }

    }
    else
    {
        cerr<<"ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown, unknown pass " << pass << endl;
    }

    return RESULT_CONTINUE;

}

void ComplianceSolver::MatrixAssemblyVisitor::processNodeBottomUp(simulation::Node* node)
{
    if( pass==COMPUTE_SIZE )
    {

    }
    else if (pass==MATRIX_ASSEMBLY)
    {
        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            jStack.pop();
        }

        // ==== mechanical mapping
        if ( node->mechanicalMapping != NULL )
        {
            jStack.pop();
        }
    }
    else if (pass==VECTOR_ASSEMBLY )
    {
    }
    else if (pass==VECTOR_DISTRIBUTE )
    {
    }
    else
    {
        cerr<<"ComplianceSolver::MatrixAssemblyVisitor::processNodeBottomUp, unknown pass " << pass << endl;
    }

}

/// Return a rectangular matrix (cols>rows), with (offset-1) null columns, then the (rows*rows) identity, then null columns.
/// This is used to shift a "local" matrix to the global indices of an assembly matrix.
ComplianceSolver::DMatrix ComplianceSolver::MatrixAssemblyVisitor::createShiftMatrix( unsigned rows, unsigned cols, unsigned offset )
{
    DMatrix m(rows,cols);
    for(unsigned i=0; i<rows; i++ )
        m.coeffRef(i,offset+i)=1;
    return m;
}

ComplianceSolver::DMatrix ComplianceSolver::MatrixAssemblyVisitor::createIdentityMatrix( unsigned size )
{
    DMatrix m(size,size);
    for(unsigned i=0; i<size; i++ )
        m.coeffRef(i,i)=1;
    return m;
}

/// Converts a BaseMatrix to the matrix type used here.
ComplianceSolver::DMatrix ComplianceSolver::MatrixAssemblyVisitor::toMatrix( const defaulttype::BaseMatrix* m)
{
    ComplianceSolver::DMatrix result(m->rowSize(), m->colSize());

    int R = m->getBlockRows();
    int C = m->getBlockCols();
    //    cerr<<"ComplianceSolver::MatrixAssemblyVisitor::toMatrix, R = " << R << ", C = " << C << endl;
    for(defaulttype::BaseMatrix::RowBlockConstIterator ri= m->bRowsBegin(), end_rows=m->bRowsEnd(); ri!=end_rows; ri++ ) // for each row of blocks
    {
        for( defaulttype::BaseMatrix::ColBlockConstIterator ci = ri.begin(), end_cols=ri.end(); ci!=end_cols; ci++  )
        {
            const defaulttype::BaseMatrix::BlockConstAccessor& b= ci.bloc();
            for(int i=0; i<R; i++ )   // for each scalar row of the blocks
            {
                for(int j=0; j<C; j++)
                {
                    //                    cerr<<"ComplianceSolver::toMatrix insert value "<<  b.element(i,j) << " in row " << b.getRow()+i << ", col " << b.getCol()+j << endl;
                    result.coeffRef( R*b.getRow()+i, C*b.getCol()+j ) = b.element(i,j);
                }
            }
        }
    }
    return result;
}


/// Converts a BaseMatrix to the matrix type used here.
ComplianceSolver::SMatrix ComplianceSolver::inverseMatrix( const DMatrix& M, SReal threshold)
{
    assert(M.rows()==M.cols());
    SparseLDLT Mdcmp(M);
    SMatrix result(M.rows(),M.rows());
    for( int i=0; i<M.rows(); i++ )
    {
        VectorEigen v(M.rows());
        v.fill(0);
        v(i)=1;
        Mdcmp.solveInPlace(v);
        result.startVec(i);
        for(int j=0; j<M.rows(); j++)
        {
            if( fabs(v(j))>=threshold )
                result.insertBack(j,i) = v(j);
        }
    }
    result.finalize();
    return result;
}



}
}
}
