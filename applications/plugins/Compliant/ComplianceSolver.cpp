#include "ComplianceSolver.h"
#include <sofa/helper/AdvancedTimer.h>
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
    : implicitVelocity( initData(&implicitVelocity,(SReal)1.,"implicitVelocity","Weight of the next forces in the average forces used to update the velocities. 1 is implicit, 0 is explicit."))
    , implicitPosition( initData(&implicitPosition,(SReal)1.,"implicitPosition","Weight of the next velocities in the average velocities used to update the positions. 1 is implicit, 0 is explicit."))
    , f_rayleighStiffness( initData(&f_rayleighStiffness,0.1,"rayleighStiffness","Rayleigh damping coefficient related to stiffness, > 0") )
    , f_rayleighMass( initData(&f_rayleighMass,0.1,"rayleighMass","Rayleigh damping coefficient related to mass, > 0"))
    , verbose( initData(&verbose,false,"verbose","Print a lot of info for debug"))
{
}


void ComplianceSolver::solveEquation()
{
    if( _matC.rows() != 0 ) // solve constrained dynamics using a Schur complement (J.P.M^{-1}.P.J^T + C).lambda = c - J.M^{-1}.f
    {
        SMatrix schur( _matJ * PMinvP() * _matJ.transpose() + _matC );
        SparseLDLT schurDcmp(schur);
        VectorEigen& lambda = _vecLambda.getVectorEigen() =  _vecPhi.getVectorEigen() - _matJ * ( PMinvP() * _vecF.getVectorEigen() ); // right-hand term
        if( verbose.getValue() )
        {
            cerr<<"ComplianceSolver::solveEquation, schur complement = " << endl << Eigen::MatrixXd(schur) << endl;
            cerr<<"ComplianceSolver::solvEquatione,  Minv * vecF.getVectorEigen() = " << (PMinvP() * _vecF.getVectorEigen()).transpose() << endl;
            cerr<<"ComplianceSolver::solveEquation, matJ * ( Minv * vecF.getVectorEigen())  = " << ( _matJ * ( PMinvP() * _vecF.getVectorEigen())).transpose() << endl;
            cerr<<"ComplianceSolver::solveEquation,  vecPhi.getVectorEigen()  = " <<  _vecPhi.getVectorEigen().transpose() << endl;
            cerr<<"ComplianceSolver::solveEquation, right-hand term = " << lambda.transpose() << endl;
        }
        schurDcmp.solveInPlace( lambda );                                              // solve (J.M^{-1}.J^T + C).lambda = c - J.M^{-1}.f
        _vecF.getVectorEigen() = _vecF.getVectorEigen() + _matJ.transpose() * lambda ; // f = f_ext + J^T.lambda
        _vecDv.getVectorEigen() = PMinvP() * _vecF.getVectorEigen();                   // v = M^{-1}.f
        if( verbose.getValue() )
        {
            cerr<<"ComplianceSolver::solveEquation, constraint forces = " << lambda.transpose() << endl;
            cerr<<"ComplianceSolver::solveEquation, net forces = " << _vecF << endl;
            cerr<<"ComplianceSolver::solveEquation, vecDv = " << _vecDv << endl;
        }
    }
    else   // unconstrained dynamics, solve M.dv = f
    {
        SparseLDLT ldlt;
        ldlt.compute( _matM );
        {
            dv() = P() * f();
            ldlt.solveInPlace( dv() );
            dv() = P() * dv();
        }
//        else {
//            cerr<<"ComplianceSolver::solveEquation(), system matrix is singular"<<endl;
//        }
    }
}


void ComplianceSolver::solve(const core::ExecParams* params, double h, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    // tune parameters
    core::MechanicalParams cparams( *params);
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

    sofa::helper::AdvancedTimer::stepBegin("Build linear equation");

    // ==== Accumulate forces
    sofa::helper::AdvancedTimer::stepBegin("forces in the right-hand term");
    mop.computeForce(f);
    mop.projectResponse(f);
    if( verbose.getValue() )
    {
        cerr<<"ComplianceSolver::solve, filtered external forces = " << f << endl;
    }
    sofa::helper::AdvancedTimer::stepEnd("forces in the right-hand term");


    // ==== Resize global matrices and vectors
    _PMinvP_isDirty = true;
    MatrixAssemblyVisitor assembly(&cparams,this);
    this->getContext()->executeVisitor(&assembly(COMPUTE_SIZE)); // first the size
    //    cerr<<"ComplianceSolver::solve, sizeM = " << assembly.sizeM <<", sizeC = "<< assembly.sizeC << endl;

    _matM.resize(assembly.sizeM,assembly.sizeM);
    _matK.resize(assembly.sizeM,assembly.sizeM);
    _projMatrix.eigenMatrix = createIdentityMatrix(assembly.sizeM);
    _matC.resize(assembly.sizeC,assembly.sizeC);
    _matJ.resize(assembly.sizeC,assembly.sizeM);
    _vecF.resize(assembly.sizeM);
    _vecF.clear();
    _vecV.resize(assembly.sizeM);
    _vecV.clear();
    _vecDv.resize(assembly.sizeM);
    _vecDv.clear();
    _vecPhi.resize(assembly.sizeC);
    _vecPhi.clear();
    _vecLambda.resize(assembly.sizeC);
    _vecLambda.clear();



    // ==== Create local matrices J,K,M,C at each level
    s2mjc.clear();
    this->getContext()->executeVisitor(&assembly(DO_SYSTEM_ASSEMBLY));



    // ==== Global matrix assembly
    sofa::helper::AdvancedTimer::stepBegin("JMJt, JKJt, JCJt");
    for(State_MJC::iterator i=s2mjc.begin(),iend=s2mjc.end(); i!=iend; i++ )
    {
        core::behavior::BaseMechanicalState* s = (*i).first;
//        if(verbose.getValue()){
//            cerr<<"ComplianceSolver::solve, state = " << s->getName() << endl;
//            cerr<<"  M = " << s2mjc[s].M << endl;
//            cerr<<"  K = " << s2mjc[s].K << endl;
//            cerr<<"  J = " << s2mjc[s].J << endl;
//            cerr<<"  c_offset = " << s2mjc[s].c_offset << endl;
//            cerr<<"  C = " << s2mjc[s].C << endl;
//        }
        if( s2mjc[s].M.rows()>0 )
            _matM += s2mjc[s].J.transpose() * s2mjc[s].M * s2mjc[s].J;
        if( s2mjc[s].K.rows()>0 )
            _matK += s2mjc[s].J.transpose() * s2mjc[s].K * s2mjc[s].J;
        if( s2mjc[s].C.rows()>0 )
        {
            SMatrix C0 = createShiftMatrix( s2mjc[s].C.rows(), _matC.cols(), s2mjc[s].c_offset );
            _matC += C0.transpose() * s2mjc[s].C * C0;
//            cerr<<"  C0 = " << C0 << endl;
//            cerr<<"  J = " << s2mjc[s].J << endl;
//            cerr<<"  matJ before = " << _matJ << endl;
//            cerr<<"  matJ += " << dJ << endl;
            _matJ += SMatrix(C0.transpose() * s2mjc[s].J);        // J vertically shifted, aligned with the compliance matrix
//            cerr<<"  matJ after = " << _matJ << endl;
        }
    }
    _matK = P() * _matK * P();  /// Filter the matrix. @todo this is not enough to guarantee that the projected DOFs are isolated. M should be set diagonal.
    sofa::helper::AdvancedTimer::stepEnd("JMJt, JKJt, JCJt");

    if( verbose.getValue() )
    {
        cerr<<"ComplianceSolver::solve, final matrices ==================================== " << endl ;
        cerr<<"ComplianceSolver::solve, final M = " << endl << _matM << endl;
        cerr<<"ComplianceSolver::solve, final K = " << endl << _matK << endl;
        cerr<<"ComplianceSolver::solve, final P = " << endl << P() << endl;
        cerr<<"ComplianceSolver::solve, final C = " << endl << _matC << endl;
        cerr<<"ComplianceSolver::solve, final J = " << endl << _matJ << endl;
        cerr<<"ComplianceSolver::solve, final f = " << _vecF << endl;
        cerr<<"ComplianceSolver::solve, final phi = " << _vecPhi << endl;
        cerr<<"ComplianceSolver::solve, final v = " << _vecV << endl;
    }



    // ==== Compute the implicit matrix and right-hand term
    sofa::helper::AdvancedTimer::stepBegin("implicit equation: scaling and sum of matrices, update right-hand term ");
    SReal rs = f_rayleighStiffness.getValue();
    SReal rm = f_rayleighMass.getValue();
    // complete the right-hand term b = f0 + (h+rs) K v - rm M v    Rayleigh mass factor rm is used with a negative sign because it is recorded as a positive real, while its force is opposed to the velocity
    _vecF.getVectorEigen() += _matK * _vecV.getVectorEigen() * ( implicitVelocity.getValue() * (h + rs) );
//    cerr<<"ComplianceSolver::solve, added to f: " << (_matK * _vecV.getVectorEigen() * (implicitVelocity.getValue() * (h + rs) )) << endl;
    _vecF.getVectorEigen() -= _matM * _vecV.getVectorEigen() * rm;
//    cerr<<"ComplianceSolver::solve, substracted from f: " << (_matM * _vecV.getVectorEigen() * rm) << endl;
    _vecF.getVectorEigen() = P() * _vecF.getVectorEigen(); // filter the right-hand term
    // The implicit matrix is scaled by 1/h before solving the equation: M = ( (1+h*rm)M - h*B - h*(h+rs)K  )/h = (rm+1/h)M - (rs+h)K  since we ignore B
    _matM *= rm + 1.0/h;
    _matM -= _matK * (h+rs);
//    cerr<<"ComplianceSolver::solve, final M.size() = " <<  _matM.rows() << ", non zeros: " << _matM.nonZeros() << endl; //49548
    if( verbose.getValue() )
    {
        cerr<<"ComplianceSolver::solve, implicit matrix = " << endl << _matM << endl;
        cerr<<"ComplianceSolver::solve, right-hand term = " << _vecF << endl;
    }

    sofa::helper::AdvancedTimer::stepEnd("implicit equation: scaling and sum of matrices, update right-hand term ");
    sofa::helper::AdvancedTimer::stepEnd("Build linear equation");

    // ==== Solve equation system
    sofa::helper::AdvancedTimer::stepBegin("Solve linear equation");
    solveEquation();
    sofa::helper::AdvancedTimer::stepEnd("Solve linear equation");

    this->getContext()->executeVisitor(&assembly(DISTRIBUTE_SOLUTION));  // set dv in each MechanicalState



    // ==== Apply the result

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
    ComplianceSolver::State_MJC& s2mjc = solver->s2mjc;

    //    cerr<<"ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown visit Node "<<node->getName()<<endl;
    if( pass== COMPUTE_SIZE )
    {
        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            //        cerr<<"node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
            s2mjc[node->mechanicalState].m_offset = sizeM;
            sizeM += node->mechanicalState->getMatrixSize();
        }

        // ==== Compliances require a block in the global compliance matrix
        if( node->mechanicalState  )
        {
            for(unsigned i=0; i<node->forceField.size(); i++)
            {
                if(node->forceField[i]->getComplianceMatrix(mparams))
                {
                    s2mjc[node->mechanicalState].c_offset = sizeC;
                    sizeC += node->mechanicalState->getMatrixSize();
                }
                else      // stiffness does not contribute to matrix size, since the stiffness matrix is added to the mass matrix of the state.
                {
                }
            }
        }

        // ==== register all the DOFs
        if (node->mechanicalState != NULL)
            localDOFs.insert(node->mechanicalState);

    }
    else if (pass== DO_SYSTEM_ASSEMBLY)
    {
        if( node->mechanicalState == NULL ) return RESULT_CONTINUE;

        SMatrix& J0 = s2mjc[node->mechanicalState].J;
        SMatrix& localM  = s2mjc[node->mechanicalState].M;
        SMatrix& localC  = s2mjc[node->mechanicalState].C;
        SMatrix& localK  = s2mjc[node->mechanicalState].K;
        const unsigned& m_offset = s2mjc[node->mechanicalState].m_offset;
        const unsigned& c_offset = s2mjc[node->mechanicalState].c_offset;
        unsigned localSize = node->mechanicalState->getMatrixSize();

        // ==== independent DOFs
        if ( node->mechanicalMapping == NULL )
        {
            sofa::helper::AdvancedTimer::stepBegin("shift and project independent states");
            J0 = createShiftMatrix( localSize, sizeM, m_offset );

            // projections applied to the independent DOFs. The projection applied to mapped DOFs are ignored.
            for(unsigned i=0; i<node->projectiveConstraintSet.size(); i++)
            {
                node->projectiveConstraintSet[i]->projectMatrix(&solver->_projMatrix,m_offset);
            }

            // Right-hand term (includes force applied to slave dofs, mapped upward)
            unsigned offset = m_offset; // use a copy, because the parameter is modified by addToBaseVector
            //            cerr<<"MatrixAssemblyVisitor::processNodeTopDown, mechanical state "<< node->mechanicalState->getName() <<",  force before = " << solver->vecF << endl;
            node->mechanicalState->addToBaseVector(&solver->_vecF,core::VecDerivId::force(),offset);
            //            cerr<<"MatrixAssemblyVisitor::processNodeTopDown, mechanical state "<< node->mechanicalState->getName() <<", cumulated force = " << solver->vecF << endl;
            offset = m_offset; // use a copy, because the parameter is modified by addToBaseVector
            node->mechanicalState->addToBaseVector(&solver->_vecV, core::VecDerivId::velocity(), offset );

            sofa::helper::AdvancedTimer::stepEnd("shift and project independent states");


        }
        else    // process the mapping
        {
            sofa::helper::AdvancedTimer::stepBegin(        "J products");
//            cerr<<"MatrixAssemblyVisitor::processNodeTopDown, mapping  " << node->mechanicalMapping->getName()<< endl;
            const vector<sofa::defaulttype::BaseMatrix*>* pJs = node->mechanicalMapping->getJs();
            const vector<sofa::defaulttype::BaseMatrix*>* pKs = node->mechanicalMapping->getKs();
            vector<core::BaseState*> pStates = node->mechanicalMapping->getFrom();
            assert( pJs->size() == pStates.size());
            MechanicalState* cstate = dynamic_cast<MechanicalState*>(  node->mechanicalMapping->getTo()[0] ); // Child state. Only N-to-1 mappings are handled yet
            for( unsigned i=0; i<pStates.size(); i++ )
            {
                MechanicalState* pstate = dynamic_cast<MechanicalState*>(pStates[i]);  // parent state
                assert(pstate);
                if( localDOFs.find(pstate)==localDOFs.end() ) // skip states which are not in the scope of the solver, such as mouse DOFs
                    continue;
                SMatrix Jcp = getSMatrix( (*pJs)[i] ); // child wrt parent
                SMatrix& Jp0 = s2mjc[pstate].J;        // parent wrt global DOF

                // contribute to the Jacobian matrix of the child wrt global DOF
                SMatrix& Jc0 = s2mjc[cstate].J;  // child wrt global DOF;
                if( Jc0.rows()==0 )
                    Jc0 = SMatrix(Jcp * Jp0);
                else
                    Jc0 += SMatrix(Jcp * Jp0);

                // Geometric stiffness
                if( pKs!=NULL && (*pKs)[i]!=NULL )
                {
                    SMatrix K = getSMatrix( (*pKs)[i] ); // geometric stiffness related to this parent
                    SMatrix& pK = s2mjc[pstate].K;       // parent stiffness
                    if( pK.rows()==0 )
                        pK = K;
                    else pK += K;
//                    cerr<<"MatrixAssemblyVisitor::processNodeTopDown, mapping  " << node->mechanicalMapping->getName()<<" adding geometric stiffness to parent : " << endl << K << endl;
                }
            }
            sofa::helper::AdvancedTimer::stepEnd(        "J products");
        }

        // ==== mass
        if (node->mass != NULL  )
        {
            sofa::helper::AdvancedTimer::stepBegin( "local M");

            // todo: better way to fill the mass matrix
            typedef linearsolver::EigenSparseSquareMatrix<SReal> Sqmat;
            Sqmat sqmat(node->mechanicalState->getMatrixSize(),node->mechanicalState->getMatrixSize());
            linearsolver::SingleMatrixAccessor accessor( &sqmat );
            node->mass->addMToMatrix( mparams, &accessor );
            localM = sqmat.eigenMatrix;

            sofa::helper::AdvancedTimer::stepEnd( "local M");
        }

        // ==== compliance
        for(unsigned i=0; i<node->forceField.size(); i++ )
        {
            BaseForceField* ffield = node->forceField[i];
            //            cerr<<"MatrixAssemblyVisitor::processNodeTopDown, forcefield " << ffield->getName() << endl;
            if( ffield->getComplianceMatrix(mparams)!=NULL )
            {
                sofa::helper::AdvancedTimer::stepBegin( "local C and right-hand term");
                //                cerr<<"MatrixAssemblyVisitor::processNodeTopDown, forcefield " << ffield->getName() << "has compliance" << endl;
                SMatrix compOffset = createShiftMatrix( node->mechanicalState->getMatrixSize(), sizeC, c_offset );

                // compute scaling of C, based on time step, damping and implicit coefficients
                SReal alpha = cparams.implicitVelocity(); // implicit velocity factor in the integration scheme
                SReal beta  = cparams.implicitPosition(); // implicit position factor in the integration scheme
                SReal l = alpha * (beta * mparams->dt() + ffield->getDampingRatio() );
                if( fabs(l)<1.0e-10 ) solver->serr << ffield->getName() << ", l is not invertible in ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown" << solver->sendl;
                SReal invl = 1.0/l;

                localC = getSMatrix(ffield->getComplianceMatrix(mparams)) *  invl;

                // Right-hand term
                ffield->writeConstraintValue( &cparams, core::VecDerivId::force() );
                unsigned offset = c_offset; // use a copy, because the parameter is modified by addToBaseVector
                node->mechanicalState->addToBaseVector(&solver->_vecPhi, core::VecDerivId::force(), offset );
                sofa::helper::AdvancedTimer::stepEnd( "local C and right-hand term");
            }
            else if (ffield->getStiffnessMatrix(mparams)) // accumulate the stiffness if the matrix is not null. TODO: Rayleigh damping
            {
                sofa::helper::AdvancedTimer::stepBegin( "local K");
                //                cerr<<"MatrixAssemblyVisitor::processNodeTopDown, forcefield " << ffield->getName() << "has stiffness" << endl;
                if( localK.rows() != (int) localSize )
                    localK.resize(localSize, localSize);
                localK += getSMatrix(ffield->getStiffnessMatrix(mparams));
                sofa::helper::AdvancedTimer::stepEnd( "local K");
            }
        }

    }

    else if (pass== DISTRIBUTE_SOLUTION)
    {
        typedef defaulttype::BaseVector  SofaVector;

        // ==== independent DOFs
        if (node->mechanicalState != NULL  && node->mechanicalMapping == NULL )
        {
            //            cerr<<"pass "<< pass << ", node " << node->getName() << ", independent mechanical state: " << node->mechanicalState->getName() << endl;
            unsigned offset = s2mjc[node->mechanicalState].m_offset; // use a copy, because the parameter is modified by addToBaseVector
            node->mechanicalState->copyFromBaseVector(core::VecDerivId::force(), &solver->_vecF, offset );

            offset = s2mjc[node->mechanicalState].m_offset; // use a copy, because the parameter is modified by addToBaseVector
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
        if( !inverseDiagonalMatrix( _PMinvP_Matrix.eigenMatrix, _matM ) )
            assert(false);
        _PMinvP_Matrix.eigenMatrix = P().transpose() * _PMinvP_Matrix.eigenMatrix * P();
        _PMinvP_isDirty = false;
    }
    return _PMinvP_Matrix.eigenMatrix;
}



/// Return a rectangular matrix (cols>rows), with (offset-1) null columns, then the (rows*rows) identity, then null columns.
/// This is used to shift a "local" matrix to the global indices of an assembly matrix.
ComplianceSolver::SMatrix ComplianceSolver::createShiftMatrix( unsigned rows, unsigned cols, unsigned offset )
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


bool ComplianceSolver::inverseDiagonalMatrix( SMatrix& Minv, const SMatrix& M )
{
    if(M.rows()!=M.cols() || M.nonZeros()!=M.rows() ) // test if diagonal. WARNING: some non-diagonal matrix pass the test, but they are unlikely in this context.
        return false;
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
    return true;
}

void ComplianceSolver::inverseMatrix(SMatrix& Minv, const SMatrix& M)
{
    cerr<<"ComplianceSolver::inverseMatrix NOT IMPLEMENTED !!!!"<< endl;
}



}
}
}
