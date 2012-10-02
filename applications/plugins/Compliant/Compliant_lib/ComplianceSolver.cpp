#include "ComplianceSolver.h"
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>
#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/SingleMatrixAccessor.h>
#include <sofa/component/linearsolver/EigenVectorWrapper.h>
#include <iostream>

#include "utils/scoped.h"

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace odesolver
{
typedef linearsolver::EigenVectorWrapper<SReal> Wrap;


using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;

SOFA_DECL_CLASS(ComplianceSolver);
int ComplianceSolverClass = core::RegisterObject("A simple explicit time integrator").add< ComplianceSolver >();


#define __METHOD__ ( std::string(this->getClassName()) + "::" + __func__)
  


ComplianceSolver::ComplianceSolver()
    : implicitVelocity( initData(&implicitVelocity,(SReal)1.,"implicitVelocity","Weight of the next forces in the average forces used to update the velocities. 1 is implicit, 0 is explicit."))
    , implicitPosition( initData(&implicitPosition,(SReal)1.,"implicitPosition","Weight of the next velocities in the average velocities used to update the positions. 1 is implicit, 0 is explicit."))
    , f_rayleighStiffness( initData(&f_rayleighStiffness,0.1,"rayleighStiffness","Rayleigh damping coefficient related to stiffness, > 0") )
    , f_rayleighMass( initData(&f_rayleighMass,0.1,"rayleighMass","Rayleigh damping coefficient related to mass, > 0"))
    , verbose( initData(&verbose,false,"verbose","Print a lot of info for debug"))
{
}

void ComplianceSolver::init()
{
//    cerr<<"ComplianceSolver::init()" << endl;
}

void ComplianceSolver::MatrixAssemblyVisitor::writeLocalMatrices() const
{

        for( State_2_LocalMatrices::const_iterator i=localMatrices.begin(), iend=localMatrices.end(); i!=iend; i++ ){
            cerr<< (*i).first->getName() << ": " << (*i).second << endl;
        }
}


  ComplianceSolver::MatrixAssemblyVisitor::MatrixAssemblyVisitor(const core::MechanicalParams* params, ComplianceSolver* s)
    : simulation::MechanicalVisitor(params)
    , solver(s)
    , cparams(*params)
    , sizeM(0)
    , sizeC(0)
    , pass(COMPUTE_SIZE)
  {
	     
  
  }



void ComplianceSolver::solveEquation()
{
    if( _matC.rows() ) {
      // solve constrained dynamics using a Schur complement:
      // (J.P.M^{-1}.P.J^T + C).lambda = c - J.M^{-1}.f

      // schur complement matrix
      SMatrix schur = _matJ * PMinvP() * _matJ.transpose();
      schur += _matC ;
      
      // factorization
      Cholesky schurDcmp(schur); // TODO check singularity ?

      // right-hand side term
      VectorEigen& lambda = _vecLambda =  _vecPhi - _matJ * ( PMinvP() * _vecF ); 

      if( verbose.getValue() ) {
	cerr<< __METHOD__ << " schur complement = " << endl << Eigen::MatrixXd(schur) << endl
	    << __METHOD__ << " Minv * vecF = " << (PMinvP() * _vecF).transpose() << endl
	    << __METHOD__ << " matJ * ( Minv * vecF)  = " << ( _matJ * ( PMinvP() * _vecF)).transpose() << endl
	    << __METHOD__ << " vecPhi  = " <<  _vecPhi.transpose() << endl
	    << __METHOD__ << " right-hand term = " << lambda.transpose() << endl;
      }

      // Lagrange multipliers
      lambda = schurDcmp.solve( lambda );

      // f = f_ext + J^T.lambda
      VectorEigen netForces = _vecF + _matJ.transpose() * lambda ; 

      // v = M^{-1}.f
      _vecDv = PMinvP() * netForces; 
      
      if( verbose.getValue() ) {
	cerr<< __METHOD__ << " constraint forces = " << lambda.transpose() << endl
	    << __METHOD__ << " net forces = " << _vecF.transpose() << endl
	    << __METHOD__ << " vecDv = " << _vecDv.transpose() << endl;
      }
      
    } else {
      // unconstrained dynamics, solve M.dv = f

      Cholesky ldlt;
      ldlt.compute( _matM );	// TODO check singularity ?
        
      dv() = P() * f();
      dv() = ldlt.solve( dv() );
      dv() = P() * dv();
	
    }
}
  

ComplianceSolver::MatrixAssemblyVisitor* ComplianceSolver::newAssemblyVisitor(const core::MechanicalParams& params) {
  return new MatrixAssemblyVisitor(&params, this);
}


void ComplianceSolver::MatrixAssemblyVisitor::onCompliance(core::behavior::BaseForceField* , 
							     unsigned , 
							     unsigned ) { }
  
void ComplianceSolver::resize(unsigned sizeM, unsigned sizeC ) {
 
  _projMatrix.compressedMatrix = createIdentityMatrix( sizeM );
  
  // TODO check that multi-assignment isn't stabbing us in the back
  _vecF = _vecV = _vecDv = VectorEigen::Zero( sizeM );
  _vecPhi = _vecLambda = VectorEigen::Zero( sizeC );
  
  //    writeLocalMatrices();
}




void ComplianceSolver::solve(const core::ExecParams* params, 
			     double h, 
			     sofa::core::MultiVecCoordId xResult, 
			     sofa::core::MultiVecDerivId vResult) {
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

  // obtain an assembly visitor (possibly from derived classes)
  scoped::ptr<MatrixAssemblyVisitor> assembly( newAssemblyVisitor(cparams) );
 
  {
    scoped::timer step("Build linear equation");
    
    // === external forces
    { 	
      scoped::timer step("forces in the right-hand term");
      
      mop.computeForce(f);
      mop.projectResponse(f);
      
      if( verbose.getValue() ) {
	cerr << __METHOD__ << " filtered external forces = " << f << endl;
      }
    }
    
    // === matrices assembly
    {
      // compute system size
      (*assembly)(COMPUTE_SIZE);
      this->getContext()->executeVisitor( assembly.get() ); 
      
      // resize state vectors
      resize(assembly->sizeM, assembly->sizeC);
    
      // create local matrices J,K,M,C at each level
      (*assembly)(DO_SYSTEM_ASSEMBLY);
      this->getContext()->executeVisitor( assembly.get() );
      
      // global matrices assembly
      assembly->global(_matM, _matK, _matC, _matJ);

      // deal with projection matrices
      _matK = P() * _matK * P();  /// Filter the matrix. 
      
      // TODO this is not enough to guarantee that the projected DOFs
      // are isolated. M should be set diagonal.
      
      _PMinvP_isDirty = true;
    
      if( verbose.getValue() ) {
	cerr<<__METHOD__ << " assembly performed ==================================== " << endl ;
	cerr<<__METHOD__ << " assembled M = " << endl << DenseMatrix(_matM) << endl;
	cerr<<__METHOD__ << " assembled K = " << endl << DenseMatrix(_matK) << endl;
	cerr<<__METHOD__ << " assembled P = " << endl << DenseMatrix(P()) << endl;
	cerr<<__METHOD__ << " assembled C = " << endl << DenseMatrix(_matC) << endl;
	cerr<<__METHOD__ << " assembled J = " << endl << DenseMatrix(_matJ) << endl;
	cerr<<__METHOD__ << " assembled f = "   << _vecF.transpose() << endl;
	cerr<<__METHOD__ << " assembled phi = " << _vecPhi.transpose() << endl;
	cerr<<__METHOD__ << " assembled v = "   << _vecV.transpose() << endl;
      }
    }
     
    // ==== Compute the implicit matrix and right-hand term
    {
      scoped::timer step("implicit equation: scaling and sum of matrices, update right-hand term ");

      SReal rs = f_rayleighStiffness.getValue();
      SReal rm = f_rayleighMass.getValue();

      // complete the right-hand term b = f0 + (h+rs) K v - rm M v 
      // Rayleigh mass factor rm is used with a negative sign because it
      // is recorded as a positive real, while its force is opposed to
      // the velocity
      _vecF += _matK * _vecV * ( implicitVelocity.getValue() * (h + rs) );
      _vecF -= _matM * _vecV * rm;

      _vecF = P() * _vecF; // filter the right-hand side term
      
      // The implicit matrix is scaled by 1/h before solving the equation: 
      // M = ( (1+h*rm)M - h*B - h*(h+rs)K  )/h = (rm+1/h)M - (rs+h)K  
      // since we ignore B
      _matM *= rm + 1.0 / h;
      _matM -= _matK * (h + rs);
      
      if( verbose.getValue() ) {
	cerr << __METHOD__ << " implicit matrix = " << endl << _matM << endl
	     << __METHOD__ << " right-hand term = " << _vecF.transpose() << endl;
      }
    }
  }
    
  // ==== Solve equation system
  {
    scoped::timer step("Solve linear equation");
    solveEquation();
  }
    
  // ==== Apply the result
  {
    scoped::timer step("Apply result");
      
    (*assembly)(DISTRIBUTE_SOLUTION);
    this->getContext()->executeVisitor(assembly.get());  // set dv in each MechanicalState
    
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

    if( verbose.getValue() ) {
      mop.propagateX(nextPos);
      mop.propagateDx(nextVel);

      serr<<"ComplianceSolver, final x = "<< nextPos <<sendl;
      serr<<"ComplianceSolver, final v = "<< nextVel <<sendl;
      serr<<"-----------------------------"<<sendl;
    }
  }
}


void ComplianceSolver::MatrixAssemblyVisitor::global(SMatrix& M,
						     SMatrix& K,
						     SMatrix& C,
						     SMatrix& J) {
  // TODO assert sizes were computed first ?
    
  M.resize(sizeM, sizeM);
  K.resize(sizeM, sizeM);
  C.resize(sizeC, sizeC);
  J.resize(sizeC, sizeM);

  // ==== Global matrix assembly
  scoped::timer step("JMJt, JKJt, JCJt");
  for(State_2_LocalMatrices::iterator i = localMatrices.begin(), end = localMatrices.end(); 
      i != end; ++i) {
      
    core::behavior::BaseMechanicalState* s = (*i).first;
   
    if( localMatrices[s].M.rows() ){
      M += SMatrix( localMatrices[s].J.transpose() * localMatrices[s].M * localMatrices[s].J );
    }
    if( localMatrices[s].K.rows() ){
      K += SMatrix( localMatrices[s].J.transpose() * localMatrices[s].K * localMatrices[s].J );
    }
    if( localMatrices[s].C.rows() ){
      SMatrix C0 = createShiftMatrix( localMatrices[s].C.rows(), C.cols(), localMatrices[s].c_offset );
      C += SMatrix(C0.transpose() * localMatrices[s].C * C0);
      J += SMatrix( C0.transpose() * localMatrices[s].J ); // J vertically shifted, aligned with the compliance matrix
    }
  }
}


simulation::Visitor::Result ComplianceSolver::MatrixAssemblyVisitor::computeSize(simulation::Node* node) {

  // ==== independent DOFs
  if (node->mechanicalState && !node->mechanicalMapping ) {
    localMatrices[node->mechanicalState].m_offset = sizeM;
    sizeM += node->mechanicalState->getMatrixSize();
  }

  // ==== Compliances require a block in the global compliance matrix
  if( node->mechanicalState ) {

    for(unsigned i = 0; i < node->forceField.size(); ++i) {
      
      if(node->forceField[i]->getComplianceMatrix(mparams)) {
	localMatrices[node->mechanicalState].c_offset = sizeC;
	sizeC += node->mechanicalState->getMatrixSize();
      } else {   
	// stiffness does not contribute to matrix size, since the
	// stiffness matrix is added to the mass matrix of the state.
      }
    }
  }

  // ==== register all the DOFs
  if (node->mechanicalState ) {
    localDOFs.insert(node->mechanicalState);
  }
  
  return RESULT_CONTINUE;
}

simulation::Visitor::Result ComplianceSolver::MatrixAssemblyVisitor::doSystemAssembly(simulation::Node* node) {
  State_2_LocalMatrices& s2mjc = localMatrices;

  if( !node->mechanicalState ) return RESULT_CONTINUE;
  
  assert( s2mjc.find(node->mechanicalState) != s2mjc.end() );
  
  SMatrix& J0 = s2mjc[node->mechanicalState].J;
  SMatrix& localM  = s2mjc[node->mechanicalState].M;
  SMatrix& localC  = s2mjc[node->mechanicalState].C;
  SMatrix& localK  = s2mjc[node->mechanicalState].K;
  
  const unsigned& m_offset = s2mjc[node->mechanicalState].m_offset;
  const unsigned& c_offset = s2mjc[node->mechanicalState].c_offset;
  
  unsigned localSize = node->mechanicalState->getMatrixSize();
  
  // ==== independent DOFs
  if ( !node->mechanicalMapping ) {
    scoped::timer step("shift and project independent states");
    
    J0 = createShiftMatrix( localSize, sizeM, m_offset );

    // projections applied to the independent DOFs. The projection applied to mapped DOFs are ignored.
    for(unsigned i=0; i<node->projectiveConstraintSet.size(); i++){
      node->projectiveConstraintSet[i]->projectMatrix(&solver->_projMatrix,m_offset);
    }

    //  ==== rhs term (includes force applied to slave dofs, mapped upward)

    // use a copy, because the parameter is modified by addToBaseVector
    unsigned offset = m_offset; 
    
    // wrap as linearsolver::BaseVector for use as function parameter
    Wrap wrapF(solver->_vecF), wrapV(solver->_vecV);  
    
    node->mechanicalState->addToBaseVector( &wrapF ,core::VecDerivId::force(),offset);
    
    // use a copy, because the parameter is modified by addToBaseVector
    offset = m_offset; 
    
    node->mechanicalState->addToBaseVector(&wrapV, core::VecDerivId::velocity(), offset );
    
  } else {
    // process the mapping
    scoped::timer step("J products");
    
    const vector<sofa::defaulttype::BaseMatrix*>* pJs = node->mechanicalMapping->getJs();
    const vector<sofa::defaulttype::BaseMatrix*>* pKs = node->mechanicalMapping->getKs();
    
    vector<core::BaseState*> pStates = node->mechanicalMapping->getFrom();
    assert( pJs->size() == pStates.size());
    
    // Child state. Only N-to-1 mappings are handled yet
    MechanicalState* cstate = dynamic_cast<MechanicalState*>(  node->mechanicalMapping->getTo()[0] ); 
    
    for( unsigned i = 0; i < pStates.size(); ++i ) {

      MechanicalState* pstate = dynamic_cast<MechanicalState*>(pStates[i]);  // parent state
      assert(pstate);
      
      if( localDOFs.find(pstate) == localDOFs.end() ) {
	// skip states which are not in the scope of the solver, such
	// as mouse DOFs
	continue;
      }
        
      SMatrix Jcp = getSMatrix( (*pJs)[i] ); // child wrt parent
      SMatrix& Jp0 = s2mjc[pstate].J;        // parent wrt global DOF

      // contribute to the Jacobian matrix of the child wrt global DOF
      SMatrix& Jc0 = s2mjc[cstate].J;  // child wrt global DOF;

      if( !Jc0.rows() ) Jc0 = SMatrix(Jcp * Jp0);
      else Jc0 += SMatrix(Jcp * Jp0);
      
      // Geometric stiffness
      if( pKs && ((*pKs)[i]) ) {

	SMatrix K = getSMatrix( (*pKs)[i] ); // geometric stiffness related to this parent
	SMatrix& pK = s2mjc[pstate].K;       // parent stiffness

	if( !pK.rows() ) pK = K;
	else pK += K;
	
      }
      
    }
  }

  // ==== mass
  if (node->mass ){
    scoped::timer step("local M");
    
    // TODO: better way to fill the mass matrix
    typedef linearsolver::EigenBaseSparseMatrix<SReal> Sqmat;
    Sqmat sqmat(node->mechanicalState->getMatrixSize(),node->mechanicalState->getMatrixSize());
    linearsolver::SingleMatrixAccessor accessor( &sqmat );
    node->mass->addMToMatrix( mparams, &accessor );
    sqmat.compress();
    localM = sqmat.compressedMatrix;
  }
  
  // ==== compliance and stiffness
  for(unsigned i = 0; i < node->forceField.size(); ++i ) {
    BaseForceField* ffield = node->forceField[i];
    
    if( ffield->getComplianceMatrix(mparams) ) {
      scoped::timer step( "local C and right-hand term" );

      // compute scaling of C, based on time step, damping and
      // implicit coefficients
      
      SReal alpha = cparams.implicitVelocity(); // implicit velocity 
      SReal beta  = cparams.implicitPosition(); // implicit position

      SReal l = alpha * (beta * mparams->dt() + ffield->getDampingRatio() );
      
      // TODO HARDCODE
      if( std::abs(l) < 1.0e-10 ) {
	solver->serr << ffield->getName() << ", l is not invertible in "
		     << __METHOD__ << solver->sendl;
      }
      
      SReal invl = 1.0 / l;
      
      localC = getSMatrix(ffield->getComplianceMatrix(mparams)) *  invl;
      
      // Right-hand side term
      ffield->writeConstraintValue( &cparams, core::VecDerivId::force() );

      // use a copy, because the parameter is modified by addToBaseVector
      unsigned offset = c_offset; 
      
      Wrap wrapPhi(solver->_vecPhi); // wrap as linearsolver::BaseVector for use as function parameter
      node->mechanicalState->addToBaseVector(&wrapPhi, core::VecDerivId::force(), offset );
		
      // compliance entry point for derived classes
      onCompliance(ffield, c_offset, offset - c_offset);
		
    }
    else if (ffield->getStiffnessMatrix(mparams)) {
      // accumulate the stiffness if the matrix is not null. 
      // TODO: Rayleigh damping
         
      scoped::timer step("local K");
      
      // TODO conservativeResize here ?
      if( localK.rows() != (int) localSize ) {
	localK.resize(localSize, localSize);
      }
      localK += getSMatrix(ffield->getStiffnessMatrix(mparams));
    }
  }


  return RESULT_CONTINUE;
}


simulation::Visitor::Result ComplianceSolver::MatrixAssemblyVisitor::distributeSolution(simulation::Node* node) {
  
  // ==== independent DOFs
  if (node->mechanicalState  && !node->mechanicalMapping ) {
    // use a copy, because the parameter is modified by addToBaseVector
    unsigned offset = localMatrices[node->mechanicalState].m_offset; 

    // wrap as linearsolver::BaseVector for use as function parameter
    Wrap wrapF(solver->_vecF), wrapDv(solver->_vecDv); 
    node->mechanicalState->copyFromBaseVector(core::VecDerivId::force(), &wrapF, offset );
    
    // use a copy, because the parameter is modified by addToBaseVector
    offset = localMatrices[node->mechanicalState].m_offset; 
    node->mechanicalState->copyFromBaseVector(core::VecDerivId::dx(), &wrapDv, offset );
  }
  
  return RESULT_CONTINUE;
}

simulation::Visitor::Result ComplianceSolver::MatrixAssemblyVisitor::processNodeTopDown(simulation::Node* node)
{
  switch( pass ) {
  case COMPUTE_SIZE: return computeSize( node ); 
  case DO_SYSTEM_ASSEMBLY: return doSystemAssembly( node );
  case DISTRIBUTE_SOLUTION: return distributeSolution( node ); 
  }
  
  throw std::logic_error( "__FILE__:__LINE__: unknown pass" );
}


const ComplianceSolver::SMatrix& ComplianceSolver::PMinvP()
{
    if( _PMinvP_isDirty ) // update it
    {
      if( !inverseDiagonalMatrix( _PMinvP_Matrix.compressedMatrix, _matM ) ) {
            assert(false);
      }
        _PMinvP_Matrix.compressedMatrix = P().transpose() * _PMinvP_Matrix.compressedMatrix * P();
        _PMinvP_isDirty = false;
    }
    return _PMinvP_Matrix.compressedMatrix;
}



/// Return a rectangular matrix (cols>rows), with (offset-1) null
/// columns, then the (rows*rows) identity, then null columns.  This
/// is used to shift a "local" matrix to the global indices of an
/// assembly matrix.
ComplianceSolver::SMatrix ComplianceSolver::createShiftMatrix( unsigned rows, unsigned cols, unsigned offset )
{
    SMatrix m(rows,cols);
    for(unsigned i=0; i<rows; i++ ){
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
    for(unsigned i=0; i<size; i++ ){
        m.startVec(i);
        m.insertBack(i,i) =1;
    }
    m.finalize();
    return m;
}

ComplianceSolver::SMatrixC ComplianceSolver::createIdentityMatrixC( unsigned size )
{
    SMatrixC m(size,size);
    for(unsigned i=0; i<size; i++ ){
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
    return sm->compressedMatrix;
}


bool ComplianceSolver::inverseDiagonalMatrix( SMatrix& Minv, const SMatrix& M )
{
  if( (M.rows() != M.cols()) || (M.nonZeros() != M.rows()) ) // test if diagonal. WARNING: some non-diagonal matrix pass the test, but they are unlikely in this context.
        return false;
    Minv.resize(M.rows(),M.rows());
    for (int i=0; i<M.outerSize(); ++i){
        Minv.startVec(i);
        for (SMatrix::InnerIterator it(M,i); it; ++it)
        {
            assert(i==it.col() && "ComplianceSolver::inverseDiagonalMatrix needs a diagonal matrix");
	    assert( it.value() );
	    
            Minv.insertBack(i,i) = 1.0 / it.value();
        }
    }
    Minv.finalize();

    assert( Minv.rows() == M.rows() );
    assert( Minv.cols() == M.cols() );

    return true;
}

bool ComplianceSolver::inverseMatrix( SMatrix& , const SMatrix&  )
{
    cerr<<"ComplianceSolver::inverseMatrix is not yet implemented" << endl;
    return false;
//    Cholesky cholesky(M);
//    if( cholesky.info()!=Eigen::Success ) return false;
//    SMatrixC id = createIdentityMatrixC(M.rows());
//    SMatrixC MinvC = cholesky.solve( id ) ;
//    Minv = MinvC;
//    return true;
}




}
}
}
