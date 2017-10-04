#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/simulation/MechanicalVisitor.h>

#include "../utils/sparse.h"

#include <Compliant/config.h>

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/OwnershipSPtr.h>


namespace sofa {


// TODO move this into SofaKernel as it can be useful to any plugin.

// some helpers
template<class Matrix>
bool zero(const Matrix& m) {
    return !m.nonZeros();
}

template<class Matrix>
bool notempty(const Matrix* m) {
    return m && m->rows();
}

template<class Matrix>
bool empty(const Matrix& m) {
    return !m.rows();
}


/// test if present values are all zero
/// @warning not optimized for eigen sparse matrices
//template<class SparseMatrix>
//bool fillWithZeros(const SparseMatrix& m) {
//    for( unsigned i=0 ; i<m.data().size() ; ++i )
//    {
//        if( m.valuePtr()[i] != 0 ) return false;
//    }
//    return true;
//}


template<class LValue, class RValue>
static void add(LValue& lval, const RValue& rval) {
    if( empty(lval) ) {
        lval = rval;
    } else {
        // paranoia, i has it
        lval += rval;
    }
}

// hopefully avoids a temporary alloc/dealloc for product
template<class LValue, class LHS, class RHS>
static void add_prod(LValue& lval, const LHS& lhs, const RHS& rhs) {

    helper::ScopedAdvancedTimer advancedTimer("add_prod");

    if( empty(lval) ) {
        sparse::fast_prod(lval, lhs, rhs);
        // lval = lhs * rhs;
    } else {
        // paranoia, i has it
        sparse::fast_add_prod(lval, lhs, rhs);
        // lval = lval + lhs * rhs;
    }
}


template<class dofs_type>
static std::string pretty(dofs_type* dofs) {
    return dofs->getContext()->getName() + " / " + dofs->getName();
}


// right-shift matrix, size x (off + size) matrix: (0, id)
template<class mat>
static mat shift_right(unsigned off, unsigned size, unsigned total_cols, SReal value = 1.0 ) {
    mat res( size, total_cols);
    assert( total_cols >= (off + size) );

    res.reserve( size );

    for(unsigned i = 0; i < size; ++i) {
        res.startVec( i );
        res.insertBack(i, off + i) = value;
    }
    res.finalize();

    return res;
}

// left-shift matrix, (off + size) x size matrix: (0 ; id)
template<class mat>
static mat shift_left(unsigned off, unsigned size, unsigned total_rows, SReal value = 1.0 ) {
    mat res( total_rows, size);
    assert( total_rows >= (off + size) );

    res.reserve( size );

    for(unsigned i = 0; i < size; ++i) {
        res.startVec( off + i );
        res.insertBack(off + i, i) = value;
    }
    res.finalize();

    return res;
}

template<class Triplet, class mat>
static void add_shifted_right( std::vector<Triplet>& res, const mat& m, unsigned off, SReal factor = 1.0 )
{
    for( int k = 0, n = m.outerSize(); k < n; ++k ) {
        for( typename mat::InnerIterator it(m,k) ; it ; ++it ) {
            res.push_back( Triplet( off + it.row(), off + it.col(), it.value() * factor ) );
        }
    }
}

template<class mat>
static void add_shifted_right( mat& res, const mat& m, unsigned off, SReal factor = 1.0 )
{
    for( int k=0 ; k<m.outerSize() ; ++k )
        for( typename mat::InnerIterator it(m,k) ; it ; ++it )
        {
            res.coeffRef(off+it.row(), off+it.col()) += it.value()*factor;
        }
}

template<class densemat, class mat>
static void add_shifted_right( densemat& res, const mat& m, unsigned off, SReal factor = 1.0 )
{
    for( int k=0 ; k<m.outerSize() ; ++k )
        for( typename mat::InnerIterator it(m,k) ; it ; ++it )
        {
            res(off+it.row(), off+it.col()) += it.value()*factor;
        }
}


/// return the shifted matrix @m (so returns a larger matrix with nb col = @total_cols) where @m begin at the index @off (m is possibly multiplied by @factor)
template<class real>
static Eigen::SparseMatrix<real, Eigen::RowMajor> shifted_matrix( const Eigen::SparseMatrix<real, Eigen::RowMajor>& m, unsigned off, unsigned total_cols, real factor = 1.0 )
{
    // let's play with eigen black magic

#if 0

    Eigen::SparseMatrix<real, Eigen::RowMajor> res = m * factor; // weighted by factor
    res.conservativeResize( m.rows(), total_cols );
    for( unsigned i=0 ; i<res.data().size() ; ++i )
    {
        res.innerIndexPtr()[i] += off; // where the shifting occurs
    }

#else

    const_cast<Eigen::SparseMatrix<real, Eigen::RowMajor>&>(m).makeCompressed();
    Eigen::SparseMatrix<real, Eigen::RowMajor> res( m.rows(), total_cols );

    res.data() = m.data();
    for( unsigned i=0 ; i<res.data().size() ; ++i )
    {
        res.valuePtr()[i] *= factor; // weighted by factor
        res.innerIndexPtr()[i] += off; // where the shifting occurs
    }

    memcpy(res.outerIndexPtr(), m.outerIndexPtr(), (m.outerSize()+1)*sizeof(typename Eigen::SparseMatrix<real, Eigen::RowMajor>::Index));

    assert( !res.innerNonZeroPtr() ); // should be NULL because compressed // *res.innerNonZeroPtr() = *m.innerNonZeroPtr();

#endif

    return res;
}

template<class densemat, class mat>
static void convertDenseToSparse( mat& res, const densemat& m )
{
    for( int i=0 ; i<m.rows() ; ++i )
    {
        res.startVec( i );
        for( int j=0 ; j<m.cols() ; ++j )
            if( m(i,j) ) res.insertBack(i, j) = m(i,j);
    }
    res.finalize();
}




// TODO move this in is own file
namespace simulation
{


/// res += constraint forces (== lambda/dt), only for mechanical object linked to a compliance
class MechanicalAddComplianceForce : public MechanicalVisitor
{
    core::MultiVecDerivId res, lambdas;
    SReal invdt;


public:
    MechanicalAddComplianceForce(const sofa::core::MechanicalParams* mparams, core::MultiVecDerivId res, core::MultiVecDerivId lambdas, SReal dt )
        : MechanicalVisitor(mparams), res(res), lambdas(lambdas), invdt(1.0/dt)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    // reset lambda where there is no compliant FF
    // these reseted lambdas were previously propagated, but were not computed from the last solve
    virtual Result fwdMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
    {
        // a compliant FF must be alone, so if there is one, it is the first one of the list.
        const core::behavior::BaseForceField* ff = NULL;

        if( !node->forceField.empty() ) ff = *node->forceField.begin();
        else if( !node->interactionForceField.empty() ) ff = *node->interactionForceField.begin();

        if( !ff || !ff->isCompliance.getValue() )
        {
            const core::VecDerivId& lambdasid = lambdas.getId(mm);
            if( !lambdasid.isNull() ) // previously allocated
            {
                mm->resetForce(this->params, lambdasid);
            }
        }
        return RESULT_CONTINUE;
    }

    virtual Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
    {
        return fwdMechanicalState( node, mm );
    }

    // pop-up lamdas without modifying f
    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        map->applyJT( this->mparams, lambdas, lambdas );
    }

    // for all dofs, f += lambda / dt
    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
    {
        const core::VecDerivId& lambdasid = lambdas.getId(mm);
        if( !lambdasid.isNull() ) // previously allocated
        {
            const core::VecDerivId& resid = res.getId(mm);

            mm->vOp( this->params, resid, resid, lambdasid, invdt ); // f += lambda / dt
        }
    }

    virtual void bwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
    {
        bwdMechanicalState( node, mm );
    }


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const {return "MechanicalAddComplianceForce";}
    virtual std::string getInfos() const
    {
        std::string name=std::string("[")+res.getName()+","+lambdas.getName()+std::string("]");
        return name;
    }

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const
    {
        return true;
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
        addWriteVector(res);
        addWriteVector(lambdas);
    }
#endif
};




/// add/propagate constraint *forces* (lambdas/dt) toward independent dofs
class SOFA_Compliant_API propagate_constraint_force_visitor : public simulation::MechanicalVisitor {

    core::MultiVecDerivId force, lambda;
    SReal factor;
    bool clear, propagate;

public:

    propagate_constraint_force_visitor(const sofa::core::MechanicalParams* mparams,
                      const core::MultiVecDerivId& out,
                      const core::MultiVecDerivId& in,
                      SReal factor, /*depending on the system formulation, constraint forces are deduced from lagrange multipliers  f=lamba for acc/dv,  f=lambda/dt for vel */
                      bool clear, /*clear existing forces*/
                      bool propagate /*propagating toward independent dofs*/)
        : simulation::MechanicalVisitor(mparams)
        , force( out )
        , lambda( in )
        , factor( factor )
        , clear(clear)
        , propagate(propagate)
    {
        assert(!propagate || clear ); // existing forces must be cleared if propagating
    }


    Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* state)
    {
        // lambdas should only be present at compliance location

        if( !node->forceField.empty() && node->forceField[0]->isCompliance.getValue() ) // TODO handle interactionFF
            // compliance should be alone in the node
            state->vOp( mparams, force.getId(state), clear?core::ConstVecId::null():force.getId(state), lambda.getId(state), factor ); // constraint force = lambda / dt
        else if( propagate )
            state->resetForce(mparams, force.getId(state));

        return RESULT_CONTINUE;
    }

    Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* state)
    {
        // compliance cannot be present at independent dof level
        if( propagate )
            state->resetForce(mparams, force.getId(state));
        return RESULT_CONTINUE;
    }

    void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        if( propagate )
            map->applyJT(mparams, force, force);
    }

    void bwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
    {
        if( propagate )
            c->projectResponse( mparams, force );
    }


};


/// TOOO move this somewhere else
/// propagate lambdas in lambda vectors
class SOFA_Compliant_API propagate_lambdas_visitor : public simulation::MechanicalVisitor {

    core::MultiVecDerivId lambda;

public:

    propagate_lambdas_visitor(const core::MechanicalParams* mparams,
                      const core::MultiVecDerivId& lambda)
        : simulation::MechanicalVisitor(mparams)
        , lambda( lambda )
    {
    }

    Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* state)
    {
        if( node->forceField.empty() || !node->forceField[0]->isCompliance.getValue() )
        {
            const core::VecDerivId& id = lambda.getId(state);
            if( !id.isNull() )
                state->resetForce( mparams, id );
        }

        return RESULT_CONTINUE;
    }

    Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* state)
    {
        // compliance cannot be present at independent dof level
        const core::VecDerivId& id = lambda.getId(state);
        if( !id.isNull() )
            state->resetForce( mparams, id );
        return RESULT_CONTINUE;
    }

    void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
    {
        map->applyJT(mparams, lambda, lambda);
    }

    void bwdProjectiveConstraintSet(simulation::Node* /*node*/, core::behavior::BaseProjectiveConstraintSet* c)
    {
        c->projectResponse( mparams, lambda );
    }

};


} // namespace simulation


} // namespace sofa

