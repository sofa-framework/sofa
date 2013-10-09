#include <sofa/component/linearsolver/EigenSparseMatrix.h>

namespace sofa {




// some helpers
template<class Matrix>
bool zero(const Matrix& m) {
    return !m.nonZeros();
}

template<class Matrix>
bool empty(const Matrix& m) {
    return !m.rows();
}


template<class LValue, class RValue>
static void add(LValue& lval, const RValue& rval) {
    if( empty(lval) ) {
        lval = rval;
    } else {
        // paranoia, i has it
        lval += rval;
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



template<class Triplet, class mat>
static void add_shifted_right( std::vector<Triplet>& res, const mat& m, unsigned off, SReal factor = 1.0 )
{
    for( int k=0 ; k<m.outerSize() ; ++k )
        for( typename mat::InnerIterator it(m,k) ; it ; ++it )
        {
            res.push_back( Triplet( off+it.row(), off+it.col(), it.value()*factor ) );
        }
}


// convert a basematrix to a sparse matrix. TODO move this somewhere else ?
template<class mat>
mat convert( const defaulttype::BaseMatrix* m) {
    assert( m );

    typedef component::linearsolver::EigenBaseSparseMatrix<double> matrixd;

    const matrixd* smd = dynamic_cast<const matrixd*> (m);
    if ( smd ) return smd->compressedMatrix.cast<SReal>();

    typedef component::linearsolver::EigenBaseSparseMatrix<float> matrixf;

    const matrixf* smf = dynamic_cast<const matrixf*>(m);
    if( smf ) return smf->compressedMatrix.cast<SReal>();


    std::cerr << "warning: slow matrix conversion (AssemblyHelper)" << std::endl;

    mat res(m->rowSize(), m->colSize());

    res.reserve(res.rows() * res.cols());
    for(unsigned i = 0, n = res.rows(); i < n; ++i) {
        res.startVec( i );
        for(unsigned j = 0, k = res.cols(); j < k; ++j) {
            res.insertBack(i, j) = m->element(i, j);
        }
    }

    return res;
}






}

