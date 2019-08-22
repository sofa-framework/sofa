#ifndef COMPLIANT_SPARSE_H
#define COMPLIANT_SPARSE_H

#include <Eigen/Sparse>


// easily restore default behavior
#define SPARSE_USE_DEFAULT_PRODUCT 0

namespace sparse {

template<class U, class F>
void fill(Eigen::SparseMatrix<U, Eigen::RowMajor>& res, unsigned nnz,
          const F& f) {
	
	res.setZero();
	res.reserve( nnz );
	
	for(unsigned i = 0, m = res.rows(); i < m; ++i) {
		res.startVec( i );
		
		for( unsigned j = 0, n = res.cols(); j < n; ++j) {
            U value = f(i, j);
			if( value ) res.insertBack(i, j) = value;
		}
	}
	
	res.finalize();
}          


namespace impl {

// prototype for col-major res/lhs/rhs
template<typename Lhs, typename Rhs, typename ResultType>
static void fast_prod_impl(ResultType& res, const Lhs& lhs, const Rhs& rhs, bool accumulate = false)
{
    using namespace Eigen;
    using namespace internal;
    
    typedef typename remove_all<Lhs>::type::Scalar Scalar;
    typedef typename remove_all<Lhs>::type::Index Index;

    // make sure to call innerSize/outerSize since we fake the storage order.
    const Index inner = lhs.innerSize();
    const Index outer = rhs.outerSize();
    eigen_assert(lhs.outerSize() == rhs.innerSize());

    const Scalar flag = std::numeric_limits<Scalar>::max();
    
    Matrix<Scalar,Dynamic,1> values; values.setConstant(inner, flag);
    Matrix<Index,Dynamic,1>  indices(inner);

    ResultType copy;

    if( accumulate ) {
        // TODO is there a better way ?
        copy = res;
    }

    // estimate the number of non zero entries
    // given a rhs column containing Y non zeros, we assume that the respective Y columns
    // of the lhs differs in average of one non zeros, thus the number of non zeros for
    // the product of a rhs column with the lhs is X+Y where X is the average number of non zero
    // per column of the lhs.
    // Therefore, we have nnz(lhs*rhs) = nnz(lhs) + nnz(rhs)
    const Index estimated_nnz_prod = (accumulate ? res.nonZeros() : 0) +
        lhs.nonZeros() + rhs.nonZeros();
    
    res.setZero();
    res.reserve( Index(estimated_nnz_prod) );
    
    // we compute each column of the result, one after the other
    for (Index j = 0; j < outer; ++j){
        
        res.startVec(j);
        Index nnz = 0;

        if( accumulate ) {

            for( typename ResultType::InnerIterator copyIt(copy, j); copyIt; ++copyIt) {
                const Index i = copyIt.index();
                const Scalar x = copyIt.value();

                // we should always be first
                assert(values[i] == flag);
                if( values[i] != flag ) {
                    throw std::logic_error("derp");
                }
                
                values[i] = x;
                indices[nnz] = i;
                ++nnz;
            }
        }
        
        for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt){
            const Scalar y = rhsIt.value();
            const Index k = rhsIt.index();
            for (typename Lhs::InnerIterator lhsIt(lhs, k); lhsIt; ++lhsIt){
                const Index i = lhsIt.index();
                const Scalar x = lhsIt.value();

                if(values[i] == flag){
                    values[i] = x * y;
                    indices[nnz] = i;
                    ++nnz;
                } else {
                    values[i] += x * y;
                }
            }
        }

        if(nnz > 1) {
            std::sort(indices.data(), indices.data() + nnz);
        }
        
        for(Index k = 0; k < nnz; ++k){
            const Index i = indices[k];
            res.insertBackByOuterInner(j,i) = values[i];
            values[i] = flag;
        }
    }
    res.finalize();

}

template<int A, int B, int C> struct requires_equal;

template<int I> struct requires_equal<I, I, I> { };

template<class A> struct row_major_bit {
    static const int value = Eigen::internal::traits<A>::Flags & Eigen::RowMajorBit;
};

template<class A, class B, class C>
struct check_row_major_bit : requires_equal< row_major_bit<A>::value,
                                             row_major_bit<B>::value,
                                             row_major_bit<C>::value > {
    static const int value = row_major_bit<A>::value;
};

template<int RowMajorBit> struct row_bit;

template<> struct row_bit<0> {

    // everyone is col-major
    template<typename Lhs, typename Rhs, typename ResultType>
    static void fast_prod(ResultType& res, const Lhs& lhs, const Rhs& rhs, bool accumulate) {
        fast_prod_impl(res, lhs, rhs, accumulate);
    }
   
};

template<> struct row_bit<1> {

    // everyone is row-major
    template<typename Lhs, typename Rhs, typename ResultType>
    static void fast_prod(ResultType& res, const Lhs& lhs, const Rhs& rhs, bool accumulate) {
        fast_prod_impl(res, rhs, lhs, accumulate);
    }

};


}

template<typename Lhs, typename Rhs, typename ResultType>
static void fast_prod(ResultType& res, const Lhs& lhs, const Rhs& rhs, bool accumulate = false) {

#if SPARSE_USE_DEFAULT_PRODUCT
        if( accumulate ) {
            res += lhs * rhs;
        } else {
            res = lhs * rhs;
        }
#else
        typedef impl::row_bit< impl::check_row_major_bit<ResultType, Lhs, Rhs>::value > select;

        // TODO assert sizes ?
        if( !accumulate ) res.resize(lhs.rows(), rhs.cols());
        else {
            assert( res.rows() == lhs.rows() );
            assert( res.cols() == rhs.cols() );
        }
        
        select::fast_prod(res, lhs, rhs, accumulate);
#endif
}

// convenience
template<typename Lhs, typename Rhs, typename ResultType>
static void fast_add_prod(ResultType& res, const Lhs& lhs, const Rhs& rhs) {
    fast_prod(res, lhs, rhs, true);
}


}




#endif
