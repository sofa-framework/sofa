#ifndef COMPLIANT_UTILS_NAN_H
#define COMPLIANT_UTILS_NAN_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cmath>

template<class Derived>
static bool has_nan(const Eigen::MatrixBase<Derived>& x) {
	for(unsigned i = 0, m = x.rows(); i < m; ++i) {
		for(unsigned j = 0, n = x.cols(); j < n; ++j) {
			typename Derived::Scalar xij = x(i, j);
            if( xij != xij )
                return true;
		}
	}
	return false;
}

template<typename _Scalar, int _Options, typename _Index>
static bool has_nan(const Eigen::SparseMatrix<_Scalar,_Options,_Index>& x)
{
    typedef Eigen::SparseMatrix<_Scalar,_Options,_Index> Mat;

    for( unsigned i = 0 ; i < x.outerSize() ; ++i )
    {
        for( typename Mat::InnerIterator it(x, i) ; it ; ++it )
        {
            _Scalar xij = it.value();
            if( xij != xij )
                return true;
        }
    }
    return false;
}

#endif
