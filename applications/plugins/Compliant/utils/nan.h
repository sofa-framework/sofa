#ifndef COMPLIANT_UTILS_NAN_H
#define COMPLIANT_UTILS_NAN_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cmath>

template<class U, int M, int N>
bool has_nan(const Eigen::Matrix<U, M, N>& x) {
	for(unsigned i = 0, m = x.rows(); i < m; ++i) {
		for(unsigned j = 0, n = x.cols(); j < n; ++j) {
			U xij = x(i, j);
            if( xij != xij )
                return true;
		}
	}
	return false;
}

template<typename _Scalar, int _Options, typename _Index>
bool has_nan(const Eigen::SparseMatrix<_Scalar,_Options,_Index>& x)
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
