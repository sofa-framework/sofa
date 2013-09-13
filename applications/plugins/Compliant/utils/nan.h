#ifndef COMPLIANT_UTILS_NAN_H
#define COMPLIANT_UTILS_NAN_H

#include <Eigen/Core>
#include <cmath>

template<class U, int M, int N>
bool has_nan(const Eigen::Matrix<U, M, N>& x) {
	for(unsigned i = 0, m = x.rows(); i < m; ++i) {
		for(unsigned j = 0, n = x.cols(); j < n; ++j) {
			U xij = x(i, j);

			if( xij != xij ) return true;
		}
	}
	return false;
}

#endif
