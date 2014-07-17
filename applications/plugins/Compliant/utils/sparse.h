#ifndef SPARSE_H
#define SPARSE_H

#include <sofa/helper/pooledEigen/Sparse>

namespace sparse {


template<class U, class F>
void fill(Eigen::SparseMatrix<U, Eigen::RowMajor>& res, unsigned nnz,
          const F& f) {
	
	res.setZero();
	res.reserve( nnz );
	
	for(unsigned i = 0, m = res.rows(); i < m; ++i) {
		res.startVec( i );
		
		for( unsigned j = 0, n = res.cols(); j < n; ++j) {
			double value = f(i, j);
			if( value ) res.insertBack(i, j) = value;
		}
	}
	
	res.finalize();
}          


}


#endif
