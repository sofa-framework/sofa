#ifndef UTILS_TYPES_H
#define UTILS_TYPES_H

#include <sofa/helper/system/config.h>

#include <Eigen/SparseCore>
#include <Eigen/Core>

namespace utils {

struct eigen_types {
    
    typedef SReal real;
	typedef Eigen::Matrix<real, Eigen::Dynamic, 1> vec;
	
	typedef Eigen::SparseMatrix<real, Eigen::RowMajor> rmat;
	typedef Eigen::SparseMatrix<real, Eigen::ColMajor> cmat;
    
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dmat;

    // TODO more as needed
};

}


#endif
