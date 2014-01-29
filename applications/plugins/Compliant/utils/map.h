#ifndef SOFA_MAP_H
#define SOFA_MAP_H

#include <sofa/defaulttype/Vec.h>
#include <Eigen/Core>

namespace utils {

/** Utility functions to map arrays to Eigen matrices back and forth. */

template<int M, class U>
static inline Eigen::Map< Eigen::Matrix<U, M, 1> > map(sofa::defaulttype::Vec<M, U>& v) {
	return Eigen::Map< Eigen::Matrix<U, M, 1> >(&v[0]);
}

template<int M, class U>
static inline Eigen::Map< Eigen::Matrix<U, M, 1> > map(U* v) {
	return Eigen::Map< Eigen::Matrix<U, M, 1> >(v);
}

template<int M, class U>
static inline Eigen::Map< const Eigen::Matrix<U, M, 1> > map(const sofa::defaulttype::Vec<M, U>& v) {
	return Eigen::Map< const Eigen::Matrix<U, M, 1> >(&v[0]);
}

template<int M, class U>
static inline Eigen::Map< const Eigen::Matrix<U, M, 1> > map(const U* v) {
	return Eigen::Map< const Eigen::Matrix<U, M, 1> >(v);
}


template<class U>
static inline Eigen::Map< const Eigen::Matrix<U, Eigen::Dynamic, 1> > map(const U* v, unsigned n) {
	return Eigen::Map< const Eigen::Matrix<U, Eigen::Dynamic, 1> >(v, n);
}

template<class U>
static inline Eigen::Map< Eigen::Matrix<U, Eigen::Dynamic, 1> > map(U* v, unsigned n) {
	return Eigen::Map< Eigen::Matrix<U, Eigen::Dynamic, 1> >(v, n);
}



}

#endif
