load(sofa/pre)

TEMPLATE = subdirs
HEADERS = 														\
	$$files(Eigen/src/Cholesky/*.h)						\
	$$files(Eigen/src/Core/*.h)								\
	$$files(Eigen/src/Core/arch/*.h)					\
	$$files(Eigen/src/Core/arch/AltiVec/*.h)	\
	$$files(Eigen/src/Core/arch/Default/*.h)	\
	$$files(Eigen/src/Core/arch/NEON/*.h)			\
	$$files(Eigen/src/Core/arch/SSE/*.h)			\
	$$files(Eigen/src/Core/products/*.h)			\
	$$files(Eigen/src/Core/util/*.h)					\
	$$files(Eigen/src/Eigen2Support/*.h)			\
	$$files(Eigen/src/Eigenvalues/*.h)				\
	$$files(Eigen/src/Geometry/*.h)						\
	$$files(Eigen/src/Geometry/arch/*.h)			\
	$$files(Eigen/src/Householder/*.h)				\
	$$files(Eigen/src/Jacobi/*.h)							\
	$$files(Eigen/src/LU/*.h)									\
	$$files(Eigen/src/LU/arch/*.h)						\
	$$files(Eigen/src/misc/*.h)								\
	$$files(Eigen/src/plugins/*.h)						\
	$$files(Eigen/src/QR/*.h)									\
	$$files(Eigen/src/Sparse/*.h)							\
	$$files(Eigen/src/StlSupport/*.h)					\
	$$files(Eigen/src/SVD/*.h)	\
	$$files(unsupported/Eigen/src/AutoDiff/*.h)\	
	$$files(unsupported/Eigen/src/BVH/*.h)	\
	$$files(unsupported/Eigen/src/FFT/*.h)	\
	$$files(unsupported/Eigen/src/IterativeSolvers/*.h)	\
	$$files(unsupported/Eigen/src/MatrixFunctions/*.h)	\
	$$files(unsupported/Eigen/src/MoreVectorization/*.h)	\
	$$files(unsupported/Eigen/src/NonLinearOptimization/*.h)\	
	$$files(unsupported/Eigen/src/NumericalDiff/*.h)	\
	$$files(unsupported/Eigen/src/Polynomials/*.h)	\
	$$files(unsupported/Eigen/src/Skyline/*.h)	\
	$$files(unsupported/Eigen/src/SparseExtra/*.h)	

load(sofa/post)
