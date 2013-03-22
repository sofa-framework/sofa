cmake_minimum_required(VERSION 2.8)

project("SofaDefaultType")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(headers

	defaulttype.h	
	BaseMatrix.h	
	BaseVector.h	
	BoundingBox.h	
	DataTypeInfo.h	
	Frame.h	 
	LaparoscopicRigidTypes.h	
	MapMapSparseMatrix.h	
	Mat.h	
	Mat_solve_LU.h	
	Mat_solve_SVD.h	
	Mat_solve_Cholesky.h	
	Quat.h	
	Quat.inl	
	#RigidInertia.h	
	#RigidInertia.inl	
	RigidTypes.h	
	RigidVec6Types.h	
	SolidTypes.h	
	SolidTypes.inl	
	#SparseConstraintTypes.h	
	Vec.h	
	VecTypes.h	
	MatSym.h	
	Vec3Types.h
	)

set(sources

	BaseMatrix.cpp
	BoundingBox.cpp
	Frame.cpp
	#RigidInertia.cpp
	SolidTypes.cpp
	)

groupFiles(headers  ".+\\\\.(h|hpp|inl)$" ${SOFA_HEADER_FILE_FILTER_NAME})
groupFiles(sources  ".+\\\\.(c|cpp)$" ${SOFA_CPP_FILE_FILTER_NAME})

add_library(${PROJECT_NAME} SHARED ${headers} ${sources})
target_link_libraries(${PROJECT_NAME} SofaHelper)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_DEFAULTTYPE")

include(${SOFA_CMAKE_DIR}/post.cmake)

