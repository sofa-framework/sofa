cmake_minimum_required(VERSION 2.8)

project("SofaDefaultType")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

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

set(SOURCE_FILES

	BaseMatrix.cpp
	BoundingBox.cpp
	Frame.cpp
	#RigidInertia.cpp
	SolidTypes.cpp
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaHelper)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_DEFAULTTYPE")

include(${SOFA_CMAKE_DIR}/post.cmake)

