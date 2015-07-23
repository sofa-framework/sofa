include(${SOFA_CMAKE_DIR}/preProject.cmake)

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
	TemplatesAliases.h
	Vec.h	
	VecTypes.h	
	MatSym.h	
	Vec3Types.h
	)

set(SOURCE_FILES

	defaulttype.cpp
	BaseMatrix.cpp
	BoundingBox.cpp
	Frame.cpp
	#RigidInertia.cpp
	SolidTypes.cpp
	TemplatesAliases.cpp
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_DEFAULTTYPE")
AddLinkerDependencies(SofaHelper)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

