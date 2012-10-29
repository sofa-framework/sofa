/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDATRIDIAGONALMATRIXUTILS_H
#define SOFA_GPU_CUDA_CUDATRIDIAGONALMATRIXUTILS_H

//#include "CudaBaseMatrix.h"

namespace sofa {

namespace gpu {

namespace cuda {

extern "C"
{
	void trimatrix_vector_productf(int dim,const void * R,const void * r,void * z);
#ifdef SOFA_GPU_CUDA_DOUBLE
	void trimatrix_vector_productd(int dim,const void * R,const void * r,void * z);
#endif

	void trimatrixtr_vector_productf(int dim,const void * R,const void * r,void * z);
#ifdef SOFA_GPU_CUDA_DOUBLE
	void trimatrixtr_vector_productd(int dim,const void * R,const void * r,void * z);
#endif

    void trimatrix_trimatrixtr_productf(int dim,const void * R1,const void * R2,void * Rout);
#ifdef SOFA_GPU_CUDA_DOUBLE
    void trimatrix_trimatrixtr_productd(int dim,const void * R1,const void * R2,void * Rout);
#endif
}

template<class real> class CudaTridiagonalMatrixUtilsKernels;

template<> class CudaTridiagonalMatrixUtilsKernels<float>
{
public:
     // z = R * r
     // R is a block 3*3 matrix
     static inline void trimatrix_vector_product(int dim,const void * R,const void * r,void * z)
     {   trimatrix_vector_productf(dim,R,r,z); }

     // z = tr(R) * r
     // R is a block 3*3 matrix
     static inline void trimatrixtr_vector_product(int dim,const void * R,const void * r,void * z)
     {   trimatrixtr_vector_productf(dim,R,r,z); }

     // Rout = R1 * tr(R2)
     // Rinv, Rcur, Rout are block 3*3 matrix
     static inline void trimatrix_trimatrixtr_product(int dim,const void * R1,const void * R2,void * Rout)
     {   trimatrix_trimatrixtr_productf(dim,R1,R2,Rout); }

};

#ifdef SOFA_GPU_CUDA_DOUBLE
template<> class CudaTridiagonalMatrixUtilsKernels<double>
{
public:
     // z = R * r
     // R is a block 3*3 matrix
     static inline void trimatrix_vector_product(int dim,const void * R,const void * r,void * z)
     {   trimatrix_vector_productd(dim,R,r,z); }

     // z = tr(R) * r
     // R is a block 3*3 matrix
     static inline void trimatrixtr_vector_product(int dim,const void * R,const void * r,void * z)
     {   trimatrixtr_vector_productd(dim,R,r,z); }

     // Rout = R1 * tr(R2)
     // Rinv, Rcur, Rout are block 3*3 matrix
     static inline void trimatrix_trimatrixtr_product(int dim,const void * R1,const void * R2,void * Rout)
     {   trimatrix_trimatrixtr_productd(dim,R1,R2,Rout); }
};
#endif


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
