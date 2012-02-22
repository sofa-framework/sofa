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
#ifndef SOFA_GPU_CUDA_CUDATYPES_BASE_H
#define SOFA_GPU_CUDA_CUDATYPES_BASE_H

#include "CudaTypes.h"
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/gpu/cuda/CudaMatrixUtils.h>

//#define DEBUG_BASE

namespace sofa
{

namespace gpu
{

namespace cuda
{

using namespace sofa::defaulttype;

template <class T>
class CudaBaseVector : public BaseVector
{
public :
    typedef T Real;

    CudaVector<T>& getCudaVector()
    {
        return v;
    }

    const CudaVector<T>& getCudaVector() const
    {
        return v;
    }

    T& operator[](int i)
    {
        return v[i];
    }

    const T& operator[](int i) const
    {
        return v[i];
    }

    void fastResize(int nbRow)
    {
        v.fastResize(nbRow);
    }

    void fastResize(int nbRow,int warp_size)
    {
        v.fastResize(nbRow,warp_size);
    }

    void resize(int nbRow)
    {
        v.resize(nbRow);
    }

    void resize(int nbRow,int warp_size)
    {
        v.resize(nbRow,warp_size);
    }

    unsigned int size() const
    {
        return v.size();
    }

    SReal element(int i) const
    {
        return v[i];
    }

    void clear()
    {
        //for (unsigned int i=0; i<size(); i++) v[i]=(T)(0.0);
        v.memsetHost();
    }

    void set(int i, SReal val)
    {
        v[i] = (T) val;
    }

    void add(int i, SReal val)
    {
        v[i] += (T)val;
    }

    void operator=(const CudaBaseVector<Real> & e)
    {
        v = e.v;
    }

    const void* deviceRead() const
    {
        return v.deviceRead();
    }

    void * deviceWrite()
    {
        return v.deviceWrite();
    }

    void invalidateDevice()
    {
        v.invalidateDevice();
    }

    const T* hostRead() const
    {
        return v.hostRead();
    }

    T * hostWrite()
    {
        return v.hostWrite();
    }

    static const char* Name(); /* {
			return "CudaBaseVector";
            }*/

    friend std::ostream& operator<< ( std::ostream& os, const CudaBaseVector<T> & vec )
    {
        os << vec.v;
        return os;
    }

private :
    CudaVector<T> v;
};

typedef CudaBaseVector<float> CudaBaseVectorf;
typedef CudaBaseVector<double> CudaBaseVectord;

template<> inline const char* CudaBaseVectorf::Name() { return "CudaBaseVectorf"; }
template<> inline const char* CudaBaseVectord::Name() { return "CudaBaseVectord"; }

template <class T>
class CudaBaseMatrix : public BaseMatrix
{
public :
    typedef T Real;

    CudaMatrix<T> & getCudaMatrix()
    {
        return m;
    }

    void resize(int nbRow, int nbCol)
    {
        m.resize(nbRow,nbCol);
    }

    void resize(int nbRow, int nbCol,int ws)
    {
        m.resize(nbRow,nbCol,ws);
    }

    void fastResize(int nbRow, int nbCol)
    {
        m.fastResize(nbRow,nbCol);
    }

    void fastResize(int nbRow, int nbCol,int ws)
    {
        m.fastResize(nbRow,nbCol,ws);
    }

    unsigned int rowSize() const
    {
        return m.getSizeY();
    }

    unsigned int colSize() const
    {
        return m.getSizeX();
    }

    SReal element(int j, int i) const
    {
        return m[j][i];
    }

    T* operator[] ( int i )
    {
        return m[i];
    }

    const T* operator[] ( int i ) const
    {
        return m[i];
    }

    void clear()
    {
// 			for (unsigned j=0; j<m.getSizeX(); j++) {
// 				for (unsigned i=0; i<m.getSizeY(); i++) {
// 				  m[j][i] = (T)(0.0);
// 				}
// 			}
        m.clear();
        //m.memsetHost();
    }

    void set(int j, int i, double v)
    {
#ifdef DEBUG_BASE
        if ((j>=rowSize()) || (i>=colSize()))
        {
            printf("forbidden acces %d %d\n",j,i);
            exit(1);
        }
#endif
        m[j][i] = (T)v;
    }

    void add(int j, int i, double v)
    {
#ifdef DEBUG_BASE
        if ((j>=rowSize()) || (i>=colSize()))
        {
            printf("forbidden acces %d %d\n",j,i);
            exit(1);
        }
#endif
        m[j][i] += (T)v;
    }

    static const char* Name();

    CudaBaseVector<Real> operator*(const CudaBaseVector<Real> & v) const
    {
        CudaBaseVector<Real> res;
        res.fastResize(rowSize());
        CudaMatrixUtilsKernels<Real>::matrix_vector_product(rowSize(),
                m.deviceRead(),
                m.getPitchDevice(),
                v.getCudaVector().deviceRead(),
                res.getCudaVector().deviceWrite());
        return res;
    }

    void mult(CudaBaseVector<Real>& v,CudaBaseVector<Real> & r)
    {
        CudaMatrixUtilsKernels<Real>::matrix_vector_product(rowSize(),
                m.deviceRead(),
                m.getPitchDevice(),
                r.getCudaVector().deviceRead(),
                v.getCudaVector().deviceWrite());
    }

    void invalidateDevices()
    {
        m.invalidateDevices();
    }

    void invalidatehost()
    {
        m.invalidatehost();
    }

    const void* deviceRead()
    {
        return m.deviceRead();
    }

    void * deviceWrite()
    {
        return m.deviceWrite();
    }

    int getPitchDevice()
    {
        return m.getPitchDevice();
    }

    int getPitchHost()
    {
        return m.getPitchHost();
    }

    friend std::ostream& operator<< ( std::ostream& os, const CudaBaseMatrix<T> & mat )
    {
        os << mat.m;
        return os;
    }

private :
    CudaMatrix<T> m;
};

typedef CudaBaseMatrix<float> CudaBaseMatrixf;
typedef CudaBaseMatrix<double> CudaBaseMatrixd;

template<> inline const char* CudaBaseMatrixf::Name() { return "CudaBaseMatrixf"; }
template<> inline const char* CudaBaseMatrixd::Name() { return "CudaBaseMatrixd"; }



template<class Real>
class CudaMatrixUtils
{
public :
    CudaMatrixUtils()
    {
        szCol = 0;
        szLin = 0;
    }

    template<class JMatrix>
    void computeJ(JMatrix& J)
    {
// 			printf("CudaCudaMatrixUtils::computeJ szJ=(%d,%d)\n",J.colSize(),J.rowSize());

        szCol = 0;
        szLin = 0;
// 			int align3 = 0;
// 			int curDof = -1;
// 			for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++) {
// 				printf("JLine = %d\n",jit1->first);
// 			}

        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
        {
// 				if (align3==3 || align3==0) {
// 					curDof = jit1->first/3;
// 					if (jit1->first % 3 != 0) {
// 					  printf("CudaCudaMatrixUtils::computeJ Error First constraint is not on a dof %d\n",jit1->first);
// 					  continue;
// 					}
// 				} else if (curDof != jit1->first/3) {
// 				  printf("Warning not the same dof %d %d\n",curDof,jit1->first/3);
// 				  for (;align3<3;align3++) {
// 				    printf("add align %d\n",align3);
// 				    szLin++;
// 				  }
//
// 				  if (jit1->first % 3 != 0) {
// 				    printf("CudaCudaMatrixUtils::computeJ Error new First constraint is not on a dof %d %d\n",jit1->first,szLin);
// 				    continue;
// 				  }
// 				  align3 = 0;
// 				  curDof = jit1->first/3;
// 				}

            int tmp = jit1->second.size();
            if (tmp>0)
            {
                if (szCol<tmp) szCol=tmp;
                szLin++;
            }
// 				align3++;
        }
        cudaJCol.clear();
        cudaJLin.clear();
        cudaJ.clear();


        cudaJ.fastResize(szLin,szCol);
        cudaJCol.fastResize(szLin,szCol);
        cudaJLin.resize(szLin);
        if ((szCol==0) || (szLin==0)) return;
        int lin = 0;
        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
        {
            if (jit1->second.size()>0)
            {
                int col = 0;
                Real * cudaJ_p = cudaJ[lin];
                int * cudaJCol_p = cudaJCol[lin];

                for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
                {
                    cudaJCol_p[col] = i1->first;
                    //Real val = i1->second;
                    //printf("set (%d,%d,%f)\n",col,lin,val);
                    cudaJ_p[col] = (Real)i1->second;
                    //cudaJ.set(col,lin,1.0);
                    col++;
                }

                for (; col < szCol; col++)
                {
                    //cudaJ_p[col] = 0.0;// pas de données
                    cudaJCol_p[col] = -1;// pas de données
                }
                cudaJLin[lin] = jit1->first;
                lin++;
            }
        }

// 			printf("%d %d %d\n",cudaJCol.getSizeX(),cudaJCol.getSizeY(),cudaJCol.getPitchHost());
// 			std::cout << "JCOL = " << std::endl;
// 			std::cout << cudaJCol << std::endl;
// 			std::cout << "JVal = " << std::endl;
// 			std::cout << cudaJ << std::endl;
    }

    template<class CMatrix>
    void computeJR(CMatrix & R)
    {
// 			printf("CudaCudaMatrixUtils::computeJR\n");
//
// 			printf("CudaCudaMatrixUtils R--------\n");
// 			for (int j=0; j<szLin;j++) {
// 			    std::cout << cudaJLin.element(j) << " : " ;
// 			    for (int i=0; i<szCol;i++) {
// 			      if (cudaJCol.element(i,j)!=-1) {
// 				printf("(%d,%.2f)\t",cudaJCol[j][i],cudaJ[j][i]);
// 			      }
// 			    }
// 			    std::cout<< std::endl;
// 			}


        CudaMatrixUtilsKernels<Real>::Cuda_Compute_JR( szCol,
                szLin,
                R.deviceRead(),
                cudaJ.deviceWrite(),
                cudaJ.getPitchDevice(),
                cudaJLin.deviceRead(),
                cudaJCol.deviceRead(),
                cudaJCol.getPitchDevice());

// 			printf("end CudaCudaMatrixUtils::computeJR\n");
//
// 			printf("CudaCudaMatrixUtils ComputeJR--------\n");
// 			for (int j=0; j<szLin;j++) {
// 			    std::cout << cudaJLin.element(j) << " : " ;
// 			    for (int i=0; i<szCol;i++) {
// 			      if (cudaJCol.element(i,j)!=-1) {
// 				printf("(%d,%.2f)\t",cudaJCol[j][i],cudaJ[j][i]);
// 			      }
// 			    }
// 			    std::cout<< std::endl;
// 			}

    }

    template<class CMatrix>
    void computeJMInvJt(CMatrix& cudaMinv,CMatrix &result, float fact, bool localW = false)
    {
//   			printf("CudaCudaMatrixUtils::computeJMInvJt %d CudaJ=(%d,%d) result=(%d,%d)\n",localW,cudaJ.colSize(),cudaJ.rowSize(),result.colSize(),result.rowSize());

        CudaMatrixUtilsKernels<Real>::Cuda_Compute_JMInvJt(result.colSize(),
                szCol,
                szLin,
                fact,
                cudaJ.deviceRead(),
                cudaJ.getPitchDevice(),
                (localW) ? NULL : cudaJLin.deviceRead(),
                cudaJCol.deviceRead(),
                cudaJCol.getPitchDevice(),
                cudaMinv.deviceRead(),
                cudaMinv.getPitchDevice(),
                result.deviceWrite(),
                result.getPitchDevice());

        /*			printf("computeJMInvJt----------------------\n");
                    for (unsigned j=0;j<result.rowSize();j++) {
                      for (unsigned i=0;i<result.colSize();i++) {
                        printf("%f ",result.element(i,j));
                      }
                      printf("\n");
                    }	*/
    }

    template<class CMatrix,class CVector>
    void computeJMInvJt_twoStep(CMatrix& cudaMinv,CVector & idActiveDofs,CVector & invActiveDofs, CMatrix &result, float fact, bool localW = false)
    {
        cudaJMinv.clear();
        //cudaJMinv.resize(szLin,cudaMinv.colSize());
        cudaJMinv.fastResize(szLin,idActiveDofs.size());

        CudaMatrixUtilsKernels<Real>::Cuda_Compute_twoStep_JMInvJt_1(cudaMinv.colSize(),
                idActiveDofs.size(),
                szCol,
                szLin,
                idActiveDofs.deviceRead(),
                cudaJ.deviceRead(),
                cudaJ.getPitchDevice(),
                cudaJCol.deviceRead(),
                cudaJCol.getPitchDevice(),
                cudaMinv.deviceRead(),
                cudaMinv.getPitchDevice(),
                cudaJMinv.deviceWrite(),
                cudaJMinv.getPitchDevice());

        CudaMatrixUtilsKernels<Real>::Cuda_Compute_twoStep_JMInvJt_2(szCol,
                szLin,
                fact,
                invActiveDofs.deviceRead(),
                cudaJ.deviceRead(),
                cudaJ.getPitchDevice(),
                (localW) ? NULL : cudaJLin.deviceRead(),
                cudaJCol.deviceRead(),
                cudaJCol.getPitchDevice(),
                cudaJMinv.deviceRead(),
                cudaJMinv.getPitchDevice(),
                result.deviceWrite(),
                result.getPitchDevice());
    }



    void addLocalW(CudaBaseMatrix<Real> & localW, defaulttype::BaseMatrix &result)
    {
        const int* jl = cudaJLin.hostRead();
        for (unsigned j=0; j<localW.rowSize(); j++)
        {
            const Real * wline = localW[j];
            int Wj = jl[j];
            for (unsigned i=0; i<localW.colSize(); i++)
            {
                int Wi = jl[i];
                result.add(Wi,Wj, wline[i]);
            }
        }
    }

    int colSize()
    {
        return szCol;
    }

    int rowSize()
    {
        return szLin;
    }

protected :
    CudaMatrix<Real> cudaJ;
    CudaMatrix<int> cudaJCol;
    CudaVector<int> cudaJLin;
    CudaMatrix<Real> cudaJMinv;

    int szCol;
    int szLin;
};

} // namespace cuda
} // namespace gpu
} // namespace sofa

#endif
