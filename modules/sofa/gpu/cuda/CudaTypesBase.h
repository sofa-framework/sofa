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
#include <sofa/component/linearsolver/SparseMatrix.h>

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
//		  v.memsetHost();
        v.clear();
    }

    void set(int i, SReal val)
    {
        v[i] = (T) val;
    }

    void add(int i, SReal val)
    {
        v[i] += (T)val;
    }

    /// v += a*f
    template<typename Real2,typename Real3>
    void peq(const CudaBaseVector<Real2>& a, Real3 f)
    {
        CudaMatrixUtilsKernels<Real>::vector_vector_peq(v.size(),
                f,
                a.deviceRead(),
                this->deviceWrite());
    }

    void operator=(const CudaBaseVector<Real> & e)
    {
        v = e.v;
    }

    const void* deviceRead(int off=0) const
    {
        return v.deviceReadAt(off);
    }

    void * deviceWrite(int off=0)
    {
        return v.deviceWriteAt(off);
    }

    void invalidateDevice()
    {
        v.invalidateDevice();
    }

    const T* hostRead(int off=0) const
    {
        return v.hostReadAt(off);
    }

    T * hostWrite(int off=0)
    {
        return v.hostWriteAt(off);
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

    void recreate(int nbRow,int nbCol)
    {
        m.recreate(nbRow,nbCol);
    }

    void eq(CudaBaseMatrix & mat)
    {
        m = mat.m;
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
        mult(res,v);
        return res;
    }

    void mul(CudaBaseVector<Real>& v,CudaBaseVector<Real> & r)
    {
        CudaMatrixUtilsKernels<Real>::matrix_vector_product(rowSize(),
                colSize(),
                m.deviceRead(),
                m.getPitchDevice(),
                1.0,
                r.getCudaVector().deviceRead(),
                0.0,
                v.getCudaVector().deviceWrite());
    }

    void mulT(CudaBaseVector<Real>& v,CudaBaseVector<Real> & r)
    {
        CudaMatrixUtilsKernels<Real>::matrixtr_vector_product(rowSize(),
                colSize(),
                m.deviceRead(),
                m.getPitchDevice(),
                1.0,
                r.getCudaVector().deviceRead(),
                0.0,
                v.getCudaVector().deviceWrite());
    }

//        template<class Real2>
//        void mul(CudaBaseVector<Real2>& res,const CudaBaseVector<Real2>& b) const {
//            for (unsigned i=0;i<m.getSizeY();++i) {
//                Real r = 0;
//                for (unsigned j=0;j<m.getSizeX();++j) {
//                    r += m[i][j] * b[j];
//                }
//                res[i] = r;
//            }
//        }

//        template<class Real2>
//        void mulT(CudaBaseVector<Real2>& res,const CudaBaseVector<Real2>& b) const {
//            for (unsigned i=0;i<m.getSizeX();++i) {
//                Real r = 0;
//                for (unsigned j=0;j<m.getSizeY();++j) {
//                    r += m[j][i] * b[j];
//                }
//                res[i] = r;
//            }
//        }

    void operator= ( const CudaBaseMatrix<T>& mat )
    {
        m = mat.m;
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

template <class T>
class CudaSparseMatrix : public BaseMatrix
{
public :
    typedef T Real;

    void resize(int nbRow, int nbCol)
    {
        colsize = nbCol;
        rowsize = nbRow;
        colptr.resize(rowsize);
    }

    unsigned int rowSize() const
    {
        return rowsize;
    }

    unsigned int colSize() const
    {
        return colsize;
    }

    SReal element(int j, int i) const
    {
        for (int k=colptr[j]; k<colptr[j+1]; k++)
        {
            if (rowind[k]==i) return values[k];
        }
        return 0.0;
    }

    void clear()
    {
        nnz = 0;
        colsize = 0;
        rowsize = 0;
        colptr.clear();
        rowind.clear();
        values.clear();
    }

    void set(int /*j*/, int /*i*/, double /*v*/)
    {
        std::cerr << "ERROR method set in sofa::gpu::cuda::CudaSparseMatrix in not implemented" << std::endl;
    }

    void add(int /*j*/, int /*i*/, double /*v*/)
    {
        std::cerr << "ERROR method add in sofa::gpu::cuda::CudaSparseMatrix in not implemented" << std::endl;
    }

    void buildFromBaseMatrix(BaseMatrix * J)
    {
        clear();
        if (sofa::component::linearsolver::SparseMatrix<double> * j = dynamic_cast<sofa::component::linearsolver::SparseMatrix<double> *>(J))
        {
            buildFromSparseMatrix(j);
        }
        else if (sofa::component::linearsolver::SparseMatrix<float> * j = dynamic_cast<sofa::component::linearsolver::SparseMatrix<float> *>(J))
        {
            buildFromSparseMatrix(j);
        }
        else if (CudaSparseMatrix<double> * j = dynamic_cast<CudaSparseMatrix<double> *>(J))
        {
            buildFromCudaSparseMatrix(j);
        }
        else if (CudaSparseMatrix<float> * j = dynamic_cast<CudaSparseMatrix<float> *>(J))
        {
            buildFromCudaSparseMatrix(j);
        }
        else
        {
            buildFromDenseMatrix(J);
        }
    }

    CudaVector<int> & getColptr()
    {
        return colptr;
    }

    CudaVector<int> & getRowind()
    {
        return rowind;
    }

    CudaVector<Real> & getValues()
    {
        return values;
    }

    unsigned getNnz()
    {
        return nnz;
    }

    static const char* Name();

protected :
    unsigned colsize,rowsize,nnz;
    CudaVector<int> colptr;
    CudaVector<int> rowind;
    CudaVector<Real> values;

    template <class JMatrix>
    void buildFromCudaSparseMatrix(JMatrix * J)
    {
        // rebuild a CRS matrix from a CSR matrix
        colptr = J->getColptr();
        rowind = J->getRowind();

        values.resize(J->getValues().size());
        for (unsigned i=0; i<J->getValues().size(); i++) values[i] = J->getValues()[i];
        // this should be replaced by : values = J->getValues();
        // for that the operator= must be able to convert vector<float> <-> vector<double>

        nnz = J->getNnz();
        colsize = J->colSize();
        rowsize = J->rowSize();
    }

    template <class JMatrix>
    void buildFromSparseMatrix(JMatrix * J)
    {
        // rebuild a CRS matrix from a sparse matrix
        colptr.recreate(J->rowSize()+1);
        int * colptr_ptr = colptr.hostWrite();

        colptr_ptr[0] = 0;
        for (unsigned j=0; j<J->rowSize(); j++)
        {
            const typename JMatrix::Line & lJ = (*J)[j];
            colptr_ptr[j+1] = colptr_ptr[j] + lJ.size();
        }
        nnz = colptr_ptr[J->rowSize()];
        rowind.recreate(nnz);
        values.recreate(nnz);
        int * rowind_ptr = rowind.hostWrite();
        Real * values_ptr = values.hostWrite();

        int id  = 0;
        for (unsigned j=0; j<J->rowSize(); j++)
        {
            const typename JMatrix::Line & lJ = (*J)[j];
            for (typename JMatrix::LElementConstIterator i1 = lJ.begin(), i1end = lJ.end(); i1 != i1end; ++i1)
            {
                rowind_ptr[id] = i1->first;
                values_ptr[id] = i1->second;
                id++;
            }
        }
        colsize = J->colSize();
        rowsize = J->rowSize();
    }

    void buildFromDenseMatrix(BaseMatrix * J)
    {
        // rebuild a CRS matrix from a dense matrix
        colptr.recreate(J->rowSize()+1);
        int * colptr_ptr = colptr.hostWrite();

        colptr_ptr[0] = 0;
        for (unsigned j=0; j<J->rowSize(); j++)
        {
            for (unsigned i=0; i<J->rowSize(); i++)
            {
                if (J->element(j,i)!=0.0)
                {
                    colptr_ptr[j+1]++;
                }
            }
        }

        nnz = colptr_ptr[J->rowSize()];
        rowind.recreate(nnz);
        values.recreate(nnz);
        int * rowind_ptr = rowind.hostWrite();
        Real * values_ptr = values.hostWrite();

        int id  = 0;
        for (unsigned j=0; j<J->rowSize(); j++)
        {
            for (unsigned i=0; i<J->rowSize(); i++)
            {
                if (J->element(j,i)!=0.0)
                {
                    rowind_ptr[id] = i;
                    values_ptr[id] = J->element(j,i);
                    id++;
                }
            }
        }

        colsize = J->colSize();
        rowsize = J->rowSize();
    }
};

typedef CudaSparseMatrix<float> CudaSparseMatrixf;
typedef CudaSparseMatrix<double> CudaSparseMatrixd;

template<> inline const char* CudaSparseMatrixf::Name() { return "CudaSparseMatrixf"; }
template<> inline const char* CudaSparseMatrixd::Name() { return "CudaSparseMatrixd"; }


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
