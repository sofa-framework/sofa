/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/helper/logging/Messaging.h>
#include <climits>

namespace sofa::linearalgebra
{

BaseMatrix::BaseMatrix() = default;

BaseMatrix::~BaseMatrix()
{}

void BaseMatrix::compress() {}

static inline void opVresize(BaseVector& vec, BaseVector::Index n) { vec.resize(n); }
template<class Real2> static inline void opVresize(Real2* vec, BaseVector::Index n) { for (const Real2* end=vec+n; vec != end; ++vec) *vec = (Real2)0; }
static inline SReal opVget(const BaseVector& vec, BaseVector::Index i) { return (SReal)vec.element(i); }
template<class Real2> static inline SReal opVget(const Real2* vec, BaseVector::Index i) { return (SReal)vec[i]; }

//this line was remove to supress a warning.
//static inline void opVset(BaseVector& vec, BaseVector::Index i, SReal v) { vec.set(i, v); }

template<class Real2> static inline void opVset(Real2* vec, BaseVector::Index i, SReal v) { vec[i] = (Real2)v; }
static inline void opVadd(BaseVector& vec, BaseVector::Index i, double v) { vec.add(i, (SReal)v); }
static inline void opVadd(BaseVector& vec, BaseVector::Index i, float v) { vec.add(i, (SReal)v); }
template<class Real2> static inline void opVadd(Real2* vec, BaseVector::Index i, double v) { vec[i] += (Real2)v; }
template<class Real2> static inline void opVadd(Real2* vec, BaseVector::Index i, float v) { vec[i] += (Real2)v; }

template <class Real, int NL, int NC, bool add, bool transpose, class M, class V1, class V2>
struct BaseMatrixLinearOpMV_BlockDiagonal
{
    typedef typename M::RowBlockConstIterator RowBlockConstIterator;
    typedef typename M::ColBlockConstIterator ColBlockConstIterator;
    typedef typename M::BlockConstAccessor BlockConstAccessor;
    typedef typename M::Index Index;
    typedef type::Mat<NL,NC,Real> BlockData;
    void operator()(const M* mat, V1& result, const V2& v)
    {
        const Index rowSize = mat->rowSize();
        const Index colSize = mat->colSize();
        BlockData buffer;

        if constexpr (!add)
        {
            if constexpr (transpose)
            {
                opVresize(result, colSize);
            }
            else
            {
                opVresize(result, rowSize);
            }
        }
        for (auto [rowIt, rowEnd] = mat->bRowsRange(); rowIt != rowEnd; ++rowIt)
        {
            auto [colBegin, colEnd] = rowIt.range();
            if (colBegin != colEnd) // diagonal block exists
            {
                BlockConstAccessor block = colBegin.bloc();
                const BlockData& bdata = *(const BlockData*)block.elements(buffer.ptr());
                const Index i = block.getRow() * NL;
                const Index j = block.getCol() * NC;
                if constexpr (!transpose)
                {
                    type::VecNoInit<NC,Real> vj;
                    for (int bj = 0; bj < NC; ++bj)
                        vj[bj] = (Real)opVget(v, j+bj);
                    type::Vec<NL,Real> resi;
                    for (int bi = 0; bi < NL; ++bi)
                        for (int bj = 0; bj < NC; ++bj)
                            resi[bi] += bdata(bi,bj) * vj[bj];
                    for (int bi = 0; bi < NL; ++bi)
                        opVadd(result, i+bi, resi[bi]);
                }
                else
                {
                    type::VecNoInit<NL,Real> vi;
                    for (int bi = 0; bi < NL; ++bi)
                        vi[bi] = (Real)opVget(v, i+bi);
                    type::Vec<NC,Real> resj;
                    for (int bi = 0; bi < NL; ++bi)
                        for (int bj = 0; bj < NC; ++bj)
                            resj[bj] += bdata(bi,bj) * vi[bi];
                    for (int bj = 0; bj < NC; ++bj)
                        opVadd(result, j+bj, resj[bj]);
                }
            }
        }
    }
};

template <sofa::Size L, sofa::Size C, class real>
void matrixAdd(BaseMatrix* self, const Index row, const Index col, const sofa::type::Mat<L, C, real>& M)
{
    for (unsigned int r = 0; r < L; ++r)
    {
        for (unsigned int c = 0; c < C; ++c)
        {
            self->add(row + r, col + c, M(r,c));
        }
    }
}

///Adding values from a 3x3d matrix this function may be overload to obtain better performances
void BaseMatrix::add(const Index row, const Index col, const type::Mat3x3d & _M)
{
    matrixAdd(this, row, col, _M);
}

///Adding values from a 3x3f matrix this function may be overload to obtain better performances
void BaseMatrix::add(const Index row, const Index col, const type::Mat3x3f & _M)
{
    matrixAdd(this, row, col, _M);
}

///Adding values from a 2x2d matrix this function may be overload to obtain better performances
void BaseMatrix::add(const Index row, const Index col, const type::Mat2x2d & _M)
{
    matrixAdd(this, row, col, _M);
}

///Adding values from a 2x2f matrix this function may be overload to obtain better performances
void BaseMatrix::add(const Index row, const Index col, const type::Mat2x2f & _M)
{
    matrixAdd(this, row, col, _M);
}

void BaseMatrix::add(const Index row, const Index col, const type::Mat6x6d& _M)
{
    matrixAdd(this, row, col, _M);
}

void BaseMatrix::add(const Index row, const Index col, const type::Mat6x6f& _M)
{
    matrixAdd(this, row, col, _M);
}


// specialication for 1x1 blocks
template <class Real, bool add, bool transpose, class M, class V1, class V2>
struct BaseMatrixLinearOpMV_BlockDiagonal<Real, 1, 1, add, transpose, M, V1, V2>
{
    typedef typename M::Index Index;
    void operator()(const M* mat, V1& result, const V2& v)
    {
        const Index rowSize = mat->rowSize();
        const Index colSize = mat->colSize();
        if constexpr (!add)
        {
            if constexpr (transpose)
            {
                opVresize(result, colSize);
            }
            else
            {
                opVresize(result, rowSize);
            }
        }
        const Index size = (rowSize < colSize) ? rowSize : colSize;
        for (Index i=0; i<size; ++i)
        {
            opVadd(result, i, opVget(v, i));
        }
    }
};


template <class Real, int NL, int NC, bool add, bool transpose, class M, class V1, class V2>
struct BaseMatrixLinearOpMV_BlockSparse
{
    typedef typename M::RowBlockConstIterator RowBlockConstIterator;
    typedef typename M::ColBlockConstIterator ColBlockConstIterator;
    typedef typename M::BlockConstAccessor BlockConstAccessor;
    typedef type::Mat<NL,NC,Real> BlockData;
    typedef typename M::Index Index;
    void operator()(const M* mat, V1& result, const V2& v)
    {
        const Index rowSize = mat->rowSize();
        const Index colSize = mat->colSize();
        BlockData buffer;
        type::Vec<NC,Real> vtmpj;
        type::Vec<NL,Real> vtmpi;
        if constexpr (!add)
        {
            opVresize(result, (transpose ? colSize : rowSize));
        }
        for (auto [rowIt, rowEnd] = mat->bRowsRange(); rowIt != rowEnd; ++rowIt)
        {
            const Index i = rowIt.row() * NL;
            if constexpr (!transpose)
            {
                for (int bi = 0; bi < NL; ++bi)
                    vtmpi[bi] = (Real)0;
            }
            else
            {
                for (int bi = 0; bi < NL; ++bi)
                    vtmpi[bi] = (Real)opVget(v, i+bi);
            }
            for (auto [colIt, colEnd] = rowIt.range(); colIt != colEnd; ++colIt)
            {
                BlockConstAccessor block = colIt.bloc();
                const BlockData& bdata = *(const BlockData*)block.elements(buffer.ptr());
                const Index j = block.getCol() * NC;
                if constexpr (!transpose)
                {
                    for (int bj = 0; bj < NC; ++bj)
                        vtmpj[bj] = (Real)opVget(v, j+bj);
                    for (int bi = 0; bi < NL; ++bi)
                        for (int bj = 0; bj < NC; ++bj)
                            vtmpi[bi] += bdata(bi,bj) * vtmpj[bj];
                }
                else
                {
                    for (int bj = 0; bj < NC; ++bj)
                        vtmpj[bj] = (Real)0;
                    for (int bi = 0; bi < NL; ++bi)
                        for (int bj = 0; bj < NC; ++bj)
                            vtmpj[bj] += bdata(bi,bj) * vtmpi[bi];
                    for (int bj = 0; bj < NC; ++bj)
                        opVadd(result, j+bj, vtmpj[bj]);
                }
            }
            if constexpr (!transpose)
            {
                for (int bi = 0; bi < NL; ++bi)
                    opVadd(result, i+bi, vtmpi[bi]);
            }
        }
    }
};

template<bool add, bool transpose>
class BaseMatrixLinearOpMV
{
public:
    typedef typename BaseMatrix::Index Index;
    template <class M, class V1, class V2>
    static inline void opFull(const M* mat, V1& result, const V2& v)
    {
        const Index rowSize = mat->rowSize();
        const Index colSize = mat->colSize();
        if constexpr (!add)
        {
            if constexpr (transpose)
            {
                opVresize(result, colSize);
            }
            else
            {
                opVresize(result, rowSize);
            }
        }
        if constexpr (!transpose)
        {
            for (Index i=0; i<rowSize; ++i)
            {
                double r = 0;
                for (Index j=0; j<colSize; ++j)
                {
                    r += mat->element(i,j) * opVget(v, j);
                }
                opVadd(result, i, r);
            }
        }
        else
        {
            for (Index i=0; i<rowSize; ++i)
            {
                const double val = opVget(v, i);
                for (Index j=0; j<colSize; ++j)
                {
                    opVadd(result, j, mat->element(i,j) * val);
                }
            }
        }
    }

    template <class M, class V1, class V2>
    static inline void opIdentity(const M* mat, V1& result, const V2& v)
    {
        const Index rowSize = mat->rowSize();
        const Index colSize = mat->colSize();
        if constexpr (!add)
        {
            if constexpr (transpose)
            {
                opVresize(result, colSize);
            }
            else
            {
                opVresize(result, rowSize);
            }
        }
        const Index size = (rowSize < colSize) ? rowSize : colSize;
        for (Index i=0; i<size; ++i)
        {
            opVadd(result, i, opVget(v, i));
        }
    }


    template <int NL, int NC, class M, class V1, class V2>
    static inline void opDiagonal(const M* mat, V1& result, const V2& v)
    {
        const Index rowSize = mat->rowSize();
        const Index colSize = mat->colSize();
        if constexpr (!add)
        {
            if (transpose)
            {
                opVresize(result, colSize);
            }
            else
            {
                opVresize(result, rowSize);
            }
        }
        const Index size = (rowSize < colSize) ? rowSize : colSize;
        for (Index i=0; i<size; ++i)
        {
            opVadd(result, i, mat->element(i,i) * opVget(v, i));
        }
    }

    template <class Real, class M, class V1, class V2>
    static inline void opDynamicRealDefault(const M* mat, V1& result, const V2& v, Index NL, Index NC, BaseMatrix::MatrixCategory /*category*/)
    {
        msg_warning("BaseMatrix") << "PERFORMANCE WARNING: multiplication by matric with block size " << NL << "x" << NC << " not optimized.";
        opFull(mat, result, v);
    }

    template <class Real, int NL, int NC, class M, class V1, class V2>
    static inline void opDynamicRealNLNC(const M* mat, V1& result, const V2& v, BaseMatrix::MatrixCategory category)
    {
        switch(category)
        {
        case BaseMatrix::MATRIX_DIAGONAL:
        {
            BaseMatrixLinearOpMV_BlockDiagonal<Real, NL, NC, add, transpose, M, V1, V2> op;
            op(mat, result, v);
            break;
        }
        default: // default to sparse
        {
            BaseMatrixLinearOpMV_BlockSparse<Real, NL, NC, add, transpose, M, V1, V2> op;
            op(mat, result, v);
            break;
        }
        }
    }

    template <class Real, int NL, class M, class V1, class V2>
    static inline void opDynamicRealNL(const M* mat, V1& result, const V2& v, Index NC, BaseMatrix::MatrixCategory category)
    {
        switch(NC)
        {
        case 1: opDynamicRealNLNC<Real, NL, 1, M, V1, V2>(mat, result, v, category); break;
        case 2: opDynamicRealNLNC<Real, NL, 2, M, V1, V2>(mat, result, v, category); break;
        case 3: opDynamicRealNLNC<Real, NL, 3, M, V1, V2>(mat, result, v, category); break;
        case 4: opDynamicRealNLNC<Real, NL, 4, M, V1, V2>(mat, result, v, category); break;
            //case 5: opDynamicRealNLNC<Real, NL, 5, M, V1, V2>(mat, result, v, category); break;
        case 6: opDynamicRealNLNC<Real, NL, 6, M, V1, V2>(mat, result, v, category); break;
        default: opDynamicRealDefault<Real, M, V1, V2>(mat, result, v, NL, NC, category); break;
        }
    }

    template <class Real, class M, class V1, class V2>
    static inline void opDynamicReal(const M* mat, V1& result, const V2& v, Index NL, Index NC, BaseMatrix::MatrixCategory category)
    {
        switch(NL)
        {
        case 1: opDynamicRealNL<Real, 1, M, V1, V2>(mat, result, v, NC, category); break;
        case 2: opDynamicRealNL<Real, 2, M, V1, V2>(mat, result, v, NC, category); break;
        case 3: opDynamicRealNL<Real, 3, M, V1, V2>(mat, result, v, NC, category); break;
        case 4: opDynamicRealNL<Real, 4, M, V1, V2>(mat, result, v, NC, category); break;
            //case 5: opDynamicRealNL<Real, 5, M, V1, V2>(mat, result, v, NC, category); break;
        case 6: opDynamicRealNL<Real, 6, M, V1, V2>(mat, result, v, NC, category); break;
        default: opDynamicRealDefault<Real, M, V1, V2>(mat, result, v, NL, NC, category); break;
        }
    }

    template <class M, class V1, class V2>
    static inline void opDynamic(const M* mat, V1& result, const V2& v)
    {
        const Index NL = mat->getBlockRows();
        const Index NC = mat->getBlockCols();
        const BaseMatrix::MatrixCategory category = mat->getCategory();
        const BaseMatrix::ElementType elementType = mat->getElementType();
        const std::size_t elementSize = mat->getElementSize();
        if (category == BaseMatrix::MATRIX_IDENTITY)
        {
            opIdentity(mat, result, v);
        }
        else if (category == BaseMatrix::MATRIX_FULL)
        {
            opFull(mat, result, v);
        }
        else if (elementType == BaseMatrix::ELEMENT_FLOAT && elementSize == sizeof(float))
        {
            opDynamicReal<float, M, V1, V2>(mat, result, v, NL, NC, category);
        }
        else // default to double
        {
            opDynamicReal<double, M, V1, V2>(mat, result, v, NL, NC, category);
        }
    }
};


class BaseMatrixLinearOpMulV : public BaseMatrixLinearOpMV<false, false> {};
class BaseMatrixLinearOpPMulV : public BaseMatrixLinearOpMV<true, false> {};
class BaseMatrixLinearOpMulTV : public BaseMatrixLinearOpMV<false, true> {};
class BaseMatrixLinearOpPMulTV : public BaseMatrixLinearOpMV<true, true> {};

/// Multiply the matrix by vector v and put the result in vector result
void BaseMatrix::opMulV(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const
{
    BaseMatrixLinearOpMulV::opDynamic(this, *result, *v);
}

/// Multiply the matrix by float vector v and put the result in vector result
void BaseMatrix::opMulV(float* result, const float* v) const
{
    BaseMatrixLinearOpMulV::opDynamic(this, result, v);
}

/// Multiply the matrix by double vector v and put the result in vector result
void BaseMatrix::opMulV(double* result, const double* v) const
{
    BaseMatrixLinearOpMulV::opDynamic(this, result, v);
}

/// Multiply the matrix by vector v and add the result in vector result
void BaseMatrix::opPMulV(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const
{
    BaseMatrixLinearOpPMulV::opDynamic(this, *result, *v);
}

/// Multiply the matrix by float vector v and add the result in vector result
void BaseMatrix::opPMulV(float* result, const float* v) const
{
    BaseMatrixLinearOpPMulV::opDynamic(this, result, v);
}

/// Multiply the matrix by double vector v and add the result in vector result
void BaseMatrix::opPMulV(double* result, const double* v) const
{
    BaseMatrixLinearOpPMulV::opDynamic(this, result, v);
}


/// Multiply the transposed matrix by vector v and put the result in vector result
void BaseMatrix::opMulTV(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const
{
    BaseMatrixLinearOpMulTV::opDynamic(this, *result, *v);
}

/// Multiply the transposed matrix by float vector v and put the result in vector result
void BaseMatrix::opMulTV(float* result, const float* v) const
{
    BaseMatrixLinearOpMulTV::opDynamic(this, result, v);
}

/// Multiply the transposed matrix by double vector v and put the result in vector result
void BaseMatrix::opMulTV(double* result, const double* v) const
{
    BaseMatrixLinearOpMulTV::opDynamic(this, result, v);
}

/// Multiply the transposed matrix by vector v and add the result in vector result
void BaseMatrix::opPMulTV(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const
{
    BaseMatrixLinearOpPMulTV::opDynamic(this, *result, *v);
}

/// Multiply the transposed matrix by float vector v and add the result in vector result
void BaseMatrix::opPMulTV(float* result, const float* v) const
{
    BaseMatrixLinearOpPMulTV::opDynamic(this, result, v);
}

/// Multiply the transposed matrix by double vector v and add the result in vector result
void BaseMatrix::opPMulTV(double* result, const double* v) const
{
    BaseMatrixLinearOpPMulTV::opDynamic(this, result, v);
}

/// Multiply the transposed matrix by matrix m and store the result in matrix result
void BaseMatrix::opMulTM(BaseMatrix * /*result*/,BaseMatrix * /*m*/) const
{
    msg_warning("BaseMatrix") <<"opMulTM not yet implemented";
}








template <class Real, int NL, int NC, bool transpose, class M1, class M2 >
struct BaseMatrixLinearOpAM_BlockSparse
{
    typedef typename M1::RowBlockConstIterator RowBlockConstIterator;
    typedef typename M1::ColBlockConstIterator ColBlockConstIterator;
    typedef typename M1::BlockConstAccessor BlockConstAccessor;
    typedef typename M1::Index Index;
    typedef type::Mat<NL,NC,Real> BlockData;

    void operator()(const M1* m1, M2* m2, double & fact)
    {
        BlockData buffer;

        for (auto [rowIt, rowEnd] = m1->bRowsRange(); rowIt != rowEnd; ++rowIt)
        {
            const Index i = rowIt.row() * NL;

            for (auto [colIt, colEnd] = rowIt.range(); colIt != colEnd; ++colIt)
            {

                BlockConstAccessor block = colIt.bloc();
                const BlockData& bdata = *(const BlockData*)block.elements(buffer.ptr());
                const Index j = block.getCol() * NC;

                if constexpr (!transpose)
                {
                    for (int bi = 0; bi < NL; ++bi)
                        for (int bj = 0; bj < NC; ++bj)
                            m2->add(i+bi,j+bj,bdata(bi,bj) * fact);
                }
                else
                {
                    for (int bi = 0; bi < NL; ++bi)
                        for (int bj = 0; bj < NC; ++bj)
                            m2->add(j+bj,i+bi,bdata(bi,bj) * fact);
                }
            }
        }
    }
};


template <class Real, int NL, int NC, bool transpose, class M1 , class M2 >
struct BaseMatrixLinearOpAMS_BlockSparse
{
    typedef typename M1::RowBlockConstIterator RowBlockConstIterator;
    typedef typename M1::ColBlockConstIterator ColBlockConstIterator;
    typedef typename M1::BlockConstAccessor BlockConstAccessor;
    typedef typename M1::Index Index;
    typedef type::Mat<NL,NC,Real> BlockData;

    void operator()(const M1* m1, M1* m2, double & fact)
    {
        BlockData buffer;

        for (auto [rowIt, rowEnd] = m1->bRowsRange(); rowIt != rowEnd; ++rowIt)
        {
            const Index i = rowIt.row() * NL;

            for (auto [colIt, colEnd] = rowIt.range(); colIt != colEnd; ++colIt)
            {
                BlockConstAccessor block = colIt.bloc();
                const BlockData& bdata = *(const BlockData*)block.elements(buffer.ptr());
                const Index j = block.getCol() * NC;
                if constexpr (!transpose)
                {
                    for (int bi = 0; bi < NL; ++bi)
                        for (int bj = 0; bj < NC; ++bj)
                            m2->add(i+bi,j+bj,bdata(bi,bj) * fact);
                }
                else
                {
                    for (int bi = 0; bi < NL; ++bi)
                        for (int bj = 0; bj < NC; ++bj)
                            m2->add(j+bj,i+bi,bdata(bi,bj) * fact);
                }
            }
        }
    }
};

template <class Real, bool transpose, class M1, class M2 >
struct BaseMatrixLinearOpAM1_BlockSparse
{
    typedef typename M1::RowBlockConstIterator RowBlockConstIterator;
    typedef typename M1::ColBlockConstIterator ColBlockConstIterator;
    typedef typename M1::BlockConstAccessor BlockConstAccessor;
    typedef typename M1::Index Index;
    typedef Real BlockData;

    void operator()(const M1* m1, M2* m2, double & fact)
    {
        BlockData buffer;

        for (auto [rowIt, rowEnd] = m1->bRowsRange(); rowIt != rowEnd; ++rowIt)
        {
            const Index i = rowIt.row();

            for (auto [colIt, colEnd] = rowIt.range(); colIt != colEnd; ++colIt)
            {
                BlockConstAccessor block = colIt.bloc();
                const BlockData& bdata = *(const BlockData*)block.elements(&buffer);
                const Index j = block.getCol();

                if constexpr (!transpose)
                {
                    m2->add(i,j,bdata * fact);
                }
                else
                {
                    m2->add(j,i,bdata * fact);
                }
            }
        }
    }
};

template< bool transpose>
class BaseMatrixLinearOpAM
{
public:
    typedef typename BaseMatrix::Index Index;
    template <class M1, class M2 >
    static inline void opFull(const M1 * m1, M2 * m2, double fact)
    {
        const Index rowSize = m1->rowSize();
        const Index colSize = m2->colSize();
        if constexpr (!transpose)
        {
            for (Index j=0; j<rowSize; ++j)
            {
                for (Index i=0; i<colSize; ++i)
                {
                    m2->add(j,i,m1->element(i,j)*fact);
                }
            }
        }
        else
        {
            for (Index j=0; j<rowSize; ++j)
            {
                for (Index i=0; i<colSize; ++i)
                {
                    m2->add(j,i,m1->element(j,i)*fact);
                }
            }
        }
    }

    template <class M1, class M2 >
    static inline void opIdentity(const M1 * m1, M2 * m2, double fact)
    {
        const Index colSize = m1->colSize();
        for (Index j=0; j<colSize; ++j)
        {
            m2->add(j,j,fact);
        }
    }


    template <int NL, int NC, class M1, class M2 >
    static inline void opDiagonal(const M1 * m1, M2 * m2, double fact)
    {
        const Index colSize = m1->colSize();
        for (Index j=0; j<colSize; ++j)
        {
            m2->add(j,j,m1->element(j,j) * fact);
        }
    }

    template <class Real, class M1, class M2 >
    static inline void opDynamicRealDefault(const M1 * m1, M2 * m2, double fact, Index NL, Index NC, BaseMatrix::MatrixCategory /*category*/)
    {
        dmsg_info("LCPcalc") << "PERFORMANCE WARNING: multiplication by matric with block size " << NL << "x" << NC << " not optimized." ;
        opFull(m1, m2, fact);
    }

    template <class Real, int NL, int NC, class M1, class M2>
    static inline void opDynamicRealNLNC(const M1 * m1, M2 * m2, double fact, BaseMatrix::MatrixCategory /* category */)
    {
        {
            const Index NL1 = m2->getBlockRows();
            const Index NC2 = m2->getBlockCols();
            if ((NL1==NL) && (NC==NC2))
            {
                BaseMatrixLinearOpAMS_BlockSparse<Real, NL, NC, transpose, M1 , M2 > op;
                op(m1, m2, fact);
            }
            else
            {
                BaseMatrixLinearOpAM_BlockSparse<Real, NL, NC, transpose, M1 , M2 > op;
                op(m1, m2, fact);
            }
        }
    }

    template <class Real, class M1, class M2>
    static inline void opDynamicReal1(const M1 * m1, M2 * m2, double fact, BaseMatrix::MatrixCategory /* category */)
    {
        {
            BaseMatrixLinearOpAM1_BlockSparse<Real , transpose, M1, M2 > op;
            op(m1, m2, fact);
        }
    }

    template <class Real, int NL, class M1, class M2 >
    static inline void opDynamicRealNL(const M1 * m1, M2 * m2, double fact, Index NC, BaseMatrix::MatrixCategory category)
    {
        switch(NC)
        {
        case 1: if (NL==1) opDynamicReal1<Real, M1, M2 >(m1, m2, fact, category);
            else opDynamicRealNLNC<Real, NL, 1, M1, M2 >(m1, m2, fact, category);
            break;
        case 2: opDynamicRealNLNC<Real, NL, 2, M1, M2 >(m1, m2, fact, category); break;
        case 3: opDynamicRealNLNC<Real, NL, 3, M1, M2 >(m1, m2, fact, category); break;
        case 4: opDynamicRealNLNC<Real, NL, 4, M1, M2 >(m1, m2, fact, category); break;
        case 5: opDynamicRealNLNC<Real, NL, 5, M1, M2 >(m1, m2, fact, category); break;
        case 6: opDynamicRealNLNC<Real, NL, 6, M1, M2 >(m1, m2, fact, category); break;
        default: opDynamicRealDefault<Real, M1, M2 >(m1, m2, fact, NL, NC, category); break;
        }
    }

    template <class Real, class M1, class M2 >
    static inline void opDynamicReal(const M1 * m1, M2 * m2, double fact, Index NL, Index NC, BaseMatrix::MatrixCategory category)
    {
        switch(NL)
        {
        case 2: opDynamicRealNL<Real, 2, M1, M2 >(m1, m2, fact, NC, category); break;
        case 3: opDynamicRealNL<Real, 3, M1, M2 >(m1, m2, fact, NC, category); break;
        case 4: opDynamicRealNL<Real, 4, M1, M2 >(m1, m2, fact, NC, category); break;
        case 5: opDynamicRealNL<Real, 5, M1, M2 >(m1, m2, fact, NC, category); break;
        case 6: opDynamicRealNL<Real, 6, M1, M2 >(m1, m2, fact, NC, category); break;
        default: opDynamicRealDefault<Real, M1, M2 >(m1, m2, fact, NL, NC, category); break;
        }
    }

    template <class M1, class M2 >
    static inline void opDynamic(const M1 * m1, M2 * m2, double fact)
    {
        const Index NL = m1->getBlockRows();
        const Index NC = m1->getBlockCols();
        const BaseMatrix::MatrixCategory category = m1->getCategory();
        const BaseMatrix::ElementType elementType = m1->getElementType();
        const std::size_t elementSize = m1->getElementSize();

        if (category == BaseMatrix::MATRIX_IDENTITY)
        {
            opIdentity(m1, m2, fact);
        }
        else if (category == BaseMatrix::MATRIX_FULL)
        {
            opFull(m1, m2, fact);
        }
        else if (elementType == BaseMatrix::ELEMENT_FLOAT && elementSize == sizeof(float))
        {
            opDynamicReal<float, M1, M2 >(m1, m2, fact, NL, NC, category);
        }
        else // default to double
        {
            opDynamicReal<double, M1, M2 >(m1, m2, fact, NL, NC, category);
        }
    }
};

class BaseMatrixLinearOpAddM : public BaseMatrixLinearOpAM< false> {};
class BaseMatrixLinearOpAddMT : public BaseMatrixLinearOpAM< true> {};

/// Multiply the matrix by vector v and put the result in vector result
void BaseMatrix::opAddM(linearalgebra::BaseMatrix* result,double fact) const
{
    BaseMatrixLinearOpAddM::opDynamic(this, result, fact);
}

/// Multiply the matrix by vector v and put the result in vector result
void BaseMatrix::opAddMT(linearalgebra::BaseMatrix* result,double fact) const
{
    BaseMatrixLinearOpAddMT::opDynamic(this, result, fact);
}


std::ostream& operator<<(std::ostream& out, const  sofa::linearalgebra::BaseMatrix& m )
{
    const Index nx = m.colSize();
    const Index ny = m.rowSize();
    out << "[";
    for (Index y=0; y<ny; ++y)
    {
        out << "\n[";
        for (Index x=0; x<nx; ++x)
        {
            out << " " << m.element(y,x);
        }
        out << " ]";
    }
    out << " ]";
    return out;
}

std::istream& operator>>( std::istream& in, sofa::linearalgebra::BaseMatrix& m )
{
    // The reading could be way simplier with an other format,
    // but I did not want to change the existing output.
    // Anyway, I guess there are better ways to perform the reading
    // but at least this one is working...

    std::vector<SReal> line;
    std::vector< std::vector<SReal> > lines;

//    unsigned l=0, c;

    in.ignore(INT_MAX, '['); // ignores all characters until it passes a [, start of the matrix

    while(true)
    {
        in.ignore(INT_MAX, '['); // ignores all characters until it passes a [, start of the line
//        c=0;

        SReal r;
        char car; in >> car;
        while( car!=']') // end of the line
        {
            in.seekg( -1, std::istream::cur ); // unread car
            in >> r;
            line.push_back(r);
//            ++c;
            in >> car;
        }

//        ++l;

        lines.push_back(line);
        line.clear();

        in >> car;
        if( car==']' ) break; // end of the matrix
        else in.seekg( -1, std::istream::cur ); // unread car

    }

    m.resize( (Index)lines.size(), (Index)lines[0].size() );

    for( size_t i=0; i<lines.size();++i)
    {
        assert( lines[i].size() == lines[0].size() ); // all line should have the same number of columns
        for( size_t j=0; j<lines[i].size();++j)
        {
            m.add( (Index)i, (Index)j, lines[i][j] );
        }
    }

    m.compress();


    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}


} // namespace sofa::linearalgebra
