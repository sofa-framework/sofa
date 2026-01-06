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
#pragma once

#include <sofa/linearalgebra/CompressedRowSparseMatrixMechanical.h>
#include <sofa/simulation/task/ParallelForEach.h>
#include <sofa/simulation/task/TaskScheduler.h>

namespace multithreading::component::linearsolver::iterative
{
using sofa::linearalgebra::CRSMechanicalPolicy;


/**
 * Simplified equivalent of CompressedRowSparseMatrixMechanical where some
 * methods are multithreaded.
 */
template<typename TBlock, typename TPolicy = CRSMechanicalPolicy >
class ParallelCompressedRowSparseMatrixMechanical : public sofa::linearalgebra::BaseMatrix
{
    using Base = sofa::linearalgebra::CompressedRowSparseMatrixMechanical<TBlock, TPolicy>;

public:

    using Real = typename Base::Real;
    using Index = typename Base::Index;

    static constexpr sofa::Index NL = Base::NL;
    static constexpr sofa::Index NC = Base::NC;

    static const char* Name()
    {
        static std::string name = std::string("Parallel") + std::string(Base::Name());
        return name.c_str();
    }

    Index rowSize() const override
    {
        return m_crs.rowSize();
    }
    Index colSize() const override
    {
        return m_crs.colSize();
    }

    void clearRowCol(Index i) override
    {
        m_crs.clearRowCol(i);
    }

    void clearRow(Index i) override
    {
        m_crs.clearRow(i);
    }

    void clearCol(Index i) override
    {
        m_crs.clearCol(i);
    }

    void set(Index i, Index j, double v) override
    {
        m_crs.set(i, j, v);
    }

    Index rowBSize() const
    {
        return m_crs.rowBSize();
    }

    template<class Vec> static void vresize(Vec& vec, Index /*blockSize*/, Index totalSize) { vec.resize( totalSize ); }
    template<class Vec> static void vresize(sofa::type::vector<Vec>&vec, Index blockSize, Index /*totalSize*/) { vec.resize( blockSize ); }

    template<class Vec> static Real vget(const Vec& vec, Index i, Index j, Index k) { return vget( vec, i*j+k ); }
    template<class Vec> static Real vget(const sofa::type::vector<Vec>&vec, Index i, Index /*j*/, Index k) { return vec[i][k]; }

    static Real  vget(const sofa::linearalgebra::BaseVector& vec, Index i) { return static_cast<Real>(vec.element(i)); }
    template<class Real2> static Real2 vget(const sofa::linearalgebra::FullVector<Real2>& vec, Index i) { return vec[i]; }


    template<class Vec> static void vset(Vec& vec, Index i, Index j, Index k, Real v) { vset( vec, i*j+k, v ); }
    template<class Vec> static void vset(sofa::type::vector<Vec>&vec, Index i, Index /*j*/, Index k, Real v) { vec[i][k] = v; }

    static void vset(sofa::linearalgebra::BaseVector& vec, Index i, Real v) { vec.set(i, v); }
    template<class Real2> static void vset(sofa::linearalgebra::FullVector<Real2>& vec, Index i, Real2 v) { vec[i] = v; }

    template<class V1, class V2>
    void tmul(V1& res, const V2& vec) const
    {
        assert(vec.size() % bColSize() == 0); // vec.size() must be a multiple of block size.

        const_cast<Base*>(&m_crs)->compress();
        vresize(res, this->rowBSize(), this->rowSize());
        sofa::simulation::parallelForEachRange(*m_taskScheduler, static_cast<std::size_t>(0), m_crs.rowIndex.size(),
        [this, &vec, &res](const auto& range)
        {
            for (auto xi = range.start; xi < range.end; ++xi)
            {
                sofa::type::Vec<NL, Real> r;
                // local block-sized vector to accumulate the product of the block row  with the large vector

                // multiply the non-null blocks with the corresponding chunks of the large vector
                typename Base::Range rowRange(m_crs.rowBegin[xi], m_crs.rowBegin[xi + 1]);
                for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
                {
                    // transfer a chunk of large vector to a local block-sized vector
                    sofa::type::Vec<NC, Real> v;
                    //Index jN = colsIndex[xj] * NC;    // scalar column index
                    for (sofa::Index bj = 0; bj < NC; ++bj)
                    {
                        v[bj] = vget(vec, m_crs.colsIndex[xj], NC, bj);
                    }

                    // multiply the block with the local vector
                    const typename Base::Block& b = m_crs.colsValue[xj];
                    // non-null block has block-indices (rowIndex[xi],colsIndex[xj]) and value colsValue[xj]
                    for (sofa::Index bi = 0; bi < NL; ++bi)
                    {
                        for (sofa::Index bj = 0; bj < NC; ++bj)
                        {
                            r[bi] += Base::traits::v(b, bi, bj) * v[bj];
                        }
                    }
                }

                // transfer the local result  to the large result vector
                //Index iN = rowIndex[xi] * NL;                      // scalar row index
                for (sofa::Index bi = 0; bi < NL; ++bi)
                {
                    vset(res, m_crs.rowIndex[xi], NL, bi, r[bi]);
                }
            }
        });

    }

    template<class Vec>
    Vec operator*(const Vec& v) const
    {
        Vec res;
        tmul( res, v );
        return res;
    }

    SReal element(BaseMatrix::Index i, BaseMatrix::Index j) const override
    {
        return m_crs.element(i, j);
    }

    void resize(BaseMatrix::Index nbRow, BaseMatrix::Index nbCol) override
    {
        m_crs.resize(nbRow, nbCol);
    }

    void clear() override
    {
        m_crs.clear();
    }

    void add(BaseMatrix::Index row, BaseMatrix::Index col, double v) override
    {
        m_crs.add(row, col, v);
    }

    void add(Index row, Index col, const sofa::type::Mat3x3d & _M) override
    {
        m_crs.add(row, col, _M);
    }

    void add(Index row, Index col, const sofa::type::Mat3x3f & _M) override
    {
        m_crs.add(row, col, _M);
    }

    void add(Index row, Index col, const sofa::type::Mat2x2d & _M) override
    {
        m_crs.BaseMatrix::add(row, col, _M);
    }

    void add(Index row, Index col, const sofa::type::Mat2x2f & _M) override
    {
        m_crs.BaseMatrix::add(row, col, _M);
    }

    void setTaskScheduler(sofa::simulation::TaskScheduler* taskScheduler)
    {
        m_taskScheduler = taskScheduler;
    }

private:
    Base m_crs;
    sofa::simulation::TaskScheduler* m_taskScheduler { nullptr };
};

}
