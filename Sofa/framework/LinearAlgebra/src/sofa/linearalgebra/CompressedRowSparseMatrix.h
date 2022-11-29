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

#include <sofa/linearalgebra/config.h>

#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/linearalgebra/matrix_bloc_traits.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixTraceLogger.h>
#include <algorithm>

namespace // anonymous
{
    // Boiler-plate code to test if a type implements a method
    // explanation https://stackoverflow.com/a/30848101

    template <typename...>
    using void_t = void;

    // Primary template handles all types not supporting the operation.
    template <typename, template <typename> class, typename = void_t<>>
    struct detectMatrix : std::false_type {};

    // Specialization recognizes/validates only types supporting the archetype.
    template <typename T, template <typename> class Op>
    struct detectMatrix<T, Op, void_t<Op<T>>> : std::true_type {};

    // Actual test if T implements transposed() (hence is a type::Mat)
    template <typename T>
    using isMatrix_t = decltype(std::declval<T>().transposed());

    template <typename T>
    using isMatrix = detectMatrix<T, isMatrix_t>;
} // anonymous

namespace sofa::linearalgebra
{

#ifdef SOFA_CRS_POLICY_CHECK
#define SOFA_CRS_POLICY_CHECK_FLAG true
#else
#define SOFA_CRS_POLICY_CHECK_FLAG false
#endif

#ifdef SOFA_CRS_POLICY_VERBOSE
#define SOFA_CRS_POLICY_VERBOSE_FLAG true
#else
#define SOFA_CRS_POLICY_VERBOSE_FLAG false
#endif

#ifdef SOFA_CRS_POLICY_LOGTRACE
#define SOFA_CRS_POLICY_LOGTRACE_FLAG true
#else
#define SOFA_CRS_POLICY_LOGTRACE_FLAG false
#endif


#ifdef SOFA_CRS_POLICY_PRINTTRACE
#define SOFA_CRS_POLICY_PRINTTRACE_FLAG true
#else
#define SOFA_CRS_POLICY_PRINTTRACE_FLAG false
#endif



/// Traits class which defines the containers to use for a given type of block
template<class Block>
struct CRSBlocTraits
{
    using VecBloc  = sofa::type::vector<Block>;
    using VecIndex = sofa::type::vector<BaseMatrix::Index>;
    using VecFlag  = sofa::type::vector<bool>;
};

class CRSDefaultPolicy
{
public:
    /// Set to true if this matrix is always square (must be true for symmetric)
    static constexpr bool IsAlwaysSquare = false;
    /// Set to true if this matrix is always symmetric (IsAlwaysSquare should be true)
    static constexpr bool IsAlwaysSymmetric = false;
    /// Set to true if the size of the matrix should be automatically increased when new blocs are added
    static constexpr bool AutoSize = false;
    /// Set to true if the matrix should be automatically compressed (easier to use, but might cause issues in multithreading)
    static constexpr bool AutoCompress = true;
    /// Set to true if the blocs that are all zeros should be removed from the matrix when compressing (expensive)
    static constexpr bool CompressZeros = true;
    /// Set to true if clear methods will put all concerned value to zero instead of clearing vectors (CompressZeros should be true)
    static constexpr bool ClearByZeros = true;
    /// Set to true if insertion in matrix are in most case at last line index or last col index
    static constexpr bool OrderedInsertion = false;
    /// Set to true if touch flags should be stored (to remove untouched blocs during compression)
    static constexpr bool StoreTouchFlags = false;
    /// Set to false to disable storage of blocs on the lower triangular part (IsAlwaysSymmetric must be true)
    static constexpr bool StoreLowerTriangularBloc = true;

    static constexpr bool Check   = SOFA_CRS_POLICY_CHECK_FLAG;    

    static constexpr bool Verbose = SOFA_CRS_POLICY_VERBOSE_FLAG;
    /// Set to true if you want to log all methods called to construct matrix in binary file
    static constexpr bool LogTrace = SOFA_CRS_POLICY_LOGTRACE_FLAG;
    /// Set to true if you want to debug binaries file generated by LogTrace Policy, all call will be logged in text file
    static constexpr bool PrintTrace = SOFA_CRS_POLICY_PRINTTRACE_FLAG;
    /// Do not change this value, has to be overrided for all derivated class
    static constexpr int  matrixType = 0;
};

template<typename TBloc, typename TPolicy = CRSDefaultPolicy>
class CompressedRowSparseMatrix : public TPolicy, public BaseMatrix
{
public:
    typedef CompressedRowSparseMatrix<TBloc,TPolicy> Matrix;

    typedef TBloc Block;
    typedef TPolicy Policy;
    typedef matrix_bloc_traits<Block, BaseMatrix::Index> traits;
    typedef typename traits::BlockTranspose BlockTranspose;
    typedef typename traits::Real Real;

    static constexpr Index NL = traits::NL;  ///< Number of rows of a block
    static constexpr Index NC = traits::NC;  ///< Number of columns of a block

    using VecBloc  = typename CRSBlocTraits<Block>::VecBloc;
    using VecIndex = typename CRSBlocTraits<Block>::VecIndex;
    using VecFlag  = typename CRSBlocTraits<Block>::VecFlag;
    typedef BaseMatrix::Index Index;

    typedef sofa::type::Vec<NC,Real> DBloc;

    static_assert(!(Policy::IsAlwaysSymmetric && !Policy::IsAlwaysSquare),
        "IsAlwaysSymmetric can only be true if IsAlwaysSquare is true");
    static_assert(!(!Policy::StoreLowerTriangularBloc && !Policy::IsAlwaysSymmetric),
        "StoreLowerTriangularBloc can only be false if IsAlwaysSymmetric is true");

    struct IndexedBlock
    {
        Index l,c;
        Block value;
        IndexedBlock() {}
        IndexedBlock(Index i, Index j) : l(i), c(j) {}
        IndexedBlock(Index i, Index j, const Block& v) : l(i), c(j), value(v) {}
        bool operator < (const IndexedBlock& b) const
        {
            return (l < b.l) || (l == b.l && c < b.c);
        }
        bool operator <= (const IndexedBlock& b) const
        {
            return (l < b.l) || (l == b.l && c <= b.c);
        }
        bool operator > (const IndexedBlock& b) const
        {
            return (l > b.l) || (l == b.l && c > b.c);
        }
        bool operator >= (const IndexedBlock& b) const
        {
            return (l > b.l) || (l == b.l && c >= b.c);
        }
        bool operator == (const IndexedBlock& b) const
        {
            return (l == b.l) && (c == b.c);
        }
        bool operator != (const IndexedBlock& b) const
        {
            return (l != b.l) || (c != b.c);
        }
    };
    typedef type::vector<IndexedBlock> VecIndexedBlock;

    class Range : public std::pair<Index, Index>
    {
        typedef std::pair<Index, Index> Inherit;
    public:
        Range() : Inherit(0,0) {}
        Range(Index begin, Index end) : Inherit(begin,end) {}
        Index begin() const { return this->first; }
        Index end() const { return this->second; }
        void setBegin(Index i) { this->first = i; }
        void setEnd(Index i) { this->second = i; }
        bool empty() const { return begin() == end(); }
        Index size() const { return end()-begin(); }
        typename VecBloc::iterator begin(VecBloc& b) const { return b.begin() + begin(); }
        typename VecBloc::iterator end  (VecBloc& b) const { return b.begin() + end  (); }
        typename VecBloc::const_iterator begin(const VecBloc& b) const { return b.begin() + begin(); }
        typename VecBloc::const_iterator end  (const VecBloc& b) const { return b.begin() + end  (); }
        typename VecIndex::iterator begin(VecIndex& b) const { return b.begin() + begin(); }
        typename VecIndex::iterator end  (VecIndex& b) const { return b.begin() + end  (); }
        typename VecIndex::const_iterator begin(const VecIndex& b) const { return b.begin() + begin(); }
        typename VecIndex::const_iterator end  (const VecIndex& b) const { return b.begin() + end  (); }
        void operator++() { ++this->first; }
        void operator++(int) { ++this->first; }
    };

    static bool sortedFind(const VecIndex& v, Range in, Index val, Index& result)
    {
        if (in.empty()) return false;
        Index candidate = (result >= in.begin() && result < in.end()) ? result : ((in.begin() + in.end()) >> 1);
        for(;;)
        {
            Index i = v[candidate];
            if (i == val) { result = candidate; return true; }
            if (i < val)  in.setBegin(candidate+1);
            else          in.setEnd(candidate);
            if (in.empty()) break;
            candidate = (in.begin() + in.end()) >> 1;
        }
        return false;
    }

    static bool sortedFind(const VecIndex& v, Index val, Index& result)
    {
        return sortedFind(v, Range(0, Index(v.size())), val, result);
    }

public :
    /// Size
    Index nBlocRow,nBlocCol; ///< Mathematical size of the matrix, in blocks.

    /// Compressed sparse data structure
    VecIndex rowIndex;    ///< indices of non-empty block rows
    VecIndex rowBegin;    ///< column indices of non-empty blocks in each row. The column indices of the non-empty block within the i-th non-empty row are all the colsIndex[j],  j  in [rowBegin[i],rowBegin[i+1])
    VecIndex colsIndex;   ///< column indices of all the non-empty blocks, sorted by increasing row index and column index
    VecBloc  colsValue;   ///< values of the non-empty blocks, in the same order as in colsIndex
    VecFlag  touchedBloc; ///< boolean vector, i-th value is true if bloc has been touched since last compression.

    /// Additional storage to make block insertion more efficient
    VecIndexedBlock btemp; ///< unsorted blocks and their indices

    /// When true, only compressBtemp if needed
    /// This is to avoid compressCRS costly method when no change into matrix size occurs.
    bool skipCompressZero;

    /// Temporary vectors used during compression
    VecIndex oldRowIndex;
    VecIndex oldRowBegin;
    VecIndex oldColsIndex;
    VecBloc  oldColsValue;

    CompressedRowSparseMatrix()
        : nBlocRow(0), nBlocCol(0), skipCompressZero(true)
    {
    }

    CompressedRowSparseMatrix(Index nbBlocRow, Index nbBlocCol)
        : nBlocRow(nbBlocRow), nBlocCol(nbBlocCol)
        , skipCompressZero(true)
    {
    }

    /// \returns the number of row blocs
    Index rowBSize() const
    {
        return nBlocRow;
    }

    /// \returns the number of col blocs
    Index colBSize() const
    {
        return nBlocCol;
    }

    const VecIndex& getRowIndex() const { return rowIndex; }
    const VecIndex& getRowBegin() const { return rowBegin; }
    Range getRowRange(Index id) const { return Range(rowBegin[id], rowBegin[id+1]); }
    const VecIndex& getColsIndex() const { return colsIndex; }
    const VecBloc& getColsValue() const { return colsValue; }

    void resizeBloc(Index nbBRow, Index nbBCol)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::resizeBloc, nbBRow, nbBCol);
        if (nBlocRow == nbBRow && nBlocRow == nbBCol)
        {
            /// Just clear the matrix
            for (Index i = 0; i < static_cast<Index>(colsValue.size()); ++i)
                traits::clear(colsValue[i]);
            skipCompressZero = colsValue.empty();
            btemp.clear();
        }
        else
        {
            nBlocRow = nbBRow;
            nBlocCol = nbBCol;
            rowIndex.clear();
            rowBegin.clear();
            colsIndex.clear();
            colsValue.clear();
            skipCompressZero = true;
            btemp.clear();
            if constexpr (Policy::StoreTouchFlags) touchedBloc.clear();
            if constexpr (Policy::Verbose) std::cout << this->Name()  << ": resizeBloc("<<nbBRow<<","<<nbBCol<<")"<<std::endl;
        }
    }

protected:

    /**
    * \brief Add a new col into matrix
    * @param colId : Index of column
    * @param bvalue : Block value to add
    * @return true if col has been added
    **/
    bool registerNewCol(Index& colId, TBloc& bvalue)
    {
        bool added = false;
        if constexpr (Policy::CompressZeros)
        {
            if (!traits::empty(bvalue))
            {
                colsIndex.push_back(colId);
                colsValue.push_back(bvalue);
                if constexpr (Policy::StoreTouchFlags) touchedBloc.push_back(false);
                added = true;
            }
        }
        else
        {
            colsIndex.push_back(colId);
            colsValue.push_back(bvalue);
            if constexpr (Policy::StoreTouchFlags) touchedBloc.push_back(false);
            added = true;
        }
        return added;
    }

    /**
    * \brief Add a complete new line from btemp into matrix
    * @param itbtemp : Reference to actual status of iterator on btemp
    * @return Number of col added
    **/
    std::pair<Index, Index> registerBtempLine(typename VecIndexedBlock::const_iterator& itbtemp)
    {
        Index curentBtempRowID = itbtemp->l;
        Index internalRowBeginCount = 0;
        Index maxColID = std::numeric_limits<Index>::min();
        typename VecIndexedBlock::const_iterator endbtemp = btemp.end();
        while (itbtemp != endbtemp && itbtemp->l == curentBtempRowID)
        {
            Index curentBtempColID = itbtemp->c;
            Block curentBtempValue = itbtemp->value;
            ++itbtemp;
            while (itbtemp != endbtemp && itbtemp->l == curentBtempRowID && itbtemp->c == curentBtempColID)
            {
                curentBtempValue += itbtemp->value;
                ++itbtemp;
            }
            if (registerNewCol(curentBtempColID, curentBtempValue))
            {
                ++internalRowBeginCount;
                if (curentBtempColID > maxColID) maxColID = curentBtempColID;
            }
        }
        return std::make_pair(internalRowBeginCount, maxColID);
    }

    /**
    * \brief Clear matrix and just add btemp array
    **/
    void fullyCompressBtemp()
    {
        rowIndex.clear();
        rowBegin.clear();
        colsIndex.clear();
        colsValue.clear();
        if constexpr (Policy::StoreTouchFlags) touchedBloc.clear();

        colsIndex.reserve(btemp.size());
        colsValue.reserve(btemp.size());

        Index rowID = 0;
        Index rowBeginID = 0;
        Index maxColID = std::numeric_limits<Index>::min();

        typename VecIndexedBlock::const_iterator itbtemp  = btemp.begin();
        typename VecIndexedBlock::const_iterator endbtemp = btemp.end();
        while(itbtemp != endbtemp)
        {
            rowID = itbtemp->l;
            rowIndex.push_back(rowID);
            rowBegin.push_back(rowBeginID);
            const auto res = registerBtempLine(itbtemp);
            rowBeginID += res.first;
            if (res.second > maxColID) maxColID = res.second;
        }
        rowBegin.push_back(rowBeginID);
        btemp.clear();

        if constexpr (Policy::AutoSize)
        {
            nBlocRow = rowIndex.back() + 1;
            nBlocCol = maxColID + 1;
        }
    }

    /**
    * \brief Method to easy insert new bloc into btemp.
    * @param Line index i and column index j
    * @return pointer on Block value
    **/
    Block* insertBtemp(const Index i, const Index j)
    {
        if (btemp.empty() || btemp.back().l != i || btemp.back().c != j)
        {
            btemp.push_back(IndexedBlock(i,j));
            traits::clear(btemp.back().value);
        }
        return &btemp.back().value;
    }

    /**
    * \brief Method to easy have the max colIndex.
    * Could only be used if AutoSize policy is activated.
    **/
    template< typename = typename std::enable_if< Policy::AutoSize> >
    Index getMaxColIndex()
    {
        Index maxColIndex = 0;
        for (Index rowId = 0; rowId < static_cast<Index>(rowIndex.size()); rowId++)
        {
            Index lastColIndex = colsIndex[rowBegin[rowId+1] - 1];
            if (lastColIndex > maxColIndex) maxColIndex = lastColIndex;
        }
        return maxColIndex;
    }

    /**
    * \brief Method to easy delete row given position in rowIndex.
    * @param RowId position on line in rowIndex
    **/
    void deleteRow(Index rowId)
    {
        Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);

        if constexpr(Policy::ClearByZeros)
        {
            for (Index j = rowRange.begin(); j < rowRange.end(); ++j)
            {
                colsValue[j] = Block();
                if constexpr(Policy::StoreTouchFlags) touchedBloc[j] = true;
            }
        }
        else
        {
            const std::size_t nnzRow  = std::size_t(rowRange.size());

            colsValue.erase(rowRange.begin(colsValue), rowRange.end(colsValue));
            colsIndex.erase(rowRange.begin(colsIndex), rowRange.end(colsIndex));

            for (std::size_t r = std::size_t(rowId); r < rowBegin.size()-1; ++r)
            {
                rowBegin[r+1] -= nnzRow;
            }
            rowBegin.erase(rowBegin.begin()+rowId);
            rowIndex.erase(rowIndex.begin()+rowId);
            const bool lastRowRemoved = rowIndex.empty();
            nBlocRow = lastRowRemoved ? 0 : rowIndex.back()+1;
            if (lastRowRemoved)
            {
                rowBegin.clear();
                nBlocCol = 0;
            }
            else
            {
                if constexpr(Policy::AutoSize)
                {
                    // scan again each row to update nbBlocCol
                    nBlocCol = getMaxColIndex()+1;
                }
            }

        }
    }

public:

    void compress() override
    {
        if (skipCompressZero && btemp.empty())
        {
            return;
        }

        if (!btemp.empty())
        {
            compressBtemp();
        }
        else
        {
            compressCSR();
        }

        if constexpr(Policy::StoreTouchFlags)
        {
            touchedBloc.clear();
            touchedBloc.resize(colsValue.size(), false);
        }

        skipCompressZero = true;
        if constexpr (Policy::LogTrace) logCall(FnEnum::compress);
    }

protected:
    /**
    * \brief Clear matrix and compute new triplet's arrays by combining old ones and btemp(VecIndexedBlock) array
    **/
    void compressBtemp()
    {
        if constexpr (Policy::Verbose) std::cout << this->Name()  << "("<<rowBSize()<<","<<colBSize()<<"): sort "<<btemp.size()<<" temp blocs."<<std::endl;

        std::sort(btemp.begin(), btemp.end());

        if constexpr (Policy::Verbose) std::cout << this->Name()  << "("<<rowBSize()<<","<<colBSize()<<"): blocs sorted."<<std::endl;

        /// In This case, matrix is empty, as btemp is sorted just need to fill triplet arrays with btemp
        if (rowIndex.empty())
        {
            fullyCompressBtemp();
            return;
        }

        /// Save actual matrix status
        oldRowIndex.swap(rowIndex);
        oldRowBegin.swap(rowBegin);
        oldColsIndex.swap(colsIndex);
        oldColsValue.swap(colsValue);

        /// New Matrix status with new bloc added by btemp will be stored here
        rowIndex.clear();
        rowBegin.clear();
        colsIndex.clear();
        colsValue.clear();
        touchedBloc.clear();

        rowIndex.reserve(oldRowIndex.size());
        rowBegin.reserve(oldRowIndex.size() + 1);
        colsIndex.reserve(oldColsIndex.size() + btemp.size());
        colsValue.reserve(oldColsValue.size() + btemp.size());

        typename VecIndexedBlock::const_iterator itbtemp  = btemp.begin();
        typename VecIndexedBlock::const_iterator endbtemp = btemp.end();

        /// Info about btemp
        Index curentBtempRowID = itbtemp->l;

        /// Info about old matrix
        Index oldRowIndexCount = 0;
        Index curentOldRowID = oldRowIndex[oldRowIndexCount];
        Index oldNbRow  = Index(oldRowIndex.size());
        Index oldMaxRowID = oldRowIndex.back();

        Index rowBeginCount = 0;
        Index maxRowID = std::numeric_limits<Index>::max();
        Index maxColID = std::numeric_limits<Index>::max();

        Index maxRegisteredColID = 0;

        while (itbtemp != endbtemp || curentOldRowID <= oldMaxRowID)
        {
            if constexpr (Policy::Verbose) std::cout << this->Name() << "("<<rowBSize()<<","<<colBSize()<<"): oldMaxRowID = "<<oldMaxRowID<<" , curentBtempRowID = "<<curentBtempRowID<<""<<std::endl;

            if (curentOldRowID < curentBtempRowID) /// In this case, we only add old line
            {
                rowIndex.push_back(curentOldRowID);
                rowBegin.push_back(rowBeginCount);
                Range inRow( oldRowBegin[oldRowIndexCount], oldRowBegin[oldRowIndexCount + 1] );
                while (!inRow.empty())
                {
                    if (registerNewCol(oldColsIndex[inRow.begin()], oldColsValue[inRow.begin()])) ++rowBeginCount;
                    ++inRow;
                }
                ++oldRowIndexCount;
                curentOldRowID = (oldRowIndexCount < oldNbRow ) ? oldRowIndex[oldRowIndexCount] : maxRowID;
            }
            else if (curentOldRowID > curentBtempRowID) /// In this case, we only add btemp line
            {
                rowIndex.push_back(curentBtempRowID);
                rowBegin.push_back(rowBeginCount);
                const auto res = registerBtempLine(itbtemp);
                rowBeginCount += res.first;
                if (res.second > maxRegisteredColID) maxRegisteredColID = res.second;
                curentBtempRowID = (itbtemp != endbtemp) ? itbtemp->l : maxRowID;
            }
            else /// In this case, we add mixed btemp line and old line
            {
                rowIndex.push_back(curentOldRowID);
                rowBegin.push_back(rowBeginCount);
                Range inRow( oldRowBegin[oldRowIndexCount], oldRowBegin[oldRowIndexCount + 1] );
                Index oldColID = (!inRow.empty()) ? oldColsIndex[inRow.begin()] : maxColID;
                Index curentBtempColID = (itbtemp != endbtemp && itbtemp->l == curentBtempRowID) ? itbtemp->c : maxColID;
                while ((itbtemp != endbtemp && itbtemp->l == curentBtempRowID) || !inRow.empty())
                {
                    if (oldColID < curentBtempColID) /// In this case, we only add old column
                    {
                        if (registerNewCol(oldColID, oldColsValue[inRow.begin()])) ++rowBeginCount;
                        ++inRow;
                        oldColID = (!inRow.empty()) ? oldColsIndex[inRow.begin()] : maxColID;
                    }
                    else if (oldColID > curentBtempColID) /// In this case, we only add btemp column
                    {
                        Block curentBtempValue = itbtemp->value;
                        ++itbtemp;
                        while (itbtemp != endbtemp && itbtemp->l == curentBtempRowID && itbtemp->c == curentBtempColID)
                        {
                            curentBtempValue += itbtemp->value;
                            ++itbtemp;
                        }
                        if (registerNewCol(curentBtempColID, curentBtempValue)) ++rowBeginCount;
                        if (curentBtempColID > maxRegisteredColID) maxRegisteredColID = curentBtempColID;
                        curentBtempColID = (itbtemp != endbtemp && itbtemp->l == curentBtempRowID) ? itbtemp->c : maxColID;
                    }
                    else
                    {
                        Block curentMixedValue = oldColsValue[inRow.begin()];
                        ++inRow;
                        while (itbtemp != endbtemp && itbtemp->l == curentBtempRowID && itbtemp->c == curentBtempColID)
                        {
                            curentMixedValue += itbtemp->value;
                            ++itbtemp;
                        }
                        if (registerNewCol(curentBtempColID, curentMixedValue)) ++rowBeginCount;
                        if (curentBtempColID > maxRegisteredColID) maxRegisteredColID = curentBtempColID;
                        oldColID = (!inRow.empty()) ? oldColsIndex[inRow.begin()] : maxColID;
                        curentBtempColID = (itbtemp != endbtemp && itbtemp->l == curentBtempRowID) ? itbtemp->c : maxColID;
                    }
                }
                ++oldRowIndexCount;
                curentBtempRowID = (itbtemp != endbtemp) ? itbtemp->l : maxRowID;
                curentOldRowID = (oldRowIndexCount < oldNbRow ) ? oldRowIndex[oldRowIndexCount] : maxRowID;
            }
        }
        if constexpr (Policy::Verbose) std::cout << this->Name() << "("<<rowBSize()<<","<<colBSize()<<"): compressed " << oldColsIndex.size()<<" old blocs and " << btemp.size() << " temp blocs into " << rowIndex.size() << " lines and " << colsIndex.size() << " blocs."<<std::endl;

        rowBegin.push_back(rowBeginCount);
        btemp.clear();

        if constexpr (Policy::AutoSize)
        {
            nBlocRow = rowIndex.back() + 1;
            if (maxRegisteredColID >= nBlocCol)
            {
                nBlocCol = maxRegisteredColID + 1;
            }
        }
    }

    void compressCSR()
    {
        if constexpr (!Policy::CompressZeros) return;
        Index outValues = 0;
        Index outRows = 0;
        for (Index r = 0; r < static_cast<Index>(rowIndex.size()); ++r)
        {
            Index row = rowIndex[r];
            Index rBegin = rowBegin[r];
            Index rEnd = rowBegin[r+1];
            Index outRBegin = outValues;
            for (Index p = rBegin; p != rEnd; ++p)
            {
                if (!traits::empty(colsValue[p]))
                {
                    // keep this value
                    if (p != outValues)
                    {
                        colsValue[outValues] = colsValue[p];
                        colsIndex[outValues] = colsIndex[p];
                    }
                    ++outValues;
                }
            }
            if(outValues != outRBegin)
            {
                // keep this row
                if (r != outRows)
                {
                    rowIndex[outRows] = row;
                }
                if (r != outRows || rBegin != outRBegin)
                {
                    rowBegin[outRows] = outRBegin;
                }
                ++outRows;
            }
        }
        if (static_cast<Index>(rowIndex.size()) != outRows || static_cast<Index>(colsIndex.size()) != outValues)
        {
            rowBegin[outRows] = outValues;
            rowIndex.resize(outRows);
            rowBegin.resize(outRows+1);
            colsIndex.resize(outValues);
            colsValue.resize(outValues);
        }
    }

public:

    void swap(Matrix& m)
    {
        Index t;
        t = nBlocRow; nBlocRow = m.nBlocRow; m.nBlocRow = t;
        t = nBlocCol; nBlocCol = m.nBlocCol; m.nBlocCol = t;
        bool b;
        b = skipCompressZero; skipCompressZero = m.skipCompressZero; m.skipCompressZero = b;
        rowIndex.swap(m.rowIndex);
        rowBegin.swap(m.rowBegin);
        colsIndex.swap(m.colsIndex);
        colsValue.swap(m.colsValue);
        btemp.swap(m.btemp);
        if constexpr (Policy::StoreTouchFlags) touchedBloc.swap(m.touchedBloc);
    }

    /// Make sure all rows have an entry even if they are empty
    void fullRows()
    {
        if constexpr (Policy::AutoCompress) compress();
        if (static_cast<Index>(rowIndex.size()) >= nBlocRow) return;
        oldRowIndex.swap(rowIndex);
        oldRowBegin.swap(rowBegin);
        rowIndex.resize(nBlocRow);
        rowBegin.resize(nBlocRow+1);
        for (Index i=0; i<nBlocRow; ++i)
            rowIndex[i] = i;
        Index j = 0;
        Index b = 0;
        for (Index i = 0; i < static_cast<Index>(oldRowIndex.size()); ++i)
        {
            b = oldRowBegin[i];
            for (; j<=oldRowIndex[i]; ++j)
                rowBegin[j] = b;
        }
        b = !oldRowBegin.empty() ? oldRowBegin[oldRowBegin.size()-1] : Index(0);
        for (; j<=nBlocRow; ++j)
            rowBegin[j] = b;
        if constexpr (Policy::LogTrace) logCall(FnEnum::fullRows);
    }

    /// Make sure all diagonal entries are present even if they are zero
    void fullDiagonal()
    {
        compress();
        Index ndiag = 0;
        for (Index r=0; r<(decltype(r))rowIndex.size(); ++r)
        {
            Index i = rowIndex[r];
            Index b = rowBegin[r];
            Index e = rowBegin[r+1];
            Index t = b;
            while (b < e && (decltype(i))colsIndex[t] != i)
            {
                if ((decltype(i))colsIndex[t] < i)
                    b = t+1;
                else
                    e = t;
                t = (b+e)>>1;
            }
            if (b<e) ++ndiag;
        }
        if (ndiag == nRow) return;

        oldRowIndex.swap(rowIndex);
        oldRowBegin.swap(rowBegin);
        oldColsIndex.swap(colsIndex);
        oldColsValue.swap(colsValue);
        rowIndex.resize(nRow);
        rowBegin.resize(nRow+1);
        colsIndex.resize(oldColsIndex.size()+nRow-ndiag);
        colsValue.resize(oldColsValue.size()+nRow-ndiag);
        Index nv = 0;
        for (Index i=0; i<nRow; ++i) rowIndex[i] = i;
        Index j = 0;
        for (Index i=0; i<(decltype(i))oldRowIndex.size(); ++i)
        {
            for (; j<(decltype(j))oldRowIndex[i]; ++j)
            {
                rowBegin[j] = nv;
                colsIndex[nv] = j;
                traits::clear(colsValue[nv]);
                ++nv;
            }
            rowBegin[j] = nv;
            Index b = oldRowBegin[i];
            Index e = oldRowBegin[i+1];
            for (; b<e && (decltype(j))oldColsIndex[b] < j; ++b)
            {
                colsIndex[nv] = oldColsIndex[b];
                colsValue[nv] = oldColsValue[b];
                ++nv;
            }
            if (b>=e || (decltype(j))oldColsIndex[b] > j)
            {
                colsIndex[nv] = j;
                traits::clear(colsValue[nv]);
                ++nv;
            }
            for (; b<e; ++b)
            {
                colsIndex[nv] = oldColsIndex[b];
                colsValue[nv] = oldColsValue[b];
                ++nv;
            }
            ++j;
        }
        for (; j<nRow; ++j)
        {
            rowBegin[j] = nv;
            colsIndex[nv] = j;
            traits::clear(colsValue[nv]);
            ++nv;
        }
        rowBegin[j] = nv;
    }

    /// Add the given base to all indices.
    /// Use 1 to convert do Fortran 1-based notation.
    /// Note that the matrix will no longer be valid
    /// from the point of view of C/C++ codes. You need
    /// to call again with -1 as base to undo it.
    void shiftIndices(Index base)
    {
        for (Index i=0; i<(Index)rowIndex.size(); ++i)
            rowIndex[i] += base;
        for (Index i=0; i<(Index)rowBegin.size(); ++i)
            rowBegin[i] += base;
        for (Index i=0; i<(Index)colsIndex.size(); ++i)
            colsIndex[i] += base;
    }

protected:

    /**
    * \brief Get bloc method
    * @param Line index i and column index j
    * @return Block value if exist or empty Block if not
    **/
    const Block& bloc(Index i, Index j) const
    {
        static Block empty;
        if constexpr (Policy::Check)
        {
            if (i >= rowBSize() || j >= colBSize())
            {
                std::cerr << "ERROR: invalid read access to bloc ("<<i<<","<<j<<") in "<< this->Name() <<" of bloc size ("<<rowBSize()<<","<<colBSize()<<")"<<std::endl;
                return empty;
            }
        }

        /// \warning this violates the const-ness of the method !
        /// But if AutoCompress policy is activated, we neeed to be sure not missing btemp registered value.
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress();

        if (rowIndex.empty() || i > rowIndex.back()) return empty; /// Matrix is empty or index is upper than registered lines
        if constexpr (Policy::AutoSize) if (j > nBlocCol) return empty; /// Matrix is auto sized so requested column could not exist

        Index rowId = 0;
        if (i == rowIndex.back()) rowId = Index(rowIndex.size() - 1); /// Optimization to avoid do a find when looking for the last line registred
        else if (i == rowIndex.front()) rowId = 0;             /// Optimization to avoid do a find when looking for the first line registred
        else
        {
            rowId = (nBlocRow == 0) ? 0 : Index(i * rowIndex.size() / nBlocRow);
            if (!sortedFind(rowIndex, i, rowId)) return empty;
        }

        Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
        Index colId = 0;
        if (j == colsIndex[rowRange.first]) colId = rowRange.first;                /// Optimization to avoid do a find when looking for the first column registred for specific column
        else if (j == colsIndex[rowRange.second - 1]) colId = rowRange.second - 1; /// Optimization to avoid do a find when looking for the last column registred for specific column
        else
        {
            colId = (nBlocCol == 0) ? 0 : rowRange.begin() + j * rowRange.size() / nBlocCol;
            if (!sortedFind(colsIndex, rowRange, j, colId)) return empty;
        }

        return colsValue[colId];
    }

public:
    /**
    * \brief Write bloc method
    * @param Line index i and column index j
    * @param create, boolean to decide if wbloc could add new value into not existing line/column
    * @return pointer on Block value if exist or nullptr if not
    **/
    Block* wbloc(Index i, Index j, bool create = false)
    {
        if constexpr (Policy::Check && !Policy::AutoSize)
        {
            if (!create && (i >= rowBSize() || j >= colBSize()))
            {
                std::cerr << "ERROR: invalid write access to bloc ("<<i<<","<<j<<") in "<< this->Name() <<" of bloc size ("<<rowBSize()<<","<<colBSize()<<")"<<std::endl;
                return nullptr;
            }
        }

        if constexpr (Policy::OrderedInsertion)
        {
            /// Matrix is empty or index is upper than registered lines
            if (rowIndex.empty() || i > rowIndex.back()) /// Optimization we are registering value at end
            {
                if (!create) return nullptr;
                if (rowIndex.empty() && rowBegin.empty()) rowBegin.push_back(0);
                rowIndex.push_back(i);
                colsIndex.push_back(j);
                colsValue.push_back(Block());
                rowBegin.push_back(colsIndex.size());

                if constexpr (Policy::StoreTouchFlags) touchedBloc.push_back(false);
                if constexpr (Policy::AutoSize)
                {
                    nBlocRow = i + 1;
                    if (j > nBlocCol) nBlocCol = j + 1;
                }
                return &colsValue.back();
            }
            else if (i == rowIndex.back()) /// In this case, we are trying to write on last registered line
            {
                Index rowId = rowIndex.size() - 1;
                Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
                if (j == colsIndex[rowRange.second - 1]) /// In this case, we are trying to write on last registered column, directly return ref on it
                {
                    if constexpr(Policy::StoreTouchFlags) touchedBloc[rowRange.second - 1] = true;
                    return &colsValue[rowRange.second - 1];
                }
                else if (j > colsIndex[rowRange.second - 1]) /// Optimization we are trying to write on last line et upper of last column, directly create it.
                {
                    if (!create) return nullptr;
                    colsIndex.push_back(j);
                    colsValue.push_back(Block());
                    rowBegin.back()++;
                    if constexpr (Policy::StoreTouchFlags) touchedBloc.push_back(false);
                    if constexpr (Policy::AutoSize)
                    {
                        if (j > nBlocCol) nBlocCol = j + 1;
                    }
                    return &colsValue.back();
                }
                else
                {
                    Index colId = (nBlocCol == 0) ? 0 : rowRange.begin() + j * rowRange.size() / nBlocCol;
                    if (!sortedFind(colsIndex, rowRange, j, colId)) return create ? insertBtemp(i,j) : nullptr;
                    if constexpr (Policy::StoreTouchFlags) touchedBloc[colId] = true;
                    return &colsValue[colId];
                }
            }

            if constexpr (Policy::AutoSize) if (j > nBlocCol) return create ? insertBtemp(i,j) : nullptr; /// Matrix is auto sized so requested column could not exist

            Index rowId = 0;
            if (i == rowIndex.back()) rowId = rowIndex.size() - 1;      /// Optimization to avoid do a find when looking for the last line registred
            else if (i == rowIndex.front()) rowId = 0;                  /// Optimization to avoid do a find when looking for the first line registred
            else
            {
                rowId = (nBlocRow == 0) ? 0 : i * rowIndex.size() / nBlocRow;
                if (!sortedFind(rowIndex, i, rowId)) return create ? insertBtemp(i,j) : nullptr;
            }

            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            Index colId = 0;
            if (j == colsIndex[rowRange.first]) colId = rowRange.first;                /// Optimization to avoid do a find when looking for the first column registred for specific column
            else if (j == colsIndex[rowRange.second - 1]) colId = rowRange.second - 1; /// Optimization to avoid do a find when looking for the last column registred for specific column
            else
            {
                colId = (nBlocCol == 0) ? 0 : rowRange.begin() + j * rowRange.size() / nBlocCol;
                if (!sortedFind(colsIndex, rowRange, j, colId)) return create ? insertBtemp(i,j) : nullptr;
            }

            if constexpr(Policy::StoreTouchFlags) touchedBloc[colId] = true;
            return &colsValue[colId];
        }
        else
        {
            Index rowId = (nBlocRow == 0) ? 0 : Index(i * rowIndex.size() / nBlocRow);
            if (sortedFind(rowIndex, i, rowId))
            {
                Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
                Index colId = (nBlocCol == 0) ? 0 : rowRange.begin() + j * rowRange.size() / nBlocCol;
                if (sortedFind(colsIndex, rowRange, j, colId))
                {
                    if constexpr(Policy::StoreTouchFlags) touchedBloc[colId] = true;
                    return &colsValue[colId];
                }
            }
            if (create)
            {
                if (btemp.empty() || btemp.back().l != i || btemp.back().c != j)
                {
                    btemp.push_back(IndexedBlock(i,j));
                    traits::clear(btemp.back().value);
                }
                return &btemp.back().value;
            }
            return nullptr;
        }
    }

    /**
    * \brief Write bloc method when rowId and colId are known, this is an optimized wbloc specification
    * @param Line index i and column index j
    * @param rowId : Index of value i into rowIndex internal vector
    * @param colId : Index of value j into colIndex internal vector
    * @param create, boolean to decide if wbloc could add new value into not existing line/column
    * @return pointer on Block value if exist or nullptr if not
    **/
    Block* wbloc(Index i, Index j, Index& rowId, Index& colId, bool create = false)
    {
        if constexpr (Policy::Check && !Policy::AutoSize)
        {
            if (!create && (i >= rowBSize() || j >= colBSize()))
            {
                std::cerr << "ERROR: invalid write access to bloc ("<<i<<","<<j<<") in "<< this->Name() <<" of bloc size ("<<rowBSize()<<","<<colBSize()<<")"<<std::endl;
                return nullptr;
            }
        }

        bool rowFound = true;
        if (rowId < 0 || rowId >= static_cast<Index>(rowIndex.size()) || rowIndex[rowId] != i)
        {
            rowId = Index(i * rowIndex.size() / nBlocRow);
            rowFound = sortedFind(rowIndex, i, rowId);
        }
        if (rowFound)
        {
            bool colFound = true;
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            if (colId < rowRange.begin() || colId >= rowRange.end() || colsIndex[colId] != j)
            {
                colId = rowRange.begin() + j * rowRange.size() / nBlocCol;
                colFound = sortedFind(colsIndex, rowRange, j, colId);
            }
            if (colFound)
            {
                return &colsValue[colId];
            }
        }

        if (create)
        {
            if (btemp.empty() || btemp.back().l != i || btemp.back().c != j)
            {
                btemp.push_back(IndexedBlock(i,j));
                traits::clear(btemp.back().value);
            }
            return &btemp.back().value;
        }
        return nullptr;
    }

    const Block& getBloc(Index i, Index j) const
    {
        if constexpr (!Policy::StoreLowerTriangularBloc) if (i > j) assert(false);
        return bloc(i,j);
    }

    const BlockTranspose getSymBloc(Index i, Index j) const
    {
        if constexpr (!Policy::StoreLowerTriangularBloc) if (i > j) return traits::transposed( bloc(i,j) );
        return getBloc(i,j);
    }

    void setBloc(Index i, Index j, const Block& v)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::setBloc, i, j, v);
        if constexpr (!Policy::StoreLowerTriangularBloc) if (i > j) return;
        *wbloc(i,j,true) = v;
    }

    void addBloc(Index i, Index j, const Block& v)
    {
        if constexpr (!Policy::StoreLowerTriangularBloc) if (i > j) return;
        *wbloc(i,j,true) += v;
    }

    void setBloc(Index i, Index j, Index& rowId, Index& colId, const Block& v)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::setBlocId, i, j, rowId, colId, v);
        if constexpr (!Policy::StoreLowerTriangularBloc) if (i > j) return;
        *wbloc(i,j,rowId,colId,true) = v;
    }

    void addBloc(Index i, Index j, Index& rowId, Index& colId, const Block& v)
    {
        if constexpr (!Policy::StoreLowerTriangularBloc) if (i > j) return;
        *wbloc(i,j,rowId,colId,true) += v;
    }

    Block* getWBloc(Index i, Index j, bool create = false)
    {
        if constexpr (!Policy::StoreLowerTriangularBloc) if (i > j) return nullptr;
        return wbloc(i,j,create);
    }

    /**
    * \brief Clear row bloc method. Clear all col of this line.
    * @param i : Line index considering size of matrix in bloc.
    * \warning if ClearByZeros Policy is activated all col value of line will be set to zero using default constructor
    **/
    void clearRowBloc(Index i)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::clearRowBloc, i);
        if constexpr (Policy::Verbose)
        {
            if constexpr (Policy::ClearByZeros) std::cout << this->Name()  << "("<<rowBSize()<<","<<colBSize()<<"): row("<<i<<") = 0"<<std::endl;
            else std::cout << this->Name()  << "("<<rowBSize()<<","<<colBSize()<<"): row("<<i<<") cleared"<<std::endl;
        }
        if constexpr (Policy::Check)
        {
            if (i >= rowBSize())
            {
                std::cerr << "ERROR: invalid write access to row "<<i<<" in "<< this->Name() << " of size ("<<rowBSize()<<","<<colBSize()<<")"<<std::endl;
                return;
            }
        }
        if constexpr (Policy::AutoCompress) compress(); /// If AutoCompress policy is activated, we neeed to be sure not missing btemp registered value.
        if constexpr (Policy::IsAlwaysSquare && !Policy::ClearByZeros) /// In Square case removing only Row will produce not quare matrix
        {
            clearRowColBloc(i);
            return;
        }

        Index rowId = 0;
        if (i == rowIndex.back()) rowId = Index(rowIndex.size() - 1);      /// Optimization to avoid do a find when looking for the last line registred
        else if (i == rowIndex.front()) rowId = 0;                  /// Optimization to avoid do a find when looking for the first line registred
        else
        {
            rowId = (nBlocRow == 0) ? 0 : Index(i * rowIndex.size() / nBlocRow);
            if (!sortedFind(rowIndex, i, rowId)) return;
        }

        deleteRow(rowId);

        if constexpr (Policy::AutoCompress && Policy::ClearByZeros) compress(); /// If AutoCompress policy is activated, need to compress zeros.
    }

    /**
    * \brief Clear col bloc method. Clear this col in all row of matrix.
    * @param j : Col index considering size of matrix in bloc.
    * \warning if ClearByZeros Policy is activated all col j of each line will be set to zero using default constructor
    **/
    void clearColBloc(Index j)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::clearColBloc, j);
        if constexpr (Policy::Verbose)
        {
            if constexpr (Policy::ClearByZeros) std::cout << this->Name()  << "("<<rowBSize()<<","<<colBSize()<<"): col("<<j<<") = 0"<<std::endl;
            else std::cout << this->Name()  << "("<<rowBSize()<<","<<colBSize()<<"): col("<<j<<") cleared"<<std::endl;
        }
        if constexpr (Policy::Check)
        {
            if (j >= colBSize())
            {
                std::cerr << "ERROR: invalid write access to col "<<j<<" in "<< this->Name() << " of size ("<<rowBSize()<<","<<colBSize()<<")"<<std::endl;
                return;
            }
        }
        /// If AutoCompress policy is activated, we neeed to be sure not missing btemp registered value.
        if constexpr (Policy::AutoCompress) compress();
        if constexpr (Policy::IsAlwaysSquare && !Policy::ClearByZeros) /// In Square case removing only Col will produce not square matrix
        {
            clearRowColBloc(j);
            return;
        }

        for (Index rowId = static_cast<Index>(rowIndex.size())-1; rowId >=0 ; --rowId)
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);

            Index colId = -1;
            if (j == colsIndex[rowRange.first]) colId = rowRange.first;                /// Optimization to avoid do a find when looking for the first column registred for specific column
            else if (j == colsIndex[rowRange.second - 1]) colId = rowRange.second - 1; /// Optimization to avoid do a find when looking for the last column registred for specific column
            else
            {
                colId = (nBlocCol == 0) ? 0 : rowRange.begin() + j * rowRange.size() / nBlocCol;
                if (!sortedFind(colsIndex, rowRange, j, colId)) colId = -1;
            }
            if (colId != -1) /// Means col exist in this line
            {
                if constexpr (Policy::ClearByZeros)
                {
                    colsValue[colId] = Block();
                    if constexpr (Policy::StoreTouchFlags) touchedBloc[colId] = true;
                }
                else
                {
                    if constexpr (Policy::AutoCompress)
                    {
                        /// In this case, line was containing only this column, directly clearing is faster than putting to zero and compressing.
                        if (rowRange.second - 1 == rowRange.first)
                        {
                            deleteRow(rowId);
                            continue;
                        }
                    }

                    for (auto it = std::next(rowBegin.begin(), rowId + 1); it != rowBegin.end(); it++)
                        *it -= 1;

                    colsIndex.erase(std::next(colsIndex.begin(), colId));
                    colsValue.erase(std::next(colsValue.begin(), colId));
                }
            }
        }

        if constexpr (Policy::AutoSize)
        {
            nBlocRow = rowIndex.empty() ? 0 : rowIndex.back() + 1; /// To be sure if row has been erased
            if (j == nBlocCol - 1) nBlocCol = getMaxColIndex() + 1;
        }

        if constexpr (Policy::AutoCompress && Policy::ClearByZeros) compress(); /// If AutoCompress policy is activated, need to compress zeros.
    }

    std::size_t countEmptyBlocs() const
    {
        return std::count_if(this->colsValue.cbegin(), this->colsValue.cend(), [] (const Block& b)
        {
            return traits::empty(b);
        });
    }

    /**
    * \brief Clear both row i and column i in a square matrix
    * @param i : Row and Col index considering size of matrix in bloc.
    * \warning if ClearByZeros Policy is activated all col i and line i values of will be set to zero using default constructor
    **/
    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void clearRowColBloc(Index i)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::clearRowColBloc, i);
        if constexpr (Policy::Verbose) std::cout << this->Name()  << "("<<rowBSize()<<","<<colBSize()<<"): row("<<i<<") = 0 and col("<<i<<") = 0"<<std::endl;
        if constexpr (Policy::Check)
        {
            if (i >= rowBSize() || i >= colBSize())
            {
                std::cerr << "ERROR: invalid write access to row and column "<<i<<" in "<< this->Name() << " of size ("<<rowBSize()<<","<<colBSize()<<")"<<std::endl;
                return;
            }
        }
        /// If AutoCompress policy is activated, we neeed to be sure not missing btemp registered value.
        if constexpr (Policy::AutoCompress) compress();

        bool foundRowId = true;
        Index rowId = 0;
        if (i == rowIndex.back()) rowId = rowIndex.size() - 1;      /// Optimization to avoid do a find when looking for the last line registred
        else if (i == rowIndex.front()) rowId = 0;                  /// Optimization to avoid do a find when looking for the first line registred
        else
        {
            rowId = (nBlocRow == 0) ? 0 : i * rowIndex.size() / nBlocRow;
            if (!sortedFind(rowIndex, i, rowId)) foundRowId = false;
        }

        bool foundColId = true;
        Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
        Index colId = 0;
        if (i == colsIndex[rowRange.first]) colId = rowRange.first;                /// Optimization to avoid do a find when looking for the first column registred for specific column
        else if (i == colsIndex[rowRange.second - 1]) colId = rowRange.second - 1; /// Optimization to avoid do a find when looking for the last column registred for specific column
        else
        {
            colId = (nBlocCol == 0) ? 0 : rowRange.begin() + i * rowRange.size() / nBlocCol;
            if (!sortedFind(colsIndex, rowRange, i, colId)) foundColId = false;;
        }

        if (!foundRowId && !foundColId)
        {
            std::cerr << "ERROR: invalid write access to row and column "<<i<<" in "<< this->Name() << " of size ("<<rowBSize()<<","<<colBSize()<<")"<<std::endl;
            return;
        }

        deleteRow(rowId); /// Do not call clearRow to only compress zero if activated once.
        clearColBloc(i);
    }

    /**
    * \brief Completely clear the matrix
    * \warning if ClearByZeros Policy is activated all value in colsValue will be set to zero using default constructor
    **/
    void clear() override
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::clear);
        if constexpr (Policy::ClearByZeros)
        {
            for (Index i = 0; i < static_cast<Index>(colsValue.size()); ++i)
                traits::clear(colsValue[i]);
            skipCompressZero = colsValue.empty();
        }
        else
        {
            rowIndex.clear();
            rowBegin.clear();
            colsIndex.clear();
            colsValue.clear();
            nBlocRow = 0;
            nBlocCol = 0;
            skipCompressZero = true;
            if constexpr (Policy::StoreTouchFlags) touchedBloc.clear();
        }

        btemp.clear();
    }


/// @name BlocMatrixWriter operators
/// @{

    // The SFINAE stuff disables some warnings about hiding overloads from BaseMatrix::add
    template <typename T = Block, typename std::enable_if_t<
                  !std::is_same_v<T, double>, int > = 0 >
    void add(unsigned int bi, unsigned int bj, const Block& b)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::add, bi, bj, b);
        addBloc(bi, bj, b);
    }

    template <typename T = Block, typename std::enable_if_t<
                  std::is_same_v<T, double>, int > = 0 >
    void add(unsigned int bi, unsigned int bj, const Block& b)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::add, bi, bj, b);
        addBloc(bi, bj, b);
    }

    template <typename T = Block, typename std::enable_if_t<
                  !std::is_same_v<T, double>, int > = 0 >
    void add(unsigned int bi, unsigned int bj, int& rowId, int& colId, const Block& b)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::addId, bi, bj, rowId, colId, b);
        addBloc(bi, bj, rowId, colId, b);
    }

    template <typename T = Block, typename std::enable_if_t<
                  std::is_same_v<T, double>, int > = 0 >
    void add(unsigned int bi, unsigned int bj, int& rowId, int& colId, const Block& b)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::addId, bi, bj, rowId, colId, b);
        addBloc(bi, bj, rowId, colId, b);
    }

    void addDBloc(unsigned int bi, unsigned int bj, const DBloc& b)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::addDBloc, bi, bj, b);
        if constexpr (!Policy::StoreLowerTriangularBloc) if (bi > bj) return;
        Block* mb = wbloc(bi, bj, true);

        for (unsigned int i = 0; i < NL; ++i)
            traits::vadd(*mb, i, i, b[i] );
    }

    void addDValue(unsigned int bi, unsigned int bj, const Real b)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::addDValue, bi, bj, b);
        if constexpr (!Policy::StoreLowerTriangularBloc) if (bi > bj) return;
        Block* mb = wbloc(bi, bj, true);

        for (unsigned int i = 0; i < NL; ++i)
            traits::vadd(*mb, i, i, b);
    }

    void addDValue(unsigned int bi, unsigned int bj, int& rowId, int& colId, const Real b)
    {
        if constexpr (Policy::LogTrace) logCall(FnEnum::addDValueId, bi, bj, rowId, colId, b);
        if constexpr (!Policy::StoreLowerTriangularBloc) if (bi > bj) return;
        Block* mb = wbloc(bi, bj, rowId, colId, true);

        for (unsigned int i = 0; i < NL; ++i)
            traits::vadd(*mb, i, i, b);
    }

    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void addDiag(unsigned int bi, const Block& b)
    {
        add(bi, bi, b);
    }

    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void addDiag(unsigned int bi, int& rowId, int& colId, const Block &b)
    {
        add(bi, bi, rowId, colId, b);
    }

    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void addDiagDBloc(unsigned int bi, const DBloc& b)
    {
        addDBloc(bi, bi, b);
    }

    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void addDiagDValue(unsigned int bi, const Real b)
    {
        addDValue(bi, bi, b);
    }

    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void addDiagDValue(unsigned int bi, int& rowId, int& colId, const Real b)
    {
        addDValue(bi, bi, rowId, colId, b);
    }

    template< typename = typename std::enable_if< Policy::IsAlwaysSymmetric> >
    void addSym(unsigned int bi, unsigned int bj, const Block& b)
    {
        if constexpr(Policy::StoreLowerTriangularBloc)
        {
            add(bi, bj, b);
            add(bj, bi, traits::transposed(b) );
        }
        else
        {
            if (bi > bj) // the block we received is in the lower triangular
            {
                add(bj, bi, traits::transposed(b) );
            }
            else
            {
                add(bi, bj, b);
            }
        }
    }

    template< typename = typename std::enable_if< Policy::IsAlwaysSymmetric> >
    void addSym(unsigned int bi, unsigned int bj, int& rowId, int& colId, int& rowIdT, int& colIdT, const Block &b)
    {
        if constexpr(Policy::StoreLowerTriangularBloc)
        {
            add(bi, bj, rowId, colId, b);
            add(bj, bi, rowIdT, colIdT, traits::transposed(b) );
        }
        else
        {
            if (bi > bj) // the block we received is in the lower triangular
            {
                add(bj, bi, rowIdT, colIdT, traits::transposed(b) );
            }
            else
            {
                add(bi, bj, rowId, colId, b);
            }
        }
    }

    template< typename = typename std::enable_if< Policy::IsAlwaysSymmetric> >
    void addSymDBloc(unsigned int bi, unsigned int bj, const DBloc& b)
    {
        unsigned int i = std::min(bi, bj);
        unsigned int j = std::max(bi, bj);
        addDBloc(i, j, b);
        if constexpr (Policy::StoreLowerTriangularBloc) addDBloc(j, i, b);
    }

    template< typename = typename std::enable_if< Policy::IsAlwaysSymmetric> >
    void addSymDValue(unsigned int bi, unsigned int bj, const Real b)
    {
        unsigned int i = std::min(bi, bj);
        unsigned int j = std::max(bi, bj);
        addDValue(i, j, b);
        if constexpr (Policy::StoreLowerTriangularBloc) addDValue(j, i, b);
    }

    template< typename = typename std::enable_if< Policy::IsAlwaysSymmetric> >
    void addSymDValue(unsigned int bi, unsigned int bj, int& rowId, int& colId, int& rowIdT, int& colIdT, Real b)
    {
        unsigned int i = std::min(bi, bj);
        unsigned int j = std::max(bi, bj);
        addDValue(i, j, rowId, colId, b);
        if constexpr (Policy::StoreLowerTriangularBloc) addDValue(j, i, rowIdT, colIdT, b);
    }

/// @}


/// @name Matrix operators
/// @{

    /// Transpose the matrix into res, works only for 3 array variant ("full rows") matrices, ie which can be expressed using the rowBegin, colsIndex and colsValue arrays solely
    template<typename TBloc2, typename TPolicy2>
    void transposeFullRows(CompressedRowSparseMatrix<TBloc2, TPolicy2>& res) const
    {
        res.nBlocCol = nBlocRow;
        res.nBlocRow = nBlocCol;

        res.rowBegin.clear();
        res.rowBegin.resize(res.nBlocRow+1,0);

        res.colsIndex.clear();
        res.colsIndex.resize(this->colsIndex.size(),0);

        res.colsValue.clear();
        res.colsValue.resize(this->colsValue.size());

        res.rowIndex.clear();

        for (unsigned i = 0; i<rowIndex.size(); ++i)
        {
            for (int p = rowBegin[i]; p<rowBegin[i+1]; ++p)
            {
                ++res.rowBegin[colsIndex[p]];
            }
        }

        Index count = 0;
        VecIndex positions(res.nBlocRow);

        for (int i=0; i<res.nBlocRow; ++i)
        {
            Index tmp = res.rowBegin[i];
            res.rowBegin[i] = count;
            positions[i] = count;
            count += tmp;
        }
        res.rowBegin[ res.nBlocRow ] = count;

        for (unsigned i=0; i<rowIndex.size(); ++i)
        {
            int row = rowIndex[i];

            for (int p=rowBegin[i]; p<rowBegin[i+1]; ++p)
            {
                Index col                 = colsIndex[p];
                Index pos                 = positions[col];
                res.colsIndex[pos]        = row;
                res.colsValue[pos]        = colsValue[p];
                ++positions[col];
             }
        }

        res.rowIndex.resize(res.rowBegin.size()-1);
        for (unsigned i=0; i<res.rowIndex.size(); ++i)
        {
            res.rowIndex[i] = i;
        }
    }

    /** Compute res = this * m
      @warning The block sizes must be compatible, i.e. this::NC==m::NR and res::NR==this::NR and res::NC==m::NC.
      The basic algorithm consists in accumulating rows of m to rows of res: foreach row { foreach col { res[row] += this[row,col] * m[col] } }
      @warning matrices this and m must be compressed
      */
    template<typename RB, typename RP, typename MB, typename MP >
    void mul( CompressedRowSparseMatrix<RB,RP>& res, const CompressedRowSparseMatrix<MB,MP>& m ) const
    {
        assert( Block::nbCols == MB::nbLines );
        assert( RB::nbLines == Block::nbLines );
        assert( MB::nbCols == RB::nbCols );

        assert( colBSize() == m.rowBSize() );

        if constexpr (Policy::AutoCompress)
        {
            const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
            (const_cast<CompressedRowSparseMatrix<MB,MP>*>(&m))->compress();  /// \warning this violates the const-ness of the parameter
        }

        res.resizeBloc( this->nBlocRow, m.nBlocCol );  // clear and resize the result

        if( m.rowIndex.empty() ) return; // if m is null

        for( Index xi = 0; xi < Index(rowIndex.size()); ++xi )  // for each non-null block row
        {
            unsigned mr = 0; // block row index in m

            Index row = rowIndex[xi];      // block row

            Range rowRange( rowBegin[xi], rowBegin[xi+1] );
            for( Index xj = rowRange.begin() ; xj < rowRange.end() ; ++xj )  // for each non-null block
            {
                Index col = colsIndex[xj];     // block column
                const Block& b = colsValue[xj]; // block value

                // find the non-null row in m, if any
                while( mr<m.rowIndex.size() && m.rowIndex[mr]<col ) mr++;
                if( mr==m.rowIndex.size() || m.rowIndex[mr] > col ) continue;  // no matching row, ignore this block

                // Accumulate  res[row] += b * m[col]
                Range mrowRange( m.rowBegin[mr], m.rowBegin[mr+1] );
                for( Index mj = mrowRange.begin() ; mj< mrowRange.end() ; ++mj ) // for each non-null block in  m[col]
                {
                    Index mcol = m.colsIndex[mj];     // column index of the non-null block
                    *res.wbloc(row,mcol,true) += b * m.colsValue[mj];  // find the matching bloc in res, and accumulate the block product
                }
            }
        }
        res.compress();
    }
    
    static auto blocMultTranspose(const Block& blockA, const Block& blockB)
    {
        if constexpr(isMatrix<Block>())
        {
            return blockA.multTranspose(blockB);
        }
        else
        {
            return blockA * blockB;
        }
    }

    /** Compute res = this.transpose * m
      @warning The block sizes must be compatible, i.e. this::NR==m::NR and res::NR==this::NC and res::NC==m::NC
      The basic algorithm consists in accumulating rows of m to rows of res: foreach row { foreach col { res[row] += this[row,col] * m[col] } }
      @warning matrices this and m must be compressed
      */
    template<typename RB, typename RP, typename MB, typename MP >
    void mulTranspose( CompressedRowSparseMatrix<RB,RP>& res, const CompressedRowSparseMatrix<MB,MP>& m ) const
    {
        assert( Block::nbLines == MB::nbLines );
        assert( RB::nbLines == Block::nbCols );
        assert( MB::nbCols == RB::nbCols );

        assert( rowBSize() == m.rowBSize() );

        if constexpr (Policy::AutoCompress)
        {
            const_cast<Matrix*>(this)->compress();  /// \warning this violates the const-ness of the method
            (const_cast<CompressedRowSparseMatrix<MB,MP>*>(&m))->compress();  /// \warning this violates the const-ness of the parameter
        }


        res.resizeBloc( this->nBlocCol, m.nBlocCol );  // clear and resize the result

        if( m.rowIndex.empty() ) return; // if m is null

        for( Index xi = 0 ; xi < rowIndex.size() ; ++xi )  // for each non-null transpose block column
        {
            unsigned mr = 0; // block row index in m

            Index col = rowIndex[xi];      // block col (transposed col = row)

            Range rowRange( rowBegin[xi], rowBegin[xi+1] );
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)  // for each non-null block
            {
                Index row = colsIndex[xj];     // block row (transposed row = col)
                const Block& b = colsValue[xj]; // block value

                // find the non-null row in m, if any
                while( mr<m.rowIndex.size() && m.rowIndex[mr]<col ) mr++;
                if( mr==m.rowIndex.size() || m.rowIndex[mr] > col ) continue;  // no matching row, ignore this block

                // Accumulate  res[row] += b^T * m[col]
                Range mrowRange( m.rowBegin[mr], m.rowBegin[mr+1] );
                for( Index mj = mrowRange.begin() ; mj< mrowRange.end() ; ++mj ) // for each non-null block in  m[col]
                {
                    Index mcol = m.colsIndex[mj];     // column index of the non-null block
                    //*res.wbloc(row,mcol,true) += b.multTranspose( m.colsValue[mj] );  // find the matching bloc in res, and accumulate the block product
                    *res.wbloc(row, mcol, true) += blocMultTranspose(b, m.colsValue[mj]);  // find the matching bloc in res, and accumulate the block product
                }
            }
        }
        res.compress();
    }

/// @}

    static const char* Name()
    {
        static std::string name = std::string("CompressedRowSparseMatrix") + std::string(traits::Name());
        return name.c_str();
    }

    bool check_matrix()
    {
        return check_matrix(
                Index(this->getColsValue().size()),
                this->rowBSize(),
                this->colBSize(),
                static_cast<Index*> (&(rowBegin[0])),
                static_cast<Index*> (&(colsIndex[0])),
                static_cast<Block*> (&(colsValue[0]))
                );
    }

    static bool check_matrix(
        Index nzmax,    // nb values
        Index m,        // number of row
        Index n,        // number of columns
        Index * a_p,    // column pointers (size n+1) or col indices (size nzmax)
        Index * a_i,    // row indices, size nzmax
        Block * a_x      // numerical values, size nzmax
    )
    {
        // check ap, size m beecause ther is at least the diagonal value wich is different of 0
        if (a_p[0]!=0)
        {
            std::cerr << "CompressedRowSparseMatrix: First value of row indices (a_p) should be 0" << std::endl;
            return false;
        }

        for (Index i=1; i<=m; i++)
        {
            if (a_p[i]<=a_p[i-1])
            {
                std::cerr << "CompressedRowSparseMatrix: Row (a_p) indices are not sorted index " << i-1 << " : " << a_p[i-1] << " , " << i << " : " << a_p[i] << std::endl;
                return false;
            }
        }
        if (nzmax == -1)
        {
            nzmax = a_p[m];
        }
        else if (a_p[m]!=nzmax)
        {
            std::cerr << "CompressedRowSparseMatrix: Last value of row indices (a_p) should be " << nzmax << " and is " << a_p[m] << std::endl;
            return false;
        }


        Index k=1;
        for (Index i=0; i<nzmax; i++)
        {
            i++;
            for (; i<a_p[k]; i++)
            {
                if (a_i[i] <= a_i[i-1])
                {
                    std::cerr << "CompressedRowSparseMatrix: Column (a_i) indices are not sorted index " << i-1 << " : " << a_i[i-1] << " , " << i << " : " << a_p[i] << std::endl;
                    return false;
                }
                if (a_i[i]<0 || a_i[i]>=n)
                {
                    std::cerr << "CompressedRowSparseMatrix: Column (a_i) indices are not correct " << i << " : " << a_i[i] << std::endl;
                    return false;
                }
            }
            k++;
        }

        for (Index i=0; i<nzmax; i++)
        {
            if (traits::empty(a_x[i]))
            {
                std::cerr << "CompressedRowSparseMatrix: Warning, matrix contains empty block at index " << i << std::endl;
                return false;
            }
        }

        if (n!=m)
        {
            std::cerr << "CompressedRowSparseMatrix: the matrix is not square" << std::endl;
            return false;
        }

        std::cerr << "Check_matrix passed successfully" << std::endl;
        return true;
    }

    void setTraceWriter(CRSTraceWriter<Matrix>* matrixTraceWriter)
    {
        if constexpr (!Policy::LogTrace) std::cout<<"ERROR : You need to activate LogTrace Policy to be able to trace matrix."<<std::endl;
        else
        {
            m_traceWriter = matrixTraceWriter;
        }
    }

    void unsetTraceWriter()
    {
        setTraceWriter(nullptr);
    }

    std::ostream& write(std::ostream& os) const
    {
        os << rowIndex << std::endl;
        os << rowBegin << std::endl;
        os << colsIndex << std::endl;
        os << colsValue << std::endl;

        return os;
    }

    std::istream& read(std::istream& is)
    {
        {
            std::string line;
            std::getline(is, line);
            std::istringstream(line) >> rowIndex;
        }
        {
            std::string line;
            std::getline(is, line);
            std::istringstream(line) >> rowBegin;
        }
        {
            std::string line;
            std::getline(is, line);
            std::istringstream(line) >> colsIndex;
        }
        {
            std::string line;
            std::getline(is, line);
            std::istringstream(line) >> colsValue;
        }

        return is;
    }

protected:

    CRSTraceWriter<Matrix>* m_traceWriter = nullptr;

    template <class ...Args>
    void logCall(const char fnId, Args... args)
    {
        if (!m_traceWriter) return;
        m_traceWriter->logCall_imp(fnId, args...);
    }

    template<typename TVec>
    void writeVector(const TVec& vec, std::ostream& os)
    {
        for (auto& v : vec)
            os <<v<<";";
        os<<std::endl;
    }

    template<typename TVec>
    void readVector(TVec& vec, std::istream& in)
    {
        std::string temp;
        while (std::getline(in, temp, ';'))
        {
            vec.push_back(std::stoi(temp));
        }
    }

    static void split_row_index(BaseMatrix::Index& index, BaseMatrix::Index& modulo) { bloc_index_func<NL, BaseMatrix::Index>::split(index, modulo); }
    static void split_col_index(BaseMatrix::Index& index, BaseMatrix::Index& modulo) { bloc_index_func<NC, BaseMatrix::Index>::split(index, modulo); }
public:
    BaseMatrix::Index rowSize(void) const override { return rowBSize(); }
    BaseMatrix::Index colSize(void) const override { return colBSize(); }
    SReal element(BaseMatrix::Index i, BaseMatrix::Index j) const override
    {
        BaseMatrix::Index bi = 0, bj = 0; split_row_index(i, bi); split_col_index(j, bj);
        ((Matrix*)this)->compress();  /// \warning this violates the const-ness of the method !
        return (SReal)traits::v(bloc(i, j), bi, bj);
    }
    void resize(BaseMatrix::Index nbRow, BaseMatrix::Index nbCol) override
    {
        resizeBloc((nbRow + NL - 1) / NL, (nbCol + NC - 1) / NC);
        nBlocRow = nbRow;
        nBlocCol = nbCol;
    }

    void set(BaseMatrix::Index i, BaseMatrix::Index j, double v) override
    {
        BaseMatrix::Index bi = 0, bj = 0; split_row_index(i, bi); split_col_index(j, bj);
        traits::v(*wbloc(i, j, true), bi, bj) = (Real)v;
    }

    void add(BaseMatrix::Index i, BaseMatrix::Index j, double v) override
    {
        BaseMatrix::Index bi = 0, bj = 0; split_row_index(i, bi); split_col_index(j, bj);
        traits::v(*wbloc(i, j, true), bi, bj) += (Real)v;
    }

    void add(BaseMatrix::Index row, BaseMatrix::Index col, const type::Mat3x3d& _M) override
    {
        BaseMatrix::add(row, col, _M);
    }

    void add(BaseMatrix::Index row, BaseMatrix::Index col, const type::Mat3x3f& _M) override
    {
        BaseMatrix::add(row, col, _M);
    }

    /// @name setter/getter & product methods on template vector types
    /// @{

    template<class Vec> static Real vget(const Vec& vec, Index i, Index j, Index k) { return vget(vec, i * j + k); }
    template<class Vec> static Real vget(const type::vector<Vec>& vec, Index i, Index /*j*/, Index k) { return vec[i][k]; }

    static auto  vget(const linearalgebra::BaseVector& vec, Index i) { return vec.element(i); }
    template<class Real2> static Real2 vget(const FullVector<Real2>& vec, Index i) { return vec[i]; }


    template<class Vec> static void vset(Vec& vec, Index i, Index j, Index k, Real v) { vset(vec, i * j + k, v); }
    template<class Vec> static void vset(type::vector<Vec>& vec, Index i, Index /*j*/, Index k, Real v) { vec[i][k] = v; }

    static void vset(linearalgebra::BaseVector& vec, Index i, Real v) { vec.set(i, v); }
    template<class Real2> static void vset(FullVector<Real2>& vec, Index i, Real2 v) { vec[i] = v; }


    template<class Vec> static void vadd(Vec& vec, Index i, Index j, Index k, Real v) { vadd(vec, i * j + k, v); }
    template<class Vec> static void vadd(type::vector<Vec>& vec, Index i, Index /*j*/, Index k, Real v) { vec[i][k] += v; }

    static void vadd(linearalgebra::BaseVector& vec, Index i, Real v) { vec.add(i, v); }
    template<class Real2> static void vadd(FullVector<Real2>& vec, Index i, Real2 v) { vec[i] += v; }

    template<class Vec> static void vresize(Vec& vec, Index /*blockSize*/, Index totalSize) { vec.resize(totalSize); }
    template<class Vec> static void vresize(type::vector<Vec>& vec, Index blockSize, Index /*totalSize*/) { vec.resize(blockSize); }

    /** Product of the matrix with a templated vector res = this * vec*/
    template<class Real2, class V1, class V2>
    void tmul(V1& res, const V2& vec) const
    {
        assert(vec.size() % bColSize() == 0); // vec.size() must be a multiple of block size.

        ((Matrix*)this)->compress();
        vresize(res, rowBSize(), rowSize());
        for (Index xi = 0; xi < (Index)rowIndex.size(); ++xi)  // for each non-empty block row
        {
            type::Vec<NL, Real2> r;  // local block-sized vector to accumulate the product of the block row  with the large vector

            // multiply the non-null blocks with the corresponding chunks of the large vector
            Range rowRange(rowBegin[xi], rowBegin[xi + 1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                // transfer a chunk of large vector to a local block-sized vector
                type::Vec<NC, Real2> v;
                //Index jN = colsIndex[xj] * NC;    // scalar column index
                for (Index bj = 0; bj < NC; ++bj)
                    v[bj] = vget(vec, colsIndex[xj], NC, bj);

                // multiply the block with the local vector
                const auto& b = colsValue[xj];    // non-null block has block-indices (rowIndex[xi],colsIndex[xj]) and value colsValue[xj]
                for (Index bi = 0; bi < NL; ++bi)
                    for (Index bj = 0; bj < NC; ++bj)
                        r[bi] += traits::v(b, bi, bj) * v[bj];
            }

            // transfer the local result  to the large result vector
            //Index iN = rowIndex[xi] * NL;                      // scalar row index
            for (Index bi = 0; bi < NL; ++bi)
                vset(res, rowIndex[xi], NL, bi, r[bi]);
        }
    }

    /// @name specialization of product methods on a few vector types
    /// @{

    /// equal result = this * v
    /// @warning The block sizes must be compatible ie v.size() must be a multiple of block size.
    template< typename V1, typename V2 >
    void mul(V2& result, const V1& v) const
    {
        tmul< Real, V2, V1 >(result, v);
    }

    /// @returns this * v
    /// @warning The block sizes must be compatible ie v.size() must be a multiple of block size.
    template<class Vec>
    Vec operator*(const Vec& v) const
    {
        Vec res;
        mul(res, v);
        return res;
    }

    typedef bool filter_fn(Index   i, Index   j, Block& val, const Block& ref);
    static bool       nonzeros(Index /*i*/, Index /*j*/, Block& val, const Block& /*ref*/) { return (!traits::empty(val)); }

    template<class TMatrix>
    void filterValues(TMatrix& M, filter_fn* filter = &nonzeros, const Block& ref = Block())
    {
        M.compress();
        nBlocRow = M.rowBSize();
        nBlocCol = M.colBSize();
        rowIndex.clear();
        rowBegin.clear();
        colsIndex.clear();
        colsValue.clear();
        btemp.clear();
        rowIndex.reserve(M.rowIndex.size());
        rowBegin.reserve(M.rowBegin.size());
        colsIndex.reserve(M.colsIndex.size());
        colsValue.reserve(M.colsValue.size());

        Index vid = 0;
        for (Index rowId = 0; rowId < (Index)M.rowIndex.size(); ++rowId)
        {
            Index i = M.rowIndex[rowId];
            rowIndex.push_back(i);
            rowBegin.push_back(vid);
            Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId + 1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index j = M.colsIndex[xj];
                Block& b = M.colsValue[xj];
                if ((*filter)(i, j, b, ref))
                {
                    colsIndex.push_back(j);
                    colsValue.push_back(b);
                    ++vid;
                }
            }
            if ((decltype(vid))rowBegin.back() == vid) // row was empty
            {
                rowIndex.pop_back();
                rowBegin.pop_back();
            }
        }
        rowBegin.push_back(vid); // end of last row
    }

    template <class TMatrix>
    void copyNonZeros(TMatrix& M)
    {
        filterValues(M, nonzeros, Block());
    }
};

//#ifdef SPARSEMATRIX_CHECK
//#undef SPARSEMATRIX_CHECK
//#endif
//#ifdef SPARSEMATRIX_VERBOSE
//#undef SPARSEMATRIX_VERBOSE
//#endif

template<> template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<double>::filterValues<CompressedRowSparseMatrix<type::Mat<3, 3, double> > >(CompressedRowSparseMatrix<type::Mat<3, 3, double> >& M, filter_fn* filter, const Block& ref);
template<> template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<double>::filterValues<CompressedRowSparseMatrix<type::Mat<3, 3, float> > >(CompressedRowSparseMatrix<type::Mat<3, 3, float> >& M, filter_fn* filter, const Block& ref);
template<> template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<float>::filterValues<CompressedRowSparseMatrix<type::Mat<3, 3, float> > >(CompressedRowSparseMatrix<type::Mat<3, 3, float> >& M, filter_fn* filter, const Block& ref);
template<> template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<float>::filterValues<CompressedRowSparseMatrix<type::Mat<3, 3, double> > >(CompressedRowSparseMatrix<type::Mat<3, 3, double> >& M, filter_fn* filter, const Block& ref);

#if !defined(SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIX_CPP) 
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<float>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<double>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<1, 1, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<1, 1, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<2, 2, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<2, 2, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<3, 3, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<3, 3, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<4, 4, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<4, 4, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<6, 6, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<6, 6, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<8, 8, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<8, 8, double> >;
#endif

} // namespace sofa::linearalgebra
