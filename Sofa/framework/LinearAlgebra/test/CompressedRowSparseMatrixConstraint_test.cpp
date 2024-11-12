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
/******************************************************************************
* Contributors:
*   - InSimo
*******************************************************************************/

#include <gtest/gtest.h>

#include <sofa/helper/RandomGenerator.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraint.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/testing/LinearCongruentialRandomGenerator.h>

#ifndef SOFA_ASSERT
#define SOFA_ASSERT(condition)
#endif

namespace sofa
{

/// Specific policy for benchmark on CRSMatrixConstraint
class CRSConstraintTestPolicy : public sofa::linearalgebra::CRSConstraintPolicy
{
public:
    static constexpr bool LogTrace = false;
    static constexpr bool PrintTrace = false;
};

template <typename TMatrix>
struct SparseMatrixTest : public ::testing::Test
{
};

typedef ::testing::Types<
            sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::defaulttype::Vec1Types::Deriv, CRSConstraintTestPolicy>,
            sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::defaulttype::Vec2Types::Deriv, CRSConstraintTestPolicy>,
            sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::defaulttype::Vec3Types::Deriv, CRSConstraintTestPolicy>,
            sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::defaulttype::Rigid2Types::Deriv, CRSConstraintTestPolicy>,
            sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::defaulttype::Rigid3Types::Deriv, CRSConstraintTestPolicy>
            > SparseMatrixTestTypes;

TYPED_TEST_CASE(SparseMatrixTest, SparseMatrixTestTypes);

namespace TestHelpers
{

//////////////////////////////////////////////////
template <typename TMatrix>
struct line_t
{
    struct Data
    {
        typename TMatrix::KeyType index;
        typename TMatrix::Data value;
    };

    typename TMatrix::KeyType rowIndex;
    Data data1, data2, data3;

    static const unsigned int initialDataCount = 3u;
};

template <typename TMatrix>
line_t<TMatrix> nullLine()
{
    line_t<TMatrix> line = { 0, { 0, typename TMatrix::Data() }, { 0, typename TMatrix::Data() }, { 0, typename TMatrix::Data() } };
    return line;
}


//////////////////////////////////////////////////
template <typename TData>
void createData(TData& data, unsigned int index)
{
    for (unsigned int i = 0; i < TData::total_size; i++)
    {
        data[i] = index + i + 1;
    }
}

//////////////////////////////////////////////////
template <typename TMatrix>
sofa::type::vector<typename TMatrix::KeyType> Populate(TMatrix& matrix, unsigned int nbLine, unsigned int nbCol)
{
    typedef TMatrix Matrix;
    typedef typename TMatrix::KeyType KeyType;

    sofa::type::vector<KeyType> lineIndices;

    for (unsigned int i = 0; i < nbLine; i++)
    {
        sofa::helper::RandomGenerator randomGenerator(i + 1);
        KeyType lindex =  randomGenerator.random(0u, 500u);
        lineIndices.push_back(lindex);

        typename Matrix::RowIterator itRow = matrix.writeLine(lindex);

        for (unsigned int j = 0; j < nbCol; j++)
        {
            typename Matrix::Data data;
            sofa::helper::RandomGenerator randomGenerator(i + j + 2);
            KeyType cindex =  randomGenerator.random(0u, 500u);

            for (unsigned int k = 0; k < Matrix::Data::total_size; k++)
            {
                data[k] = lindex + j + k + 1;
            }
            itRow.setCol(cindex, data);
        }
    }

    matrix.compress();
    return lineIndices;
}

//////////////////////////////////////////////////
template <typename TMatrix>
const line_t<TMatrix> PopulateCol(TMatrix& matrix, typename TMatrix::KeyType rowIndex = 42)
{
    typedef TMatrix Matrix;
    typedef line_t<TMatrix> line_t;

    line_t result = nullLine<TMatrix>();

    result.rowIndex = rowIndex;

    typename Matrix::RowIterator itRow = matrix.writeLine(result.rowIndex);

    sofa::helper::RandomGenerator randomGenerator1(1);
    result.data1.index = randomGenerator1.random(0u, 500u);

    for (unsigned int j = 0; j < Matrix::Data::total_size; j++)
    {
        result.data1.value[j] = result.rowIndex + j + 1;
    }
    itRow.setCol(result.data1.index, result.data1.value);

    sofa::helper::RandomGenerator randomGenerator2(2);
    result.data2.index = randomGenerator2.random(static_cast<unsigned int>(result.data1.index), 500u);

    for (unsigned int j = 0; j < Matrix::Data::total_size; j++)
    {
        result.data2.value[j] = result.rowIndex + j + 1;
    }
    itRow.setCol(result.data2.index, result.data2.value);

    matrix.compress();

    return result;
}

//////////////////////////////////////////////////
template <typename TMatrix>
const line_t<TMatrix> WriteLine(TMatrix& matrix, typename TMatrix::KeyType rowIndex,
                              typename TMatrix::KeyType startColIndex = std::numeric_limits<typename TMatrix::KeyType>::max())
{
    typedef TMatrix Matrix;
    typedef line_t<TMatrix> line_t;
    if (startColIndex == std::numeric_limits<typename Matrix::KeyType>::max())
    {
        startColIndex = rowIndex + 1;
    }

    line_t result = nullLine<TMatrix>();

    result.rowIndex = rowIndex;

    typename Matrix::RowIterator itRow = matrix.writeLine(result.rowIndex);

    {
        result.data1.index = startColIndex;
        for (unsigned int j = 0; j < Matrix::Data::total_size; j++)
        {
            result.data1.value[j] = startColIndex + j + 1;
        }
        itRow.setCol(result.data1.index, result.data1.value);
    }

    {
        result.data2.index = startColIndex + Matrix::Data::total_size + 1;
        for (unsigned int j = 0; j < Matrix::Data::total_size; j++)
        {
            result.data2.value[j] = result.data2.index + j + 1;
        }
        itRow.setCol(result.data2.index, result.data2.value);
    }

    {
        result.data3.index = startColIndex + 2 * Matrix::Data::total_size + 2;
        for (unsigned int j = 0; j < Matrix::Data::total_size; j++)
        {
            result.data3.value[j] = result.data3.index + j + 1;
        }
        itRow.setCol(result.data3.index, result.data3.value);
    }

    matrix.compress();
    return result;
}

//////////////////////////////////////////////////
template <typename TMatrix>
typename TMatrix::KeyType GetNextUniqueIndex(const typename TMatrix::KeyType rowIndex)
{
    return rowIndex + 13;
}

} // namespace TestHelpers

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatASparseMatrixIsEmptyAfterInstanciation)
{
    typedef TypeParam Matrix;
    EXPECT_TRUE(Matrix().empty());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheSizeOfASparseMatrixIsZeroAfterInstanciation)
{
    typedef TypeParam Matrix;
    EXPECT_EQ(static_cast<std::size_t>(0u), Matrix().size());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheSizeOfASparseMatrixIsOneAfterALineHasBeenWritten)
{
    typedef TypeParam Matrix;
    Matrix matrix;
    TestHelpers::PopulateCol(matrix);
    EXPECT_EQ(1u, matrix.size());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheSizeOfASparseMatrixIsTwoAfterTryingToWriteTwoLinesWithDistinctIndices)
{
    typedef TypeParam Matrix;
    Matrix matrix;
    TestHelpers::Populate(matrix, 2u, 1u);
    EXPECT_EQ(2u, matrix.size());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheSizeOfASparseMatrixIsOneAfterTryingToWriteTwoLinesWithTheSameIndex)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    {
        unsigned int lineIndex1 = 123;
        typename Matrix::Data data1;
        TestHelpers::createData(data1, lineIndex1);
        matrix.writeLine(lineIndex1).setCol(0, data1);

        typename Matrix::Data data2;
        TestHelpers::createData(data1, lineIndex1);
        matrix.writeLine(lineIndex1).setCol(1, data2);
        matrix.compress();
    }

    EXPECT_EQ(1u, matrix.size());
}


//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckOutInSerialization)
{
    typedef TypeParam Matrix;
    Matrix outMatrix;
    auto lineIndices = TestHelpers::Populate(outMatrix, 3u, 3u);
    EXPECT_EQ(3u, outMatrix.size());

    std::ostringstream oss;
    oss << outMatrix;

    Matrix inMatrix;
    std::istringstream iss(oss.str());
    iss >> inMatrix;

    EXPECT_TRUE(!lineIndices.empty());

    for (const auto& lineIndex : lineIndices)
    {
        auto inRowIt = inMatrix.readLine(lineIndex);
        auto outRowIt = outMatrix.readLine(lineIndex);

        EXPECT_TRUE(inRowIt == outRowIt);

        if (inRowIt != inMatrix.end() && outRowIt != outMatrix.end())
        {
            auto inColItEnd = inRowIt.end();
            auto outColItEnd = outRowIt.end();

            auto inColIt = inRowIt.begin();
            auto outColIt = outRowIt.begin();

            for (; 
                inColIt != inColItEnd && outColIt != outColItEnd;
                ++inColIt, ++outColIt)
            {
                EXPECT_TRUE(inColIt.index() == outColIt.index());
                EXPECT_TRUE(inColIt.val() == outColIt.val());
            }
        }
    }

}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatAMatrixIsConsideredEmptyAfterIsHasBeenCleared)
{
    typedef TypeParam Matrix;
    Matrix matrix;
    TestHelpers::PopulateCol(matrix);
    matrix.clear();
    EXPECT_TRUE(matrix.empty());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatALineCanBeCorrectlyRetrievedByItsIndex)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    sofa::type::vector<typename Matrix::KeyType> lindices = TestHelpers::Populate(matrix, 3u, 1u);
    EXPECT_TRUE(!lindices.empty());

    for (const auto& index : lindices)
    {
        EXPECT_EQ(index, matrix.readLine(index).index());
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheFirstLineOfAMatrixCanBeRetrieved)
{
    typedef TypeParam Matrix;
    Matrix matrix;
    TestHelpers::PopulateCol(matrix);

    EXPECT_EQ(42u, matrix.begin().index());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTrueIsReturnedWhenComparingTwoRowIteratorsPointingToTheSameLine)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    typename Matrix::RowIterator itRowA = matrix.writeLine(42);
    typename Matrix::RowIterator itRowB = itRowA;

    EXPECT_TRUE(itRowA == itRowB);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTrueIsReturnedWhenComparingTwoRowConstIteratorsPointingToTheSameLine)
{
    typedef TypeParam Matrix;
    Matrix matrix;
    sofa::type::vector<typename Matrix::KeyType> lindices = TestHelpers::Populate(matrix, 1u, 1u);

    EXPECT_TRUE(!lindices.empty());

    typename Matrix::RowConstIterator itRowA = matrix.readLine(lindices[0]);
    typename Matrix::RowConstIterator itRowB = matrix.readLine(lindices[0]);

    EXPECT_TRUE(itRowA == itRowB);
}

////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTrueIsReturnedWhenComparingTwoRowIteratorsPointingToDistinctLinesForInequality)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    typename Matrix::RowIterator itRowA =  matrix.writeLine(123);
    typename Matrix::RowIterator itRowB =  matrix.writeLine(456);

    EXPECT_TRUE(itRowA != itRowB);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTrueIsReturnedWhenComparingTwoRowConstIteratorsPointingToDistinctLinesForInequality)
{
    typedef TypeParam Matrix;
    Matrix matrix;
    sofa::type::vector<typename Matrix::KeyType> lindices = TestHelpers::Populate(matrix, 2u, 1u);

    EXPECT_TRUE(!lindices.empty());
    typename Matrix::RowConstIterator itRow0 = matrix.readLine(lindices[0]);
    typename Matrix::RowConstIterator itRow1 = matrix.readLine(lindices[1]);

    EXPECT_TRUE(itRow0 != itRow1);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheBeginningAndTheEndAreTheSameWhenAMatrixIsEmpty)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    EXPECT_EQ(matrix.end(), matrix.begin());
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTryingToClearAnEmptyMatrixLetItUnchanged)
{
    typedef TypeParam Matrix;
    Matrix matrix;
    matrix.clear();

    EXPECT_TRUE(matrix.empty());
    EXPECT_EQ(matrix.end(), matrix.begin());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatPreIncrementingSizeTimesTheConstIteratorToTheBeginningOfAMatrixResultsInThePastTheEndIterator)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    sofa::type::vector<typename Matrix::KeyType> lindices = TestHelpers::Populate(matrix, 3u, 1u);
    EXPECT_TRUE(!lindices.empty());

    const Matrix& constMatrix = matrix;
    typename Matrix::RowConstIterator itRow = constMatrix.begin();
    ++itRow; ++itRow; ++itRow;
    EXPECT_EQ(constMatrix.end(), itRow);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatARowIteratorCopiedFromAnotherOneIsConsideredEqualToIt)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    TestHelpers::PopulateCol(matrix);

    const typename Matrix::RowIterator itRow = matrix.writeLine(42);
    typename Matrix::RowIterator itRowCopy = matrix.writeLine(42);
    itRowCopy = itRow;

    EXPECT_EQ(itRowCopy, itRow);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatARowConstIteratorCopiedFromAnotherOneIsConsideredEqualToIt)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    matrix.writeLine(42);

    typename Matrix::RowConstIterator itRow = matrix.readLine(42);
    typename Matrix::RowConstIterator itRowCopy = matrix.readLine(42);;
    itRowCopy = itRow;

    EXPECT_EQ(itRowCopy, itRow);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatPostIncrementingSizeTimesTheConstIteratorToTheBeginningOfAMatrixResultsInThePastTheEndIterator)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    sofa::type::vector<typename Matrix::KeyType> lindices = TestHelpers::Populate(matrix, 3u, 1u);
    EXPECT_TRUE(!lindices.empty());

    const Matrix& constMatrix = matrix;
    typename Matrix::RowConstIterator itRow = constMatrix.begin();
    itRow++; itRow++; itRow++;
    EXPECT_EQ(constMatrix.end(), itRow);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatAnIteratorToALineIsReturnedWhenTryingToWriteItButItExistsAlready)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    typename Matrix::RowIterator itRowA = matrix.writeLine(42);
    typename Matrix::RowIterator itRowB = matrix.writeLine(42);

    EXPECT_EQ(itRowB, itRowA);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatThePastTheEndIteratorIsReturnedWhenTryingToReadALineThatDoesNotExist)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    {
        matrix.writeLine(42);
        matrix.writeLine(43);
    }

    const Matrix& constMatrix = matrix;

    EXPECT_EQ(constMatrix.end(), matrix.readLine(44));
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheSizeOfALineCanBeRetrieved)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::PopulateCol(matrix);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);
    EXPECT_EQ(2u, itRow.row().size());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatALineCanBeReadAndItsDataRetrievedByUsingARowInternalConstIterator)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::PopulateCol(matrix);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);
    auto itCol = itRow.begin();

    {
        EXPECT_EQ(line1.data1.index, itCol.index());
        EXPECT_EQ(line1.data1.value, itCol.val());
    }

    {
        itCol++;
        EXPECT_EQ(line1.data2.index, itCol.index());
        EXPECT_EQ(line1.data2.value, itCol.val());
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatALineCanBeIdenticallyRepopulatedAfterItHasBeenClearedInAOneLineMatrix)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, 42);

    // Clear
    {
        matrix.clearColBlock(line.rowIndex);
    }

    // Repopulation
    const TestHelpers::line_t<Matrix> newLine = TestHelpers::WriteLine(matrix, line.rowIndex);
    EXPECT_EQ(newLine.rowIndex, line.rowIndex);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
    auto itCol = itRow.begin();

    {
        EXPECT_EQ(line.data1.index, itCol.index());
        EXPECT_EQ(line.data1.value, itCol.val());
    }

    {
        itCol++;
        EXPECT_EQ(line.data2.index, itCol.index());
        EXPECT_EQ(line.data2.value, itCol.val());
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatALineCanBeRepopulatedWithDifferentColumnsIndicesAndDataAfterItHasBeenClearedInAOneLineMatrix)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::WriteLine(matrix, 42);
    // Clear
    {
        matrix.clearRowBlock(line1.rowIndex);
    }

    // Repopulation
    const TestHelpers::line_t<Matrix> newLine = TestHelpers::WriteLine(matrix, line1.rowIndex, TestHelpers::GetNextUniqueIndex<Matrix>(line1.rowIndex));
    EXPECT_EQ(newLine.rowIndex, line1.rowIndex);
    typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);
    auto itCol = itRow.begin();

    {
        EXPECT_EQ(newLine.data1.index, itCol.index());
        EXPECT_EQ(newLine.data1.value, itCol.val());
    }

    {
        itCol++;
        EXPECT_EQ(newLine.data2.index, itCol.index());
        EXPECT_EQ(newLine.data2.value, itCol.val());
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatATotallyDifferentLineCanBeAddedAfterALineHasBeenClearedInAOneLineMatrix)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::WriteLine(matrix, 42);

    // Clear
    {
        matrix.clearColBlock(line1.rowIndex);
    }

    // Repopulation
    const TestHelpers::line_t<Matrix> newLine = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line1.rowIndex));
    EXPECT_NE(newLine.rowIndex, line1.rowIndex);

    typename Matrix::RowConstIterator itRow = matrix.readLine(newLine.rowIndex);
    auto itCol = itRow.begin();

    {
        EXPECT_EQ(newLine.data1.index, itCol.index());
        EXPECT_EQ(newLine.data1.value, itCol.val());
    }

    {
        itCol++;
        EXPECT_EQ(newLine.data2.index, itCol.index());
        EXPECT_EQ(newLine.data2.value, itCol.val());
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatALineCanBeIdenticallyRepopulatedAfterItHasBeenClearedInAMultiLinesMatrix)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::WriteLine(matrix, 42);
    const TestHelpers::line_t<Matrix> line2 = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line1.rowIndex));
    TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line2.rowIndex));

    // Clear
    {
        matrix.clearColBlock(line2.rowIndex);
    }

    // Repopulation
    const TestHelpers::line_t<Matrix> newLine = TestHelpers::WriteLine(matrix, line2.rowIndex);
    EXPECT_EQ(newLine.rowIndex, line2.rowIndex);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line2.rowIndex);
    auto itCol = itRow.begin();

    {
        EXPECT_EQ(newLine.data1.index, itCol.index());
        EXPECT_EQ(newLine.data1.value, itCol.val());
    }

    {
        itCol++;
        EXPECT_EQ(newLine.data2.index, itCol.index());
        EXPECT_EQ(newLine.data2.value, itCol.val());
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatALineCanBeRepopulatedWithDifferentColumnsIndicesAndDataAfterItHasBeenClearedInAMultiLinesMatrix)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::WriteLine(matrix, 42);
    const TestHelpers::line_t<Matrix> line2 = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line1.rowIndex));
    const TestHelpers::line_t<Matrix> line3 = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line2.rowIndex));

    // Clear
    {
        matrix.clearRowBlock(line2.rowIndex);
    }

    // Repopulation
    const TestHelpers::line_t<Matrix> newLine = TestHelpers::WriteLine(matrix, line2.rowIndex, TestHelpers::GetNextUniqueIndex<Matrix>(line3.rowIndex));
    EXPECT_EQ(newLine.rowIndex, line2.rowIndex);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line2.rowIndex);
    auto itCol = itRow.begin();

    {
        EXPECT_EQ(newLine.data1.index, itCol.index());
        EXPECT_EQ(newLine.data1.value, itCol.val());
    }

    {
        itCol++;
        EXPECT_EQ(newLine.data2.index, itCol.index());
        EXPECT_EQ(newLine.data2.value, itCol.val());
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatATotallyDifferentLineCanBeAddedAfterALineHasBeenClearedInAMultiLineMatrix)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::WriteLine(matrix, 42);
    const TestHelpers::line_t<Matrix> line2 = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line1.rowIndex));
    const TestHelpers::line_t<Matrix> line3 = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line2.rowIndex));

    // Clear
    {
        matrix.clearRowBlock(line1.rowIndex);
    }

    // Repopulation
    const TestHelpers::line_t<Matrix> newLine = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line3.rowIndex));
    EXPECT_NE(newLine.rowIndex, line1.rowIndex);
    EXPECT_NE(newLine.rowIndex, line2.rowIndex);
    EXPECT_NE(newLine.rowIndex, line3.rowIndex);

    typename Matrix::RowConstIterator itRow = matrix.readLine(newLine.rowIndex);
    auto itCol = itRow.begin();

    {
        EXPECT_EQ(newLine.data1.index, itCol.index());
        EXPECT_EQ(newLine.data1.value, itCol.val());
    }

    {
        itCol++;
        EXPECT_EQ(newLine.data2.index, itCol.index());
        EXPECT_EQ(newLine.data2.value, itCol.val());
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatOlderLinesStayUnaffectedWhenAMoreRecentOneIsCleared)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::WriteLine(matrix, 42);
    const TestHelpers::line_t<Matrix> line2 = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line1.rowIndex));
    TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line2.rowIndex));

    // Clear
    {
        matrix.clearColBlock(line2.rowIndex);
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);
        auto itCol = itRow.begin();

        {
            EXPECT_EQ(line1.data1.index, itCol.index());
            EXPECT_EQ(line1.data1.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line1.data2.index, itCol.index());
            EXPECT_EQ(line1.data2.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line1.data3.index, itCol.index());
            EXPECT_EQ(line1.data3.value, itCol.val());
        }
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatMoreRecentLinesStayUnaffectedWhenAnOlderOneIsCleared)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::WriteLine(matrix, 42);
    const TestHelpers::line_t<Matrix> line2 = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line1.rowIndex));
    const TestHelpers::line_t<Matrix> line3 = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(line2.rowIndex));

    // Clear
    {
        matrix.clearColBlock(line2.rowIndex);
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(line3.rowIndex);
        auto itCol = itRow.begin();

        {
            EXPECT_EQ(line3.data1.index, itCol.index());
            EXPECT_EQ(line3.data1.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line3.data2.index, itCol.index());
            EXPECT_EQ(line3.data2.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line3.data3.index, itCol.index());
            EXPECT_EQ(line3.data3.value, itCol.val());
        }
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheSizeOfALineIsDecrementedByOneWhenAnElementIsErased)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, 42);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
    const typename Matrix::RowType& row1 = itRow.row();
    const typename Matrix::RowType& row = itRow.row();

    EXPECT_EQ(row1, row);

    const typename Matrix::KeyType expectedSize = row.size() - 1;
    matrix.clearColBlock(line.data2.index);
    EXPECT_EQ(expectedSize, itRow.row().size());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatThePastTheEndIteratorIsReturnedWhenTryingToAccessAnErasedElementInALine)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, 42);

    {
        matrix.clearColBlock(line.data3.index);
    }

    typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
    typename Matrix::ColConstIterator itData = itRow.begin();

    for (unsigned int i = 0; i < line.initialDataCount - 1; i++)  {
        ++itData;
    }

    EXPECT_EQ(itRow.end(), itData);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatOlderElementsStayUnaffectedWhenAMoreRecentOneIsErasedFromALine)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, 42);

    {
        matrix.clearColBlock(line.data2.index);
    }

    SOFA_ASSERT(line.initialDataCount == 3u);
    typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
    typename Matrix::ColConstIterator itData = itRow.begin();
    EXPECT_EQ(line.data1.index, itData.index());
    EXPECT_EQ(line.data1.value, itData.val());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatMoreRecentElementsStayUnaffectedWhenAnOlderOneIsErasedFromALine)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, 42);

    {
        matrix.clearColBlock(line.data2.index);
    }

    SOFA_ASSERT(line.initialDataCount == 3);
    typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
    typename Matrix::ColConstIterator itData = itRow.begin();
    ++itData;
    EXPECT_EQ(line.data3.index, itData.index());
    EXPECT_EQ(line.data3.value, itData.val());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTryingToEraseAColumnThatDoesNotExistInALineLeavesTheSizeOfTheRowUnchanged)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, 42);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
    const typename Matrix::KeyType expectedSize = itRow.row().size();
    matrix.clearColBlock(line.data3.index + 1);
    EXPECT_EQ(expectedSize, itRow.row().size());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTryingToEraseAColumnThatDoesNotExistInALineLeavesItUnchanged)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, 42);

    {
        matrix.clearColBlock(line.data3.index + 1);
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
        auto itCol = itRow.begin();

        {
            EXPECT_EQ(line.data1.index, itCol.index());
            EXPECT_EQ(line.data1.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line.data2.index, itCol.index());
            EXPECT_EQ(line.data2.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line.data3.index, itCol.index());
            EXPECT_EQ(line.data3.value, itCol.val());
        }
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatNewElementsCanBeInsertedIntoARowAfterAllOthersHaveBeenErased)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const typename Matrix::KeyType rowIndex = 42;

    {
        const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, rowIndex);

        {
            SOFA_ASSERT(line.initialDataCount == 3);
            matrix.clearColBlock(line.data1.index);
            matrix.clearColBlock(line.data2.index);
            matrix.clearColBlock(line.data3.index);
        }
    }

    const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, rowIndex, TestHelpers::GetNextUniqueIndex<Matrix>(rowIndex));

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
        auto itCol = itRow.begin();

        {
            EXPECT_EQ(line.data1.index, itCol.index());
            EXPECT_EQ(line.data1.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line.data2.index, itCol.index());
            EXPECT_EQ(line.data2.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line.data3.index, itCol.index());
            EXPECT_EQ(line.data3.value, itCol.val());
        }
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheColumnIndexOfAColConstIteratorCanBeRetrieved)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const typename Matrix::KeyType rowIndex = 42;

    // Populate matrix
    {
        typename Matrix::RowIterator itRow = matrix.writeLine(rowIndex);
        typename Matrix::Data data1;
        TestHelpers::createData(data1, rowIndex);
        itRow.setCol(123, data1);
        matrix.compress();
    }

    typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
    typename Matrix::ColConstIterator itData = itRow.begin();
    EXPECT_EQ(123u, itData.index());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheValueOfADataCanBeRetrieved)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const typename Matrix::KeyType rowIndex = 42;

    typename Matrix::RowIterator writeRow = matrix.writeLine(rowIndex);
    typename Matrix::Data data1;
    TestHelpers::createData(data1, rowIndex);
    writeRow.setCol(123, data1);
    matrix.compress();

    typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
    typename Matrix::ColConstIterator itData = itRow.begin();
    EXPECT_EQ(data1, itData.val());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatAValueCanBeAddedToADataInARow)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const typename Matrix::KeyType rowIndex = 42;
    const typename Matrix::KeyType colIndex = 123;

    typename Matrix::RowIterator writeRow = matrix.writeLine(rowIndex);
    typename Matrix::Data data1;
    TestHelpers::createData(data1, rowIndex);
    writeRow.setCol(colIndex, data1);
    matrix.compress();

    typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
    typename Matrix::ColConstIterator itData = itRow.begin();
    EXPECT_EQ(data1, itData.val());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatSettingAnExistingElementUpdatesItsValue)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::PopulateCol(matrix);

    typename Matrix::RowIterator writeRow = matrix.writeLine(line1.rowIndex);
    typename Matrix::Data data2;
    TestHelpers::createData(data2, line1.rowIndex);
    writeRow.setCol(line1.data2.index, data2);
    matrix.compress();

    const Matrix& constMatrix = matrix;
    typename Matrix::RowConstIterator itRow = constMatrix.begin();
    typename Matrix::ColConstIterator itData = itRow.begin();
    ++itData;
    EXPECT_EQ(data2, itData.val());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatSettingAnExistingElementLetTheOthersUnaffected)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::PopulateCol(matrix);

    typename Matrix::Data data2;
    typename Matrix::Data data3;
    TestHelpers::createData(data2, line1.rowIndex);
    TestHelpers::createData(data3, line1.rowIndex + 1);

    {
        typename Matrix::RowIterator itRow = matrix.writeLine(line1.rowIndex);
        itRow.setCol(line1.data2.index, data2);
        itRow.setCol(line1.data2.index + 1, data3);
        matrix.compress();
    }

    typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);
    typename Matrix::ColConstIterator itData = itRow.begin();
    ++itData;
    EXPECT_EQ(data2, itData.val());
    ++itData;
    EXPECT_EQ(data3, itData.val());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheValueOfAnElementCanBeSummedToTheOneOfAnExistingElementWithTheSameColIndex)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line = TestHelpers::PopulateCol(matrix);

    typename Matrix::Data data2;
    TestHelpers::createData(data2, line.rowIndex);

    {
        typename Matrix::RowIterator itRow = matrix.writeLine(line.rowIndex);
        itRow.addCol(line.data2.index, data2);
        matrix.compress();
    }

    typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
    typename Matrix::ColConstIterator itData = itRow.begin();
    ++itData;
    EXPECT_EQ(line.data2.value + data2, itData.val());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatANewElementIsInsertedIntoARowWhenTryingToSumItsValueToANonExistingOne)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::PopulateCol(matrix);

    typename Matrix::Data data2;
    TestHelpers::createData(data2, line1.rowIndex);
    {
        typename Matrix::RowIterator itRow = matrix.writeLine(line1.rowIndex);
        itRow.addCol(line1.data2.index + 1, data2);
        matrix.compress();
    }

    const Matrix& constMatrix = matrix;
    typename Matrix::RowConstIterator itRow = constMatrix.begin();
    typename Matrix::ColConstIterator itData = itRow.begin();
    ++itData; ++itData;
    EXPECT_EQ(data2, itData.val());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatANewElementIsInsertedIntoARowWhenTryingToSumItsValueToAnotherButTheRowIsEmpty)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    TestHelpers::PopulateCol(matrix, 23);

    typename Matrix::Data data2;
    TestHelpers::createData(data2, 42);
    {
        typename Matrix::RowIterator itRow = matrix.writeLine(42);
        itRow.addCol(123, data2);
        matrix.compress();
    }

    typename Matrix::RowConstIterator itRow = matrix.readLine(42);
    typename Matrix::ColConstIterator itData = itRow.begin();
    EXPECT_EQ(data2, itData.val());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatALineCanBeIdenticallyRepopulatedUsingTheSumFunctionAfterItHasBeenClearedInAOneLineMatrix)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, 42);

    // Clear
    {
        matrix.clearRowBlock(line.rowIndex);
    }

    // Repopulation
    {
        typename Matrix::RowIterator itRow = matrix.writeLine(line.rowIndex);
        itRow.addCol(line.data1.index, line.data1.value);
        itRow.addCol(line.data2.index, line.data2.value);
        matrix.compress();
    }

    typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
    auto itCol = itRow.begin();

    {
        EXPECT_EQ(line.data1.index, itCol.index());
        EXPECT_EQ(line.data1.value, itCol.val());
    }

    {
        itCol++;
        EXPECT_EQ(line.data2.index, itCol.index());
        EXPECT_EQ(line.data2.value, itCol.val());
    }
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTrueIsReturnedWhenComparingTwoColConstIteratorsPointingToTheSameElement)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::PopulateCol(matrix);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);

    typename Matrix::ColConstIterator itDataA = itRow.begin();
    typename Matrix::ColConstIterator itDataB = itDataA;

    EXPECT_TRUE(itDataA == itDataB);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTrueIsReturnedWhenComparingTwoColConstIteratorsPointingToDistinctElements)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::PopulateCol(matrix);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);

    typename Matrix::ColConstIterator itDataA = itRow.begin();
    typename Matrix::ColConstIterator itDataB = itDataA;
    ++itDataB;

    EXPECT_TRUE(itDataA != itDataB);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatPreIncrementingSizeTimesTheColConstIteratorToTheBeginningOfARowResultsInThePastTheEndIterator)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const TestHelpers::line_t<Matrix> line1 = TestHelpers::PopulateCol(matrix);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);
    typename Matrix::ColConstIterator itData = itRow.begin();
    ++itData; ++itData;
    EXPECT_EQ(itRow.end(), itData);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatPostIncrementingSizeTimesTheColConstIteratorToTheBeginningOfARowResultsInThePastTheEndIterator)
{
    typedef TypeParam Matrix;
    Matrix matrix;
    const TestHelpers::line_t<Matrix> line1 = TestHelpers::PopulateCol(matrix);

    typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);
    typename Matrix::ColConstIterator itData = itRow.begin();
    itData++; itData++;
    EXPECT_EQ(itRow.end(), itData);
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatANewLineCanBeInsertedAndItsElementsAccessedAfterTheMatrixHasBeenCleared)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    {
        TestHelpers::WriteLine(matrix, 42);
        matrix.clear();
    }

    const TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(42));

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
        auto itCol = itRow.begin();

        {
            EXPECT_EQ(line.data1.index, itCol.index());
            EXPECT_EQ(line.data1.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line.data2.index, itCol.index());
            EXPECT_EQ(line.data2.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line.data3.index, itCol.index());
            EXPECT_EQ(line.data3.value, itCol.val());
        }
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatDataCanBeRetrievedWhenAllLinesAreCreatedFirstAndThenDataInserted)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    typename Matrix::Data data1;
    typename Matrix::Data data2;
    typename Matrix::Data data3;
    typename Matrix::Data data4;
    TestHelpers::createData(data1, 23);
    TestHelpers::createData(data2, 42);
    TestHelpers::createData(data3, 23 + 1);
    TestHelpers::createData(data4, 42 + 1);

    typename Matrix::RowIterator itRow23 = matrix.writeLine(23);
    typename Matrix::RowIterator itRow42 = matrix.writeLine(42);

    {
        itRow23.setCol(123, data1);
        itRow23.setCol(456, data2);
    }

    {
        itRow42.setCol(789, data3);
        itRow42.setCol(852, data4);
    }
    matrix.compress();

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(23);
        typename Matrix::ColConstIterator itCol = itRow.begin();
        EXPECT_EQ(123u, itCol.index());
        EXPECT_EQ(data1, itCol.val());

        ++itCol;
        EXPECT_EQ(456u, itCol.index());
        EXPECT_EQ(data2, itCol.val());
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(42);
        typename Matrix::ColConstIterator itCol = itRow.begin();
        EXPECT_EQ(789u, itCol.index());
        EXPECT_EQ(data3, itCol.val());

        ++itCol;
        EXPECT_EQ(852u, itCol.index());
        EXPECT_EQ(data4, itCol.val());
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatDataCanBeRetrievedWhenDataAreInsertedImmediatelyAfterLineCreation)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    typename Matrix::Data data1;
    typename Matrix::Data data2;
    typename Matrix::Data data3;
    typename Matrix::Data data4;
    TestHelpers::createData(data1, 23);
    TestHelpers::createData(data2, 42);
    TestHelpers::createData(data3, 23 + 1);
    TestHelpers::createData(data4, 42 + 1);

    {
        typename Matrix::RowIterator itRow = matrix.writeLine(23);

        itRow.setCol(123, data1);
        itRow.setCol(456, data2);
    }

    {
        typename Matrix::RowIterator itRow = matrix.writeLine(42);

        itRow.setCol(789, data3);
        itRow.setCol(852, data4);
    }
    matrix.compress();

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(23);
        typename Matrix::ColConstIterator itCol = itRow.begin();

        EXPECT_EQ(123u, itCol.index());
        EXPECT_EQ(data1, itCol.val());

        ++itCol;
        EXPECT_EQ(456u, itCol.index());
        EXPECT_EQ(data2, itCol.val());
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(42);
        typename Matrix::ColConstIterator itCol = itRow.begin();
        EXPECT_EQ(789u, itCol.index());
        EXPECT_EQ(data3, itCol.val());

        ++itCol;
        EXPECT_EQ(852u, itCol.index());
        EXPECT_EQ(data4, itCol.val());
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatElementsOfALineAreUnaffectedWhenTryingToWriteItWhenItAlreadyExists)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    typename Matrix::Data data1;
    typename Matrix::Data data2;
    typename Matrix::Data data3;
    typename Matrix::Data data4;
    TestHelpers::createData(data1, 23);
    TestHelpers::createData(data2, 42);
    TestHelpers::createData(data3, 23 + 1);
    TestHelpers::createData(data4, 42 + 1);

    {
        typename Matrix::RowIterator itRow23 = matrix.writeLine(23);
        typename Matrix::RowIterator itRow42 = matrix.writeLine(42);

        {
            itRow23.setCol(123, data1);
            itRow23.setCol(456, data2);
        }

        {
            itRow42.setCol(789, data3);
            itRow42.setCol(852, data4);

            matrix.writeLine(42);
            matrix.compress();
        }
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(42);
        typename Matrix::ColConstIterator itCol = itRow.begin();

        EXPECT_EQ(789u, itCol.index());
        EXPECT_EQ(data3, itCol.val());

        ++itCol;
        EXPECT_EQ(852u, itCol.index());
        EXPECT_EQ(data4, itCol.val());
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatElementsCanBeInsertedWhenALineIsCreated)
{
    typedef TypeParam Matrix;
    TestHelpers::line_t<Matrix> line = TestHelpers::nullLine<Matrix>();

    Matrix matrix;
    const typename Matrix::KeyType rowIndex = 23;

    {
        Matrix sourceMatrix;
        line = TestHelpers::WriteLine(sourceMatrix, 42);
        matrix.setLine(rowIndex, sourceMatrix.begin().row());
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
        auto itCol = itRow.begin();

        {
            EXPECT_EQ(line.data1.index, itCol.index());
            EXPECT_EQ(line.data1.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line.data2.index, itCol.index());
            EXPECT_EQ(line.data2.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line.data3.index, itCol.index());
            EXPECT_EQ(line.data3.value, itCol.val());
        }
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheSizeOfALineIsUpdatedWhenItIsReplacedByAnother)
{
    typedef TypeParam Matrix;
    TestHelpers::line_t<Matrix> line = TestHelpers::nullLine<Matrix>();

    Matrix matrix;
    const typename Matrix::KeyType rowIndex = 42;

    {
        TestHelpers::WriteLine(matrix, rowIndex);
    }

    {
        Matrix sourceMatrix;
        line = TestHelpers::WriteLine(sourceMatrix, TestHelpers::GetNextUniqueIndex<Matrix>(rowIndex));
        matrix.setLine(rowIndex, sourceMatrix.begin().row());
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
        EXPECT_EQ(static_cast<std::size_t>(line.initialDataCount), itRow.row().size());
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheExpectedNumberOfElementsCanBeTraversedViaIteratorsWhenALineHasBeenReplacedByAnother)
{
    typedef TypeParam Matrix;
    TestHelpers::line_t<Matrix> line = TestHelpers::nullLine<Matrix>();

    Matrix matrix;
    const typename Matrix::KeyType rowIndex = 42;

    {
        TestHelpers::WriteLine(matrix, rowIndex);
    }

    {
        Matrix sourceMatrix;
        line = TestHelpers::WriteLine(sourceMatrix, TestHelpers::GetNextUniqueIndex<Matrix>(rowIndex));
        matrix.setLine(rowIndex, sourceMatrix.begin().row());
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
        typename Matrix::ColConstIterator itData = itRow.begin();
        for (unsigned int i = 0; i < line.initialDataCount; i++)
        {
            SOFA_ASSERT(itData != itRow.end());
            ++itData;
        }
        EXPECT_EQ(itRow.end(), itData);
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheExpectedElementsCanBeTraversedViaIteratorsWhenALineHasBeenReplacedByAnother)
{
    typedef TypeParam Matrix;
    TestHelpers::line_t<Matrix> line = TestHelpers::nullLine<Matrix>();

    Matrix matrix;
    const typename Matrix::KeyType rowIndex = 42;

    {
        TestHelpers::WriteLine(matrix, rowIndex);
    }

    {
        Matrix sourceMatrix;
        line = TestHelpers::WriteLine(sourceMatrix, TestHelpers::GetNextUniqueIndex<Matrix>(rowIndex));
        matrix.setLine(rowIndex, sourceMatrix.begin().row());
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
        typename Matrix::ColConstIterator itData = itRow.begin();

        {
            EXPECT_EQ(line.data1.index, itData.index());
            EXPECT_EQ(line.data1.value, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(line.data2.index, itData.index());
            EXPECT_EQ(line.data2.value, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(line.data3.index, itData.index());
            EXPECT_EQ(line.data3.value, itData.val());
        }
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatExistingLinesStayUnaffectedWhenALineIsReplacedByAnother)
{
    typedef TypeParam Matrix;
    TestHelpers::line_t<Matrix> line = TestHelpers::nullLine<Matrix>();

    Matrix matrix;
    const typename Matrix::KeyType rowIndex = 42;

    {
        TestHelpers::WriteLine(matrix, rowIndex);
        line = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(rowIndex));
    }

    {
        Matrix sourceMatrix;
        TestHelpers::WriteLine(sourceMatrix, TestHelpers::GetNextUniqueIndex<Matrix>(line.rowIndex));
        matrix.setLine(rowIndex, sourceMatrix.begin().row());
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(line.rowIndex);
        typename Matrix::ColConstIterator itData = itRow.begin();

        {
            EXPECT_EQ(line.data1.index, itData.index());
            EXPECT_EQ(line.data1.value, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(line.data2.index, itData.index());
            EXPECT_EQ(line.data2.value, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(line.data3.index, itData.index());
            EXPECT_EQ(line.data3.value, itData.val());
        }
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheContentOfALineCanBeSummedWithTheContentOfAnother)
{
    typedef TypeParam Matrix;
    TestHelpers::line_t<Matrix> sourceLine = TestHelpers::nullLine<Matrix>();

    Matrix matrix;
    const typename Matrix::KeyType rowIndex = 23;
    TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, rowIndex);

    {
        Matrix sourceMatrix;
        sourceLine = TestHelpers::WriteLine(sourceMatrix, 42, rowIndex + 1);
        matrix.addLine(rowIndex, sourceMatrix.begin().row());
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
        typename Matrix::ColConstIterator itData = itRow.begin();

        {
            EXPECT_EQ(line.data1.index, itData.index());
            EXPECT_EQ(line.data1.value + sourceLine.data1.value, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(line.data2.index, itData.index());
            EXPECT_EQ(line.data2.value + sourceLine.data2.value, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(line.data3.index, itData.index());
            EXPECT_EQ(line.data3.value + sourceLine.data3.value, itData.val());
        }
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheSizeOfALineIsUpdatedWhenNewElementsAreCreatedWhenTryingToSumALineToAnotherWithNoCommonColIndices)
{
    typedef TypeParam Matrix;
    TestHelpers::line_t<Matrix> sourceLine = TestHelpers::nullLine<Matrix>();

    Matrix matrix;
    const typename Matrix::KeyType rowIndex = 23;

    TestHelpers::line_t<Matrix> line = TestHelpers::WriteLine(matrix, rowIndex);

    {
        Matrix sourceMatrix;
        sourceLine = TestHelpers::WriteLine(sourceMatrix, 42, TestHelpers::GetNextUniqueIndex<Matrix>(rowIndex));

        matrix.addLine(rowIndex, sourceMatrix.begin().row());
    }

    typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
    EXPECT_EQ(line.initialDataCount * 2, itRow.row().size());
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatNewElementsAreCreatedWhenTryingToSumALineToAnotherWithNoCommonColIndices)
{
    typedef TypeParam Matrix;
    TestHelpers::line_t<Matrix> sourceLine = TestHelpers::nullLine<Matrix>();

    Matrix matrix;
    const typename Matrix::KeyType rowIndex = 23;

    {
        Matrix sourceMatrix;
        sourceLine = TestHelpers::WriteLine(sourceMatrix, 42, TestHelpers::GetNextUniqueIndex<Matrix>(rowIndex));
        matrix.addLine(rowIndex, sourceMatrix.begin().row());
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
        typename Matrix::ColConstIterator itData = itRow.begin();

        {
            EXPECT_EQ(sourceLine.data1.index, itData.index());
            EXPECT_EQ(sourceLine.data1.value, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(sourceLine.data2.index, itData.index());
            EXPECT_EQ(sourceLine.data2.value, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(sourceLine.data3.index, itData.index());
            EXPECT_EQ(sourceLine.data3.value, itData.val());
        }
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheSizeOfALineIsUpdatedWhenNewElementsAreCreatedWhenTryingToSumALineToAnotherWithCommonAndDistinctIndices)
{
    typedef TypeParam Matrix;
    TestHelpers::line_t<Matrix> sourceLine = TestHelpers::nullLine<Matrix>();

    Matrix matrix;
    const typename Matrix::KeyType rowIndex = 23;

    {
        Matrix sourceMatrix;
        sourceLine = TestHelpers::WriteLine(sourceMatrix, 42);

        typename Matrix::RowIterator itRow = matrix.writeLine(rowIndex);
        itRow.setCol(TestHelpers::GetNextUniqueIndex<Matrix>(rowIndex), sourceLine.data2.value);
        matrix.addLine(rowIndex, sourceMatrix.begin().row());
    }

    typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
    EXPECT_EQ(sourceLine.initialDataCount + 1, itRow.row().size());
}

//////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatTheContentOfALineRemainsSortedByColIndex)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    typename Matrix::RowIterator itRow = matrix.writeLine(42);

    {
        typename Matrix::Data data1;
        TestHelpers::createData(data1, 23);
        itRow.setCol(123, data1);
    }

    {
        typename Matrix::Data data2;
        TestHelpers::createData(data2, 23 + 1);
        itRow.setCol(789, data2);
    }

    {
        typename Matrix::Data data3;
        TestHelpers::createData(data3, 23 + 2);
        itRow.setCol(456, data3);
    }

}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatElementsHaveTheExpectedValuesWhenTryingToSumALineToAnotherWithNoCommonColIndices)
{
    typedef TypeParam Matrix;
    TestHelpers::line_t<Matrix> sourceLine = TestHelpers::nullLine<Matrix>();

    Matrix matrix;
    const typename Matrix::KeyType rowIndex = 23;
    typename Matrix::Data data1;
    TestHelpers::createData(data1, 0);

    {
        Matrix sourceMatrix;
        sourceLine = TestHelpers::WriteLine(sourceMatrix, 42);

        typename Matrix::RowIterator itRow = matrix.writeLine(rowIndex);
        itRow.setCol(1u, data1);
        matrix.compress();

        matrix.addLine(rowIndex, sourceMatrix.begin().row());
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(rowIndex);
        typename Matrix::ColConstIterator itData = itRow.begin();

        {
            EXPECT_EQ(1u, itData.index());
            EXPECT_EQ(data1, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(sourceLine.data1.index, itData.index());
            EXPECT_EQ(sourceLine.data1.value, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(sourceLine.data2.index, itData.index());
            EXPECT_EQ(sourceLine.data2.value, itData.val());
        }

        ++itData;
        {
            EXPECT_EQ(sourceLine.data3.index, itData.index());
            EXPECT_EQ(sourceLine.data3.value, itData.val());
        }
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatDataCanBeRetrievedWhenCreatingAndFillingLinesInDescendingIndexOrder)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    TestHelpers::line_t<Matrix> line1 = TestHelpers::WriteLine(matrix, TestHelpers::GetNextUniqueIndex<Matrix>(42));
    TestHelpers::line_t<Matrix> line2 = TestHelpers::WriteLine(matrix, 42);

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);
        auto itCol = itRow.begin();

        {
            EXPECT_EQ(line1.data1.index, itCol.index());
            EXPECT_EQ(line1.data1.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line1.data2.index, itCol.index());
            EXPECT_EQ(line1.data2.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line1.data3.index, itCol.index());
            EXPECT_EQ(line1.data3.value, itCol.val());
        }
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(line2.rowIndex);
        auto itCol = itRow.begin();

        {
            EXPECT_EQ(line2.data1.index, itCol.index());
            EXPECT_EQ(line2.data1.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line2.data2.index, itCol.index());
            EXPECT_EQ(line2.data2.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line2.data3.index, itCol.index());
            EXPECT_EQ(line2.data3.value, itCol.val());
        }
    }
}

////////////////////////////////////////////////////
TYPED_TEST(SparseMatrixTest, CheckThatDataCanBeRetrievedWhenCreatingLinesInDescendingIndexOrderAndInsertingADataInTheFirstOneAfterwards)
{
    typedef TypeParam Matrix;
    Matrix matrix;

    const typename Matrix::KeyType rowIndex2 = 42;
    const typename Matrix::KeyType rowIndex1 = TestHelpers::GetNextUniqueIndex<Matrix>(rowIndex2);

    TestHelpers::line_t<Matrix> line1 = TestHelpers::WriteLine(matrix, rowIndex1);
    TestHelpers::line_t<Matrix> line2 = TestHelpers::WriteLine(matrix, rowIndex2);

    {
        typename Matrix::RowIterator itRow = matrix.writeLine(rowIndex1);
        itRow.setCol(1u, line2.data3.value * 2);
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(line1.rowIndex);
        auto itCol = itRow.begin();

        {
            EXPECT_EQ(1u, itCol.index());
            EXPECT_EQ(line2.data3.value * 2, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line1.data1.index, itCol.index());
            EXPECT_EQ(line1.data1.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line1.data2.index, itCol.index());
            EXPECT_EQ(line1.data2.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line1.data3.index, itCol.index());
            EXPECT_EQ(line1.data3.value, itCol.val());
        }
    }

    {
        typename Matrix::RowConstIterator itRow = matrix.readLine(line2.rowIndex);
        auto itCol = itRow.begin();

        {
            EXPECT_EQ(line2.data1.index, itCol.index());
            EXPECT_EQ(line2.data1.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line2.data2.index, itCol.index());
            EXPECT_EQ(line2.data2.value, itCol.val());
        }

        {
            itCol++;
            EXPECT_EQ(line2.data3.index, itCol.index());
            EXPECT_EQ(line2.data3.value, itCol.val());
        }
    }
}


template <typename TMatrix>
struct CompressedRowSparseMatrixConstraintTest : public TMatrix, ::testing::Test
{
};

typedef ::testing::Types<
    sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::defaulttype::Vec1Types::Deriv, CRSConstraintTestPolicy>,
    sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::defaulttype::Vec2Types::Deriv, CRSConstraintTestPolicy>,
    sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::defaulttype::Vec3Types::Deriv, CRSConstraintTestPolicy>,
    sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::defaulttype::Rigid2Types::Deriv, CRSConstraintTestPolicy>,
    sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::defaulttype::Rigid3Types::Deriv, CRSConstraintTestPolicy>
> CompressedRowSparseMatrixConstraintTestTypes;

TYPED_TEST_CASE(CompressedRowSparseMatrixConstraintTest, CompressedRowSparseMatrixConstraintTestTypes);


TYPED_TEST(CompressedRowSparseMatrixConstraintTest, checkRowDeletionUntilMatrixIsEmpty)
{
    typedef TypeParam ConstraintMatrix;
    ConstraintMatrix& matrix = *this;

    typedef typename ConstraintMatrix::Block Block;

    // initialize matrix with some arbitrary easy to track values
    Block bEmpty;
    Block bOne;
    Block bTwo;
    Block bThree;
    for (std::size_t i=0; i<Block::size(); ++i)
    {
        bOne[i] = 1;
        bTwo[i] = 2;
        bThree[i] = 3;
    }

    matrix.resizeBlock(3, 4);

    matrix.setBlock(0, 0, bOne);
    matrix.setBlock(0, 2, bOne);
    matrix.setBlock(0, 3, bOne);

    matrix.setBlock(1, 1, bTwo);

    matrix.setBlock(2, 1, bThree);
    matrix.setBlock(2, 3, bThree);

    matrix.compress();

    this->deleteRow(1); // need to use "this" since deleteRow is protected

    EXPECT_EQ(matrix.rowBSize(), 3);
    EXPECT_EQ(matrix.colBSize(), 4);

    ASSERT_EQ(matrix.getBlock(0, 0), bOne);
    ASSERT_EQ(matrix.getBlock(0, 2), bOne);
    ASSERT_EQ(matrix.getBlock(0, 3), bOne);

    ASSERT_EQ(matrix.getBlock(1, 1), bEmpty);

    ASSERT_EQ(matrix.getBlock(2, 1), bThree);
    ASSERT_EQ(matrix.getBlock(2, 3), bThree);

    this->deleteRow(0); // need to use "this" since deleteRow is protected

    EXPECT_EQ(matrix.rowBSize(), 3);
    EXPECT_EQ(matrix.colBSize(), 4);

    ASSERT_EQ(matrix.getBlock(0, 0), bEmpty);
    ASSERT_EQ(matrix.getBlock(0, 2), bEmpty);
    ASSERT_EQ(matrix.getBlock(0, 3), bEmpty);

    ASSERT_EQ(matrix.getBlock(1, 1), bEmpty);

    ASSERT_EQ(matrix.getBlock(2, 1), bThree);
    ASSERT_EQ(matrix.getBlock(2, 3), bThree);

    this->deleteRow(0); // need to use "this" since deleteRow is protected

    // from now on there should be nothing left in the matrix
    EXPECT_EQ(matrix.rowBSize(), 0);
    EXPECT_EQ(matrix.colBSize(), 0);

    ASSERT_EQ(matrix.getBlock(0, 0), bEmpty);
    ASSERT_EQ(matrix.getBlock(0, 2), bEmpty);
    ASSERT_EQ(matrix.getBlock(0, 3), bEmpty);

    ASSERT_EQ(matrix.getBlock(1, 1), bEmpty);

    ASSERT_EQ(matrix.getBlock(2, 1), bEmpty);
    ASSERT_EQ(matrix.getBlock(2, 3), bEmpty);

    ASSERT_TRUE(matrix.colsValue.empty());
    ASSERT_TRUE(matrix.colsIndex.empty());
    ASSERT_TRUE(matrix.rowIndex.empty());
    ASSERT_TRUE(matrix.rowBegin.empty());

}

TEST(CompressedRowSparseMatrixTest, checkTransposition)
{
    using Block      = sofa::type::Mat3x3d;
    using BSRMatrix = sofa::linearalgebra::CompressedRowSparseMatrixGeneric< Block >;


    Block b;
    for (std::size_t i=0; i<3; ++i)
    {
        b(i, 0) = i+1;
        b(i, 1) = i+1;
        b(i, 2) = i+1;
    }
    

    BSRMatrix matrix;

    matrix.resizeBlock(4, 5);

    matrix.addBlock(1, 1, b);
    matrix.addBlock(1, 2, b);
    matrix.addBlock(2, 2, b);
    
    matrix.compress();
    matrix.fullRows();
    BSRMatrix matrixTranspose;
    matrix.transposeFullRows(matrixTranspose);

    ASSERT_EQ(matrixTranspose.nBlockRow, 5);
    ASSERT_EQ(matrixTranspose.nBlockCol, 4);
    
    ASSERT_EQ(matrixTranspose.getBlock(1, 1), b);
    ASSERT_EQ(matrixTranspose.getBlock(2, 1), b);
    ASSERT_EQ(matrixTranspose.getBlock(2, 2), b);
}

template<typename TBlock>
void generateMatrix(sofa::linearalgebra::CompressedRowSparseMatrixConstraint<TBlock>& matrix,
    sofa::SignedIndex nbRows, sofa::SignedIndex nbCols,
    typename sofa::linearalgebra::CompressedRowSparseMatrixConstraint<TBlock>::Real sparsity,
    long seed)
{
    using Matrix = sofa::linearalgebra::CompressedRowSparseMatrixConstraint<TBlock>;
    using Real = typename Matrix::Real;
    const auto nbNonZero = static_cast<sofa::SignedIndex>(sparsity * static_cast<Real>(nbRows*nbCols));

    sofa::testing::LinearCongruentialRandomGenerator lcg(seed);

    matrix.resizeBlock(nbRows / Matrix::NL, nbCols / Matrix::NC);

    for (sofa::SignedIndex i = 0; i < nbNonZero; ++i)
    {
        const auto value = lcg.generateInUnitRange<Real>();
        const auto row = static_cast<sofa::Index>(lcg.generateInRange(0., nbRows));
        const auto col = static_cast<sofa::Index>(lcg.generateInRange(0., nbCols));

        auto line = matrix.writeLine(row);
        TBlock block;
        block[col % Matrix::NC] = value;
        line.addCol(col, block);
    }
    matrix.compress();
}

TEST(CompressedRowSparseMatrixConstraint, emptyMatrixGetRowRange)
{
    EXPECT_EQ(sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal>::s_invalidIndex, std::numeric_limits<sofa::SignedIndex>::lowest());

    const sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal> A;

    const auto begin = A.begin();
    EXPECT_EQ(begin.getInternal(), sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal>::s_invalidIndex);

    const auto end = A.end();
    EXPECT_EQ(end.getInternal(), sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal>::s_invalidIndex);

    EXPECT_EQ(begin, end);

    const auto checkIterator = [](const auto& iterator)
    {
        const auto itBegin = iterator.begin();
        const auto itEnd = iterator.end();

        EXPECT_EQ(itBegin.getInternal(), sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal>::s_invalidIndex);
        EXPECT_EQ(itEnd.getInternal(), sofa::linearalgebra::CompressedRowSparseMatrixConstraint<SReal>::s_invalidIndex);
        EXPECT_EQ(itBegin, itEnd);
    };

    checkIterator(begin);
    checkIterator(end);
}

TEST(CompressedRowSparseMatrixConstraint, ostream)
{
    sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::type::Vec3> A;

    generateMatrix(A, 5, 15, 0.1, 7);

    std::stringstream ss;
    ss << A;

    static const std::string expectedOutput =
R"(4 0 2 1 0 0.360985 0 12 0.926981 0 0 2 1 9 0.451858 0 0 3 1 6 0.777417 0 0 4 3 5 0 0 0.474108 7 0 0.983937 0 9 0.238781 0 0 )";

    EXPECT_EQ(ss.str(), expectedOutput);
}

TEST(CompressedRowSparseMatrixConstraint, prettyPrint)
{
    sofa::linearalgebra::CompressedRowSparseMatrixConstraint<sofa::type::Vec3> A;

    generateMatrix(A, 5, 15, 0.1, 7);

    static const std::string expectedOutput =
R"(Constraint ID : 0  dof ID : 1  value : 0 0.360985 0  dof ID : 12  value : 0.926981 0 0
Constraint ID : 2  dof ID : 9  value : 0.451858 0 0
Constraint ID : 3  dof ID : 6  value : 0.777417 0 0
Constraint ID : 4  dof ID : 5  value : 0 0 0.474108  dof ID : 7  value : 0 0.983937 0  dof ID : 9  value : 0.238781 0 0
)";
    
    std::ostringstream oss;
    A.prettyPrint(oss);

    EXPECT_EQ(oss.str(), expectedOutput);
}

} // namespace sofa
