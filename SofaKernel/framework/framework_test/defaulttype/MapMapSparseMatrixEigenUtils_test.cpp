/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <gtest/gtest.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/defaulttype/MapMapSparseMatrixEigenUtils.h>
#include <Eigen/Sparse>
#include <sofa/defaulttype/Vec3Types.h>

namespace
{


TEST(MapMapSparseMatrixEigenUtilsTest, checkEigenSparseMatrixLowLeveAPI)
{
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> EigenSparseMatrix;

    EigenSparseMatrix mat(5,5);

    std::vector< Eigen::Triplet<double> > matEntries =
    {
        {0,0,0}, {0,1,3},
        {1,0,22}, {1,4,17},
        {2,0,5}, {2,1,5}, {2,4,1},
        {4,2,14}, {4,4,8}
    };

    mat.setFromTriplets(matEntries.begin(), matEntries.end());

    mat.makeCompressed();

    int*    outerIndexPtr = mat.outerIndexPtr();
    int*    innerIndexPtr = mat.innerIndexPtr();
    double* valuePtr = mat.valuePtr();


    int nonZero_0 = *(outerIndexPtr + 0+1) - *(outerIndexPtr+0);
    EXPECT_EQ(2, nonZero_0);
    
    int nonZero_1 = *(outerIndexPtr + 1+1) - *(outerIndexPtr+1);
    EXPECT_EQ(2, nonZero_1);

    int nonZero_2 = *(outerIndexPtr + 2 + 1) - *(outerIndexPtr + 2);
    EXPECT_EQ(3, nonZero_2);

    int nonZero_3 = *(outerIndexPtr + 3 + 1) - *(outerIndexPtr + 3);
    EXPECT_EQ(0, nonZero_3);

    int nonZero_4 = *(outerIndexPtr + 4 + 1) - *(outerIndexPtr + 4);
    EXPECT_EQ(2, nonZero_4);

    EXPECT_EQ(0,*(innerIndexPtr + 0));
    EXPECT_EQ(1,*(innerIndexPtr + 1));
    EXPECT_EQ(0,*(innerIndexPtr + 2));
    EXPECT_EQ(4,*(innerIndexPtr + 3));
    EXPECT_EQ(0,*(innerIndexPtr + 4));
    EXPECT_EQ(1,*(innerIndexPtr + 5));
    EXPECT_EQ(4,*(innerIndexPtr + 6));
    EXPECT_EQ(2,*(innerIndexPtr + 7));
    EXPECT_EQ(4,*(innerIndexPtr + 8));

    EXPECT_EQ(matEntries.size(), mat.nonZeros());
    
    for (int i = 0; i < mat.nonZeros(); ++i)
    {
        double value = *(valuePtr + i);
        EXPECT_EQ(matEntries[i].value(), value);
    }
}


TEST(MapMapSparseMatrixEigenUtilsTest, checkConversionEigenSparseMapMapSparseVec1d)
{
    typedef sofa::defaulttype::Vec<1, double > TVec;

    typedef sofa::defaulttype::EigenSparseToMapMapSparseMatrix< TVec > EigenSparseToMapMapSparseVec1d;
    EigenSparseToMapMapSparseVec1d eigenSparseToMapMapVec1d;
    typedef typename EigenSparseToMapMapSparseVec1d::EigenSparseMatrix EigenSparseMatrix;
    typedef typename EigenSparseToMapMapSparseVec1d::TMapMapSparseMatrix TMapMapSparseMatrix;

    std::vector< Eigen::Triplet<double> > matEntries =
    {
        { 0,0,0 },{ 0,1,3 },
        { 1,0,22 },{ 1,4,17 },
        { 2,0,5 },{ 2,1,5 },{ 2,4,1 },
        { 4,2,14 },{ 4,4,8 }
    };

    EigenSparseMatrix eigenMat(5,5);
    eigenMat.setFromTriplets(matEntries.begin(),matEntries.end() );


    TMapMapSparseMatrix mat = eigenSparseToMapMapVec1d(eigenMat);

    int indexEntry = 0;
    for (auto row = mat.begin(); row != mat.end(); ++row)
    {

        for (auto col = row.begin(); col != row.end(); ++col)
        {
            EXPECT_EQ( matEntries[indexEntry++].value(), col.val()[0]  );
        }
    }

    EXPECT_EQ(matEntries.size(), indexEntry);
}


TEST(MapMapSparseMatrixEigenUtilsTest, checkConversionEigenSparseMapMapSparseVec3d)
{
    typedef  sofa::defaulttype::Vec<3, double > TVec;
    typedef sofa::defaulttype::EigenSparseToMapMapSparseMatrix< TVec > EigenSparseToMapMapSparseVec3d;
    EigenSparseToMapMapSparseVec3d eigenSparseToMapMapVec3d;
    typedef typename EigenSparseToMapMapSparseVec3d::EigenSparseMatrix EigenSparseMatrix;
    typedef typename EigenSparseToMapMapSparseVec3d::TMapMapSparseMatrix TMapMapSparseMatrix;


    EigenSparseMatrix eigenMat(12, 12);
    std::vector< Eigen::Triplet<double> > matEntries =
    {
        { 3,3,0.1 },{ 3,4,0.2 },{ 3,5,0.3 }
    };

    eigenMat.setFromTriplets(matEntries.begin(), matEntries.end());

    TMapMapSparseMatrix mat = eigenSparseToMapMapVec3d(eigenMat);

    int indexEntry = 0;
    for (auto row = mat.begin(); row != mat.end(); ++row)
    {
        for (auto col = row.begin(); col != row.end(); ++col)
        {
            for (std::size_t i = 0; i < TVec::total_size; ++i)
            {
                EXPECT_EQ(matEntries[indexEntry++].value(), col.val()[i]);
            }
        }
    }

    EXPECT_EQ(matEntries.size(), indexEntry);
}


TEST(MapMapSparseMatrixEigenUtilsTest, checkConversionMapMapSparseVec1dEigenSparse)
{
    typedef sofa::defaulttype::Vec<1, double > TVec;
    typedef sofa::defaulttype::MapMapSparseMatrixToEigenSparse< TVec > MapMapSparseMatrixToEigenSparseVec1d;
    MapMapSparseMatrixToEigenSparseVec1d mapmapSparseToEigenSparse;
    typedef typename MapMapSparseMatrixToEigenSparseVec1d::EigenSparseMatrix EigenSparseMatrix;
    typedef typename MapMapSparseMatrixToEigenSparseVec1d::TMapMapSparseMatrix TMapMapSparseMatrix;

    TMapMapSparseMatrix mat;

    std::vector< Eigen::Triplet<double> > matEntries =
    {
        { 0,0,0 },{ 0,1,3 },
        { 1,0,22 },{ 1,4,17 },
        { 2,0,5 },{ 2,1,5 },{ 2,4,1 },
        { 4,2,14 },{ 4,4,8 }
    };

    auto it = matEntries.begin();

    while(it != matEntries.end())
    {
        int row = it->row();
        int col = it->col();
        TVec vec;
        for (std::size_t i = 0; i < TVec::size(); ++i)
        {
            vec[i] = it->value();
            ++it;
        }

        mat.writeLine(row).setCol(col,vec);
    }

    EigenSparseMatrix eigenMat = mapmapSparseToEigenSparse(mat, 5);

    for (auto row = mat.begin(); row != mat.end(); ++row)
    {
        for (auto col = row.begin(); col != row.end(); ++col)
        {
            for (std::size_t i = 0; i < TVec::size(); ++i)
            {
                EXPECT_EQ(col.val()[i], eigenMat.coeff(row.index(), col.index()+i));
            }
        }
    }

    EXPECT_EQ(matEntries.size(), eigenMat.nonZeros());
}


TEST(MapMapSparseMatrixEigenUtilsTest, checkConversionMapMapSparseVec3dEigenSparse)
{
    typedef sofa::defaulttype::Vec<1, double > TVec;
    typedef sofa::defaulttype::MapMapSparseMatrixToEigenSparse< TVec > checkConversionMapMapSparseVec3dEigenSparse;
    checkConversionMapMapSparseVec3dEigenSparse mapmapSparseToEigenSparse;
    typedef typename checkConversionMapMapSparseVec3dEigenSparse::EigenSparseMatrix EigenSparseMatrix;
    typedef typename checkConversionMapMapSparseVec3dEigenSparse::TMapMapSparseMatrix TMapMapSparseMatrix;

    TMapMapSparseMatrix mat;

    std::vector< Eigen::Triplet<double> > matEntries =
    {
        { 3,3,0.1 },{ 3,4,0.2 },{ 3,5,0.3 }
    };


    auto it = matEntries.begin();

    while (it != matEntries.end())
    {
        int row = it->row();
        int col = it->col();
        TVec vec;
        for (std::size_t i = 0; i < TVec::size(); ++i)
        {
            vec[i] = it->value();
            ++it;
        }

        mat.writeLine(row).setCol(col, vec);
    }

    EigenSparseMatrix eigenMat = mapmapSparseToEigenSparse(mat, 12);

    for (auto row = mat.begin(); row != mat.end(); ++row)
    {
        for (auto col = row.begin(); col != row.end(); ++col)
        {
            for (std::size_t i = 0; i < TVec::size(); ++i)
            {
                EXPECT_EQ(col.val()[i], eigenMat.coeff(row.index(), col.index()+i));
            }
        }
    }

    EXPECT_EQ(matEntries.size(), eigenMat.nonZeros());
}


TEST(MapMapSparseMatrixEigenUtilsTest, checkAddMultTransposeEigenForCumulativeWrite)
{
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> EigenSparseMatrix;

    EigenSparseMatrix jacobian(3, 6); // as in rigid mapping block

    std::vector< Eigen::Triplet<double> > matEntries =
    {
        { 0,0,1 }, { 1,1,1 }, {2,2,1}, // identity block

        { 0,4,2 }, {0, 5, -1}, // skew symmetric block
        { 1,3,-2}, {1, 5, -4},
        { 2,3,1},  {2,4,4}
    };

    jacobian.setFromTriplets(matEntries.begin(), matEntries.end());
    jacobian.makeCompressed();


    //std::cout << jacobian << std::endl;

    typedef sofa::defaulttype::Rigid3Types::Deriv RigidDeriv;
    typedef sofa::defaulttype::Vec3Types::Deriv  Deriv;
    typedef sofa::defaulttype::MapMapSparseMatrix<Deriv> RhsType;
    typedef sofa::defaulttype::MapMapSparseMatrix<RigidDeriv> LhsType;

    LhsType lhs;

    {
        RhsType rhs;
        auto col = rhs.writeLine(0);
        col.addCol(0, sofa::defaulttype::Vec3d(1, 0, 0));
        sofa::defaulttype::addMultTransposeEigen(lhs, jacobian, rhs);
    }

    {
        RhsType rhs;
        auto col = rhs.writeLine(1);
        col.addCol(0, sofa::defaulttype::Vec3d(0, 1, 0));
        sofa::defaulttype::addMultTransposeEigen(lhs, jacobian, rhs);
    }

    {
        RhsType rhs;
        auto col = rhs.writeLine(2);
        col.addCol(0, sofa::defaulttype::Vec3d(0, 0, 1));
        sofa::defaulttype::addMultTransposeEigen(lhs, jacobian, rhs);
    }

    std::cout << lhs << std::endl;

    for (auto row = lhs.begin(), rowEnd = lhs.end(); row != rowEnd; ++row)
    {
        for (auto col = row.begin(), colEnd = row.end(); col != colEnd; ++col)
        {
            for (std::size_t i = 0; i < RigidDeriv::total_size; ++i)
            {
                EXPECT_EQ(col.val()[i], jacobian.coeff(row.index(), col.index()*RigidDeriv::total_size + i ));
            }
        }

    }
}


}