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
#include <sofa/testing/BaseTest.h>
#include <sofa/component/linearsolver/direct/SparseLDLSolver.h>
#include <sofa/component/linearsolver/direct/SparseCommon.h>

#include <sofa/testing/NumericTest.h>

extern "C" {
    #include <metis.h>
}

/**
 * This test is a substep of the SparseLDLSolver.MatrixFactorization test
 * It verifies a call of a METIS function.
 * This function relies on random number generation. In order for the test to be portable accross different
 * architectures, USE_GKRAND must be defined.
 */
TEST(Metis, permutation)
{
    int n = 15;

    //input
    sofa::type::vector<int> xadj
    {  0, 0, 0, 0, 3, 6, 7, 10, 14, 16, 16, 18, 20, 20, 21, 22 };
    sofa::type::vector<int> adj
    { 4,6,7,3,6,7,8,3,4,7,3,4,6,10,5,11,7,13,8,14,10,11 };

    //output
    sofa::type::vector<int> perm(n);
    sofa::type::vector<int> invperm(n);

    auto res = METIS_NodeND(&n , xadj.data(), adj.data(), nullptr, nullptr, perm.data(),invperm.data());
    EXPECT_EQ(res, METIS_OK);

    sofa::type::vector<int> expectedPerm
    { 14, 12, 5, 2, 0, 11, 8, 13, 9, 1, 10, 6, 3, 4, 7 };
    sofa::type::vector<int> expectedInvPerm
    { 4, 9, 3, 12, 13, 2, 11, 14, 6, 8, 10, 5, 1, 7, 0 };

    EXPECT_EQ(perm, expectedPerm);
    EXPECT_EQ(invperm, expectedInvPerm);
}

/**
 * Test of the factorization of a small matrix.
 * The input matrix is extracted from a pendulum problem.
 * This test is valid only if USE_GKRAND is defined
 */
TEST(SparseLDLSolver, MatrixFactorization)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    MatrixType matrix;

    matrix.resize(15, 15);

    matrix.add(0, 0, 1);
    matrix.add(1, 1, 1);
    matrix.add(2, 2, 1);
    matrix.add(3, 3, 0.0002500000050388081);
    matrix.add(3, 4, 4.199174733595127e-10);
    matrix.add(3, 6, -7.832191102079255e-21);
    matrix.add(3, 7, -1.399845452902599e-14);
    matrix.add(4, 3, 4.199174733595127e-10);
    matrix.add(4, 4, 0.0002500599908095969);
    matrix.add(4, 6, -1.399845452902599e-14);
    matrix.add(4, 7, -2.499862007711381e-08);
    matrix.add(5, 5, 0.0002500599958484051);
    matrix.add(5, 8, -2.499862007712165e-08);
    matrix.add(6, 3, -7.832191102079255e-21);
    matrix.add(6, 4, -1.399845452902599e-14);
    matrix.add(6, 6, 0.00025);
    matrix.add(6, 7, 1.399845452902599e-14);
    matrix.add(7, 3, -1.399845452902599e-14);
    matrix.add(7, 4, -2.499862007711381e-08);
    matrix.add(7, 6, 1.399845452902599e-14);
    matrix.add(7, 7, 0.0002500399977921234);
    matrix.add(7, 10, -1.499917204627533e-08);
    matrix.add(8, 5, -2.499862007712165e-08);
    matrix.add(8, 8, 0.0002500399977921234);
    matrix.add(8, 11, -1.499917204627533e-08);
    matrix.add(9, 9, 0.00025);
    matrix.add(10, 7, -1.499917204627533e-08);
    matrix.add(10, 10, 0.0002500199988960617);
    matrix.add(10, 13, -4.999724015425112e-09);
    matrix.add(11, 8, -1.499917204627533e-08);
    matrix.add(11, 11, 0.0002500199988960617);
    matrix.add(11, 14, -4.999724015425112e-09);
    matrix.add(12, 12, 0.000125);
    matrix.add(13, 10, -4.999724015425112e-09);
    matrix.add(13, 13, 0.0001250049997240154);
    matrix.add(14, 11, -4.999724015425112e-09);
    matrix.add(14, 14, 0.0001250049997240154);

    matrix.compress();

    int * M_colptr = (int *) &matrix.getRowBegin()[0];
    int * M_rowind = (int *) &matrix.getColsIndex()[0];

    sofa::type::vector<int> adj, xadj, t_adj, t_xadj, tran_countvec;
    sofa::component::linearsolver::direct::csrToAdj( 15,
        M_colptr, M_rowind, adj, xadj, t_adj, t_xadj, tran_countvec );

    sofa::type::vector<int> expected_xadj
    {  0, 0, 0, 0, 3, 6, 7, 10, 14, 16, 16, 18, 20, 20, 21, 22 };
    sofa::type::vector<int> expected_adj
    { 4,6,7,3,6,7,8,3,4,7,3,4,6,10,5,11,7,13,8,14,10,11 };

    EXPECT_EQ(xadj, expected_xadj);
    EXPECT_EQ(adj, expected_adj);


    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, sofa::linearalgebra::FullVector<SReal> >;
    Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    solver->init();

    solver->invert(matrix);

    auto* genericInvertData = solver->getMatrixInvertData(&matrix);
    EXPECT_NE(genericInvertData, nullptr);

    using InvertDataType = Solver::InvertData;
    auto* invertData = dynamic_cast<InvertDataType*>(genericInvertData);
    EXPECT_NE(invertData, nullptr);

    static const sofa::type::vector<int> expected_perm_Values {
        14, 12, 5, 2, 0, 11, 8, 13, 9, 1, 10, 6, 3, 4, 7
    };
    EXPECT_EQ(invertData->perm.size(), expected_perm_Values.size());
    for (std::size_t i = 0; i < invertData->perm.size(); ++i)
    {
        EXPECT_EQ(invertData->perm[i], expected_perm_Values[i]);
    }

    static const sofa::type::vector<SReal> expected_L_Values {
        -3.9996192364012998418654198928834e-05, -9.9970489051262180859626360618364e-05, -5.9991889146865234069914279979585e-05, -3.9996192364012998418654198928834e-05, -5.9991889146865234069914279979585e-05, -3.1328764408317019078016126538053e-17, -5.5993818116103955171753428679725e-11, 5.5993818116103955171753428679725e-11, 1.6796698595839146789081026273083e-06, -5.599381698753556925340375666935e-11, -9.9970491065863868913876633115478e-05
    };

    ASSERT_EQ(invertData->L_nnz, expected_L_Values.size());
    for (std::size_t i = 0; i < expected_L_Values.size(); ++i)
    {
        // std::cout << std::setprecision(32) << invertData->L_values[i] << std::endl;
        EXPECT_FLOATINGPOINT_EQ(invertData->L_values[i], expected_L_Values[i])
    }

}
