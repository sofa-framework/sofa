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

#include <sofa/type/Mat_solve_LCP.h>
#include <algorithm>
#include <iterator>

SOFA_DEPRECATED_HEADER("v21.06", "v21.12", "sofa/type/Mat_solve_LCP.h")

namespace sofa::helper
{
	
template <int dim>
class LCPSolver
{
public:
    using Matrix = double [dim][dim];
    LCPSolver() = default;

    // assuming that q, M and res had a correct allocation...
    bool solve(const double *q, const Matrix &M, double *res)
    {
        constexpr auto sqdim = dim * dim;

        sofa::type::Vec<dim, double> tempQ; 
        sofa::type::Mat<dim, dim, double> tempM;
        sofa::type::Vec<dim*2, double> tempRes;

        //not possible because of const double(?)
        //std::copy_n(M, sqdim, tempM.data());
        //std::copy_n(q, dim, tempQ.data());
        for (auto i = 0; i < dim; i++)
        {
            tempQ[i] = q[i];
            for (auto j = 0; j < dim; j++)
            {
                tempM[i][j] = M[i][j];
            }
        }


        auto ret = sofa::type::solveLCP(tempQ, tempM, tempRes);

        std::copy_n(std::begin(tempRes), sqdim, res);

        return res;
    }

    void  printInfo(double* q, Matrix M ) = delete;
};

} // namespace sofa::helper
