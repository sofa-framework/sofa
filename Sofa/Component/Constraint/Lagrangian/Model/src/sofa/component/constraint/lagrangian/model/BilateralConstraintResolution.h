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
#include <sofa/component/constraint/lagrangian/model/config.h>

#include <sofa/type/Mat.h>
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/core/behavior/ConstraintResolution.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>

namespace sofa::component::constraint::lagrangian::model::bilateralconstraintresolution
{

using sofa::core::behavior::ConstraintResolution ;

class BilateralConstraintResolution : public ConstraintResolution
{
public:
    BilateralConstraintResolution(SReal* initF=nullptr) 
        : ConstraintResolution(1)
        , _f(initF) {}
    void resolution(int line, SReal** w, SReal* d, SReal* force, SReal *dfree) override
    {
        SOFA_UNUSED(dfree);
        force[line] -= d[line] / w[line][line];
    }

    void init(int line, SReal** /*w*/, SReal* force) override
    {
        if(_f) { force[line] = *_f; }
    }

    void initForce(int line, SReal* force) override
    {
        if(_f) { force[line] = *_f; }
    }

    void store(int line, SReal* force, bool /*convergence*/) override
    {
        if(_f) *_f = force[line];
    }

protected:
    SReal* _f;
};

class BilateralConstraintResolution3Dof : public ConstraintResolution
{
public:

    BilateralConstraintResolution3Dof(sofa::type::Vec3d* vec = nullptr)
        : ConstraintResolution(3)
        , _f(vec)
    {
    }
    void init(int line, SReal** w, SReal *force) override
    {
        sofa::type::Mat<3,3,SReal> temp;
        temp[0][0] = w[line][line];
        temp[0][1] = w[line][line+1];
        temp[0][2] = w[line][line+2];
        temp[1][0] = w[line+1][line];
        temp[1][1] = w[line+1][line+1];
        temp[1][2] = w[line+1][line+2];
        temp[2][0] = w[line+2][line];
        temp[2][1] = w[line+2][line+1];
        temp[2][2] = w[line+2][line+2];

        const bool canInvert = sofa::type::invertMatrix(invW, temp);
        assert(canInvert);
        SOFA_UNUSED(canInvert);

        // invW is unsused in this scope, remove the warning:
        SOFA_UNUSED(invW);

        if(_f)
        {
            for(int i=0; i<3; i++)
                force[line+i] = (*_f)[i];
        }
    }

    void initForce(int line, SReal* force) override
    {
        if(_f)
        {
            for(int i=0; i<3; i++)
                force[line+i] = (*_f)[i];
        }
    }

    void resolution(int line, SReal** /*w*/, SReal* d, SReal* force, SReal * dFree) override
    {
        SOFA_UNUSED(dFree);
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<3; j++)
                force[line+i] -= d[line+j] * invW[i][j];
        }
    }

    void store(int line, SReal* force, bool /*convergence*/) override
    {
        if(_f)
        {
            for(int i=0; i<3; i++)
                (*_f)[i] = force[line+i];
        }
    }

protected:
    sofa::type::Mat<3,3,SReal> invW;
    sofa::type::Vec3d* _f;
};

class BilateralConstraintResolutionNDof : public ConstraintResolution
{
public:
    using EigenVectorX = Eigen::Matrix<SReal, Eigen::Dynamic, 1>;
    using EigenMatrixX = Eigen::Matrix<SReal, Eigen::Dynamic, -1>;
    BilateralConstraintResolutionNDof(unsigned blockSize ) 
    : ConstraintResolution(blockSize)
    , wBlock(EigenMatrixX(blockSize, blockSize))
    {
    }
    void init(int line, SReal** w, SReal * /*force*/) override
    {
        for (auto i = 0; i < wBlock.rows(); ++i)   
        {
            wBlock.row(i) = EigenVectorX::Map(&w[line + i][line], wBlock.cols());
        }
        wBlockInv.compute(wBlock);
    }

    void resolution(int line, SReal** /*w*/, SReal* displacement, SReal* force, SReal* /*dFree*/) override
    {
        Eigen::Map< EigenVectorX > f(&force[line], wBlock.cols());
        const Eigen::Map< EigenVectorX > d(&displacement[line], wBlock.cols());
        const EigenVectorX f_local = wBlockInv.solve(d);
        f -= f_local;
    }

protected:
    EigenMatrixX wBlock;
    Eigen::LDLT< EigenMatrixX > wBlockInv;
};

} // namespace sofa::component::constraint::lagrangian::model::bilateralconstraintresolution

namespace sofa::component::constraint::lagrangian::model
{

    /// Import the following into the constraintset namespace to preserve
/// compatibility with the existing sofa source code.
using bilateralconstraintresolution::BilateralConstraintResolution;
using bilateralconstraintresolution::BilateralConstraintResolution3Dof;
using bilateralconstraintresolution::BilateralConstraintResolutionNDof;

} //namespace sofa::component::constraint::lagrangian::model
