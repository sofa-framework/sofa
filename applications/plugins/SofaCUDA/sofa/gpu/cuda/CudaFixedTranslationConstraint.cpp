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
#include "CudaTypes.h"
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <SofaBoundaryCondition/FixedTranslationConstraint.h>
#include <SofaBoundaryCondition/FixedTranslationConstraint.inl>


#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

template<>
void FixedTranslationConstraint<gpu::cuda::CudaVec6fTypes>::draw(const core::visual::VisualParams* vparams);
#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
void FixedTranslationConstraint<gpu::cuda::CudaVec6dTypes>::draw(const core::visual::VisualParams* vparams);
#endif // SOFA_GPU_CUDA_DOUBLE


template <>
void component::projectiveconstraintset::FixedTranslationConstraint<gpu::cuda::CudaVec6fTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = f_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    glDisable(GL_LIGHTING);
    glPointSize(10);
    glColor4f(1, 0.5, 0.5, 1);
    glBegin(GL_POINTS);
    if (f_fixAll.getValue() == true)
    {
        for (unsigned i = 0; i < x.size(); i++)
        {
            helper::gl::glVertexT(defaulttype::Vec<3,float>(x[i][0], x[i][1], x[i][2]));
        }
    }
    else
    {
        for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            helper::gl::glVertexT(defaulttype::Vec<3,float>(x[*it][0], x[*it][1], x[*it][2]));
        }
    }
    glEnd();
}

#ifdef SOFA_GPU_CUDA_DOUBLE
template <>
void component::projectiveconstraintset::FixedTranslationConstraint<gpu::cuda::CudaVec6dTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = f_indices.getValue();
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    glDisable(GL_LIGHTING);
    glPointSize(10);
    glColor4f(1, 0.5, 0.5, 1);
    glBegin(GL_POINTS);
    if (f_fixAll.getValue() == true)
    {
        for (unsigned i = 0; i < x.size(); i++)
        {
            helper::gl::glVertexT(defaulttype::Vec<3,float>(x[i][0], x[i][1], x[i][2]));
        }
    }
    else
    {
        for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            helper::gl::glVertexT(defaulttype::Vec<3,float>(x[*it][0], x[*it][1], x[*it][2]));
        }
    }
    glEnd();
}
#endif // SOFA_GPU_CUDA_DOUBLE


template class FixedTranslationConstraint<gpu::cuda::CudaVec6fTypes>;
template class FixedTranslationConstraint<gpu::cuda::CudaRigid3fTypes>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class FixedTranslationConstraint<gpu::cuda::CudaVec6dTypes>;
template class FixedTranslationConstraint<gpu::cuda::CudaRigid3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE

}// namespace projectiveconstraintset

}// namespace component

namespace gpu
{

namespace cuda
{


SOFA_DECL_CLASS(CudaFixedTranslationConstraint)

int FixedTranslationConstraintCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
// .add< component::projectiveconstraintset::FixedTranslationConstraint<CudaVec3fTypes> >()
// .add< component::projectiveconstraintset::FixedTranslationConstraint<CudaVec3f1Types> >()
        .add< component::projectiveconstraintset::FixedTranslationConstraint<CudaVec6fTypes> >()
        .add< component::projectiveconstraintset::FixedTranslationConstraint<CudaRigid3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
// .add< component::projectiveconstraintset::FixedTranslationConstraint<CudaVec3dTypes> >()
// .add< component::projectiveconstraintset::FixedTranslationConstraint<CudaVec3d1Types> >()
        .add< component::projectiveconstraintset::FixedTranslationConstraint<CudaVec6dTypes> >()
        .add< component::projectiveconstraintset::FixedTranslationConstraint<CudaRigid3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;


} // namespace cuda

} // namespace gpu

} // namespace sofa
