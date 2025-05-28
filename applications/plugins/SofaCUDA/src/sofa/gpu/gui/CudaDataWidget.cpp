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
#include <sofa/gpu/gui/CudaDataWidget.h>
#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/helper/Factory.inl>
#include <sofa/gui/qt/DataWidget.h>
#include <sofa/gui/qt/SimpleDataWidget.h>
#include <sofa/gui/qt/TableDataWidget.h>

namespace sofa::gui::qt
{
using sofa::helper::Creator;
using namespace sofa::type;
using namespace sofa::defaulttype;

Creator<DataWidgetFactory, SimpleDataWidget< Vec<1, int> > > DWClass_Vec12i("default", true);

template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<int> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<unsigned int> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<float> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<double> >;

template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec1i> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec2i> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec3i> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec4i> >;

template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec1f> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec2f> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec3f> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec4f> >;

template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec1d> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec2d> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec3d> >;
template class SOFA_GPU_CUDA_API TDataWidget<sofa::gpu::cuda::CudaVector<Vec4d> >;


Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<int>, TABLE_HORIZONTAL > > DWClass_cudaVectori("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<unsigned int>, TABLE_HORIZONTAL > > DWClass_cudaVectorui("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<float>, TABLE_HORIZONTAL > > DWClass_cudaVectorf("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<double>, TABLE_HORIZONTAL > > DWClass_cudaVectord("default", true);

Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec1i> > > DWClass_cudaVectorVec1i("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec2i> > > DWClass_cudaVectorVec2i("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec3i> > > DWClass_cudaVectorVec3i("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec4i> > > DWClass_cudaVectorVec4i("default", true);

Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec1f> > > DWClass_cudaVectorVec1f("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec2f> > > DWClass_cudaVectorVec2f("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec3f> > > DWClass_cudaVectorVec3f("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec4f> > > DWClass_cudaVectorVec4f("default", true);

Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec1d> > > DWClass_cudaVectorVec1d("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec2d> > > DWClass_cudaVectorVec2d("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec3d> > > DWClass_cudaVectorVec3d("default", true);
Creator<DataWidgetFactory, TableDataWidget< sofa::gpu::cuda::CudaVector<Vec4d> > > DWClass_cudaVectorVec4d("default", true);


} // namespace sofa::gui::qt
