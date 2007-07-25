/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDACONTACTMAPPER_H
#define SOFA_GPU_CUDA_CUDACONTACTMAPPER_H

#include <sofa/component/collision/BarycentricContactMapper.h>
#include <sofa/gpu/cuda/CudaDistanceGridCollisionModel.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;


/// Mapper for CudaRigidDistanceGridCollisionModel
template <class DataTypes>
class ContactMapper<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,DataTypes> : public RigidContactMapper<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,DataTypes>
{
public:
    typedef RigidContactMapper<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,DataTypes> Inherit;
    typedef typename Inherit::MMechanicalState MMechanicalState;
    typedef typename Inherit::MCollisionModel MCollisionModel;

    int addPoint(const Vector3& P, int index)
    {
        int i = Inherit::addPoint(P, index);
        if (!this->mapping)
        {
            MCollisionModel* model = this->model;
            MMechanicalState* outmodel = this->outmodel;
            typename DataTypes::Coord& x = (*outmodel->getX())[i];
            typename DataTypes::Deriv& v = (*outmodel->getV())[i];
            if (model->isTransformed(index))
            {
                x = model->getTranslation(index) + model->getRotation(index) * P;
            }
            else
            {
                x = P;
            }
            v = typename DataTypes::Deriv();
        }
        return i;
    }
};




} // namespace collision

} // namespace component

} // namespace sofa

#endif
