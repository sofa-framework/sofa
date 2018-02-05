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
#ifndef CUDAOGLTETRAHEDRALMODEL_H_
#define CUDAOGLTETRAHEDRALMODEL_H_

#include <VolumetricRendering/OglTetrahedralModel.h>
#include <sofa/gpu/cuda/CudaTypes.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

template<class TCoord, class TDeriv, class TReal>
class OglTetrahedralModel< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > : public core::visual::VisualModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(OglTetrahedralModel,SOFA_TEMPLATE3(gpu::cuda::CudaVectorTypes,TCoord,TDeriv,TReal)),core::visual::VisualModel);
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;

private:
    core::topology::BaseMeshTopology* topo;
    core::behavior::MechanicalState<DataTypes>* nodes;

    bool needUpdateTopology;
    gpu::cuda::CudaVector<Tetra> tetras;

    Data<bool> depthTest;
    Data<bool> blending;
    Data<bool> useVBO;

public:
    OglTetrahedralModel();
    virtual ~OglTetrahedralModel();

    void init();
    void drawTransparent(const core::visual::VisualParams*);
    bool addBBox(SReal* minBBox, SReal* maxBBox);

    void handleTopologyChange()
    {
        needUpdateTopology = true;
    }

    void updateVisual()
    {
        //if (!getContext()->getShowVisualModels()) return;
        updateTopology();
    }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

protected:

    void updateTopology()
    {
        if (!topo || !nodes) return;
        if (!needUpdateTopology) return;
        needUpdateTopology = false;
        const SeqTetrahedra& t = topo->getTetrahedra();
        tetras.clear();
        if (!t.empty())
        {
            tetras.fastResize(t.size());
            std::copy ( t.begin(), t.end(), tetras.hostWrite() );
        }
    }

};

#endif /*OGLTETRAHEDRALMODEL_H_*/
}
}
}
