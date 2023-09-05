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
#ifndef SOFA_GPU_CUDA_CUDAVISUALMODEL_H
#define SOFA_GPU_CUDA_CUDAVISUALMODEL_H

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/State.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/gpu/cuda/CudaTypes.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class DataTypes>
class CudaKernelsCudaVisualModel;

} // namespace cuda

} // namespace gpu


namespace component
{

namespace visualmodel
{

template <class TDataTypes>
class CudaVisualModel : public core::visual::VisualModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CudaVisualModel, TDataTypes), VisualModel);
    //typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef core::topology::BaseMeshTopology::PointID Index;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef core::topology::BaseMeshTopology::Quad Quad;
    typedef core::topology::BaseMeshTopology::SeqQuads SeqQuads;

    typedef core::State<DataTypes> TState;

    typedef gpu::cuda::CudaKernelsCudaVisualModel<DataTypes> Kernels;

    bool needUpdateTopology;
    gpu::cuda::CudaVector<Triangle> triangles;
    gpu::cuda::CudaVector<Quad> quads;

    VecCoord fnormals;
    VecCoord vnormals;

    int nbElement; ///< number of elements
    int nbVertex; ///< number of vertices to process to compute all elements
    int nbElementPerVertex; ///< max number of elements connected to a vertex
    /// Index of elements attached to each points (layout per bloc of NBLOC vertices, with first element of each vertex, then second element, etc)
    gpu::cuda::CudaVector<int> velems;

    Data<type::Vec4f> matAmbient; ///< material ambient color
    Data<type::Vec4f> matDiffuse; ///< material diffuse color and alpha
    Data<type::Vec4f> matSpecular; ///< material specular color
    Data<type::Vec4f> matEmissive; ///< material emissive color
    Data<float> matShininess; ///< material specular shininess
    Data<bool> useVBO; ///< true to activate Vertex Buffer Object
    Data<bool> computeNormals; ///< true to compute smooth normals

    CudaVisualModel()
        : needUpdateTopology(true), nbElement(0), nbVertex(0), nbElementPerVertex(0)
        , matAmbient  ( initData( &matAmbient,   type::Vec4f(0.1f,0.1f,0.1f,0.0f), "ambient",   "material ambient color") )
        , matDiffuse  ( initData( &matDiffuse,   type::Vec4f(0.8f,0.8f,0.8f,1.0f), "diffuse",   "material diffuse color and alpha") )
        , matSpecular ( initData( &matSpecular,  type::Vec4f(1.0f,1.0f,1.0f,0.0f), "specular",  "material specular color") )
        , matEmissive ( initData( &matEmissive,  type::Vec4f(0.0f,0.0f,0.0f,0.0f), "emissive",  "material emissive color") )
        , matShininess( initData( &matShininess, 45.0f,                                   "shininess", "material specular shininess") )
        , useVBO( initData( &useVBO, false, "useVBO", "true to activate Vertex Buffer Object") )
        , computeNormals( initData( &computeNormals, false, "computeNormals", "true to compute smooth normals") )
        , state(initLink("state", "State used by this component"))
        , topology(initLink("topology", "Topology used by this component"))
    {}

    virtual void init() override;
    virtual void reinit() override;
    virtual void internalDraw(const core::visual::VisualParams* vparams);
    virtual void doDrawVisual(const core::visual::VisualParams*) override;
    virtual void drawTransparent(const core::visual::VisualParams*) override;
    virtual void drawShadow(const core::visual::VisualParams*) override;
    virtual void updateVisual() override;
    virtual void updateTopology();
    virtual void updateNormals();
    virtual void handleTopologyChange() override;

    virtual void computeBBox(const core::ExecParams* params, bool=false) override;

protected:



    void initV(int nbe, int nbv, int nbelemperv)
    {
        nbElement = nbe;
        nbVertex = nbv;
        nbElementPerVertex = nbelemperv;
        const int nbloc = (nbVertex+BSIZE-1)/BSIZE;
        velems.resize(nbloc*nbElementPerVertex*BSIZE);
        for (unsigned int i=0; i<velems.size(); i++)
            velems[i] = 0;
    }

    void setV(int vertex, int num, int index)
    {
        const int bloc = vertex/BSIZE;
        const int b_x  = vertex%BSIZE;
        velems[ bloc*BSIZE*nbElementPerVertex // start of the bloc
                + num*BSIZE                     // offset to the element
                + b_x                           // offset to the vertex
              ] = index+1;
    }


    SingleLink<CudaVisualModel<DataTypes>, TState, BaseLink::FLAG_STRONGLINK> state;
    SingleLink<CudaVisualModel<DataTypes>, core::topology::BaseMeshTopology, BaseLink::FLAG_STRONGLINK> topology;
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
