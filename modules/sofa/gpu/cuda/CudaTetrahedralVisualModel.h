#ifndef CUDAOGLTETRAHEDRALMODEL_H_
#define CUDAOGLTETRAHEDRALMODEL_H_

#include <sofa/component/visualmodel/OglTetrahedralModel.h>
#include <sofa/gpu/cuda/CudaTypes.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

template<class TCoord, class TDeriv, class TReal>
class OglTetrahedralModel< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > : public core::VisualModel
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef core::componentmodel::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::componentmodel::topology::BaseMeshTopology::SeqTetras SeqTetras;

private:
    core::componentmodel::topology::BaseMeshTopology* topo;
    core::componentmodel::behavior::MechanicalState<DataTypes>* nodes;

    bool needUpdateTopology;
    gpu::cuda::CudaVector<Tetra> tetras;

    Data<bool> depthTest;
    Data<bool> blending;
    Data<bool> useVBO;

public:
    OglTetrahedralModel();
    virtual ~OglTetrahedralModel();

    void init();
    void drawVisual();
    bool addBBox(double* minBBox, double* maxBBox);

    void updateVisual()
    {
        //if (!getContext()->getShowVisualModels()) return;
        updateTopology();
    }

protected:

    void updateTopology()
    {
        if (!topo || !nodes) return;
        if (!needUpdateTopology) return;
        needUpdateTopology = false;
        const SeqTetras& t = topo->getTetras();
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
