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
        if (dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
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
