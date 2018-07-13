#ifndef SOFA_COMPONENT_TOPOLOGY_SURFACEMASKTRAVERSAL_H
#define SOFA_COMPONENT_TOPOLOGY_SURFACEMASKTRAVERSAL_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <SofaBaseTopology/SurfaceTopologyContainer.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa
{

namespace component
{

namespace topology
{

class SurfaceMaskTraversal: public virtual core::objectmodel::BaseObject
{

public:
    SOFA_CLASS(SurfaceMaskTraversal, BaseObject);

    using SurfaceTopology = sofa::component::topology::SurfaceTopologyContainer;
    using Vertex = SurfaceTopology::Vertex;
    using Edge = SurfaceTopology::Edge;
    using Face = SurfaceTopology::Face;

protected:
    SurfaceMaskTraversal()
        :BaseObject()
    {}

public:
    virtual bool operator() (Vertex vertex) = 0;

    virtual bool operator() (Edge edge) = 0;

    virtual bool operator() (Face face) = 0;
};


template<class TDataTypes>
class FixedConstraintMask : public SurfaceMaskTraversal
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedConstraintMask, TDataTypes), SurfaceMaskTraversal);

    using SurfaceTopology = SurfaceMaskTraversal::SurfaceTopology;
    using Face = SurfaceTopology::Face;
    typedef unsigned int Index;
    typedef typename TDataTypes::VecCoord VecCoord;
    typedef typename TDataTypes::Coord Coord;
    typedef typename Coord::value_type Real;

protected:
    FixedConstraintMask()
        :BaseObject()
        , f_coef(initData(&f_coef,(Real)0.,"coef","Coef example"))
        , mstate(initLink("mstate", "MechanicalState used by this ForceField"))
    {}

public:
    SurfaceTopology* topology_;
    VecCoord x_;
    Data<Real> f_coef;

    SingleLink<FixedConstraintMask<TDataTypes>,sofa::core::behavior::MechanicalState<TDataTypes>,BaseLink::FLAG_STRONGLINK> mstate;

public:
    FixedConstraintMask(const FixedConstraintMask& rhs)
    {
        topology_= rhs.topology_;
        x_ = rhs.x_;
        mstate = rhs.mstate;
    }

    virtual ~FixedConstraintMask()
    { }

    virtual void init() override
    {
        this->getContext()->get(topology_);
        if (mstate.get() == NULL)
            msg_warning() << "mstate not found";
        x_ = mstate->read(core::ConstVecCoordId::position())->getValue();
    }

    bool operator() (Vertex vertex) override
    { }

    bool operator() (Edge edge) override
    { }

    bool operator() (Face face) override
    {
        const auto& dofs = topology_->get_dofs(Face(face));

        Index a = dofs[0];
        Index b = dofs[1];
        Index c = dofs[2];

//        return x_[a][2] <= f_coef.getValue() && x_[b][2] <= f_coef.getValue() && x_[c][2] <= f_coef.getValue();
                return x_[a][2] <= 31. && x_[b][2] <= -48. && x_[c][2] <= -38;
    }

    /*
    struct SomeStruct  {
        auto func_name(int x, int y) -> int;
    };

    auto SomeStruct::func_name(int x, int y) -> int {
        return x + y;
    }
    */
};


} //end namespace topology

} //end namespace component

} //end namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_SURFACEMASKTRAVERSAL_H
