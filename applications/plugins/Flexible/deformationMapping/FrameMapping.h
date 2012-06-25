#ifndef FRAMEMAPPING_H
#define FRAMEMAPPING_H

#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <../Flexible/shapeFunction/BaseShapeFunction.h>
#include "../initFlexible.h"

namespace sofa
{

template< class OutDataTypes>
class OutDataTypesInfo
{
public:
    enum {material_dimensions = OutDataTypes::material_dimensions};
};

template<class TCoord, class TDeriv, class TReal>
class OutDataTypesInfo<defaulttype::StdVectorTypes<TCoord, TDeriv, TReal> >
{
public:
    enum {material_dimensions = TCoord::spatial_dimensions};
};

template<class TCoord, class TDeriv, class TReal>
class OutDataTypesInfo<defaulttype::ExtVectorTypes<TCoord, TDeriv, TReal> >
{
public:
    enum {material_dimensions = TCoord::spatial_dimensions};
};


namespace component
{

namespace mapping
{


template<class TIn, class TOut>
class SOFA_Flexible_API FrameMapping : public core::Mapping<TIn, TOut>
{
public :
    SOFA_CLASS(SOFA_TEMPLATE2(FrameMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;

    enum { spatial_dimensions = Out::spatial_dimensions };
    enum { material_dimensions = OutDataTypesInfo<Out>::material_dimensions };
    typedef core::behavior::ShapeFunctionTypes<material_dimensions,Real> ShapeFunctionType;
    typedef core::behavior::BaseShapeFunction<ShapeFunctionType> BaseShapeFunction;
    typedef typename BaseShapeFunction::VReal VReal;
    typedef typename BaseShapeFunction::VGradient VGradient;
    typedef typename BaseShapeFunction::VHessian VHessian;
    typedef typename BaseShapeFunction::VRef VRef;
    typedef typename BaseShapeFunction::VMaterialToSpatial VMaterialToSpatial;
    typedef typename BaseShapeFunction::Coord mCoord; ///< material coordinates

    virtual void init();

    virtual void apply(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<OutVecCoord>& out, const Data<InVecCoord>& in);
    virtual void applyJ( const core::MechanicalParams* mparams /* PARAMS FIRST */, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in);
    virtual void applyJT(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in);

    Data<InVecCoord> m_restFrame;
    Data<OutVecCoord> m_restPos;
    Data<vector<VRef> > m_indices;
    Data<vector<VRef> > m_pointsToFramesIndices;
    Data<vector<VReal> > m_w;
    Data<vector<VGradient> > m_dw;
    Data<vector<VHessian> > m_ddw;
    Data<vector<VReal> > m_pointsW;
    Data<vector<VGradient> > m_pointsDw;
    Data<vector<VHessian> > m_pointsDdw;

protected :
    FrameMapping();
    virtual ~FrameMapping();

    BaseShapeFunction* m_shapeFun;
    BaseShapeFunction* m_pointShapeFun;
    InVecDeriv m_frameInitVel;
    OutVecDeriv m_particleInitVel;

};

using sofa::defaulttype::Rigid3fTypes;
using sofa::defaulttype::Vec3fTypes;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_FRAME_MAPPING)
#ifdef SOFA_FLOAT
extern template class SOFA_Flexible_API FrameMapping< Rigid3fTypes, Vec3fTypes >;
#endif
#endif

} // namespace mapping
} // namespace component
} // namespace sofa

#endif // FRAMEMAPPING_H
