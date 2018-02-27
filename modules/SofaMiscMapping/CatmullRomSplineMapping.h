/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_CATMULLROMSPLINEMAPPING_H
#define SOFA_COMPONENT_MAPPING_CATMULLROMSPLINEMAPPING_H
#include "config.h"

#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>

#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

/**
 *  \brief CatmullRomSplineMapping ....  TODO!!
 *
 */

template <class TIn, class TOut>
class CatmullRomSplineMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CatmullRomSplineMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;

    // Input types
    typedef TIn In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;

    // Output types
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real OutReal;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

    typedef sofa::core::topology::BaseMeshTopology BaseMeshTopology;
    typedef BaseMeshTopology::Edge       Edge;
    typedef BaseMeshTopology::SeqEdges   SeqEdges;
    typedef BaseMeshTopology::Triangle       Triangle;
    typedef BaseMeshTopology::SeqTriangles   SeqTriangles;
    typedef BaseMeshTopology::index_type ID;

    typedef typename Inherit::ForceMask ForceMask;

protected:

    CatmullRomSplineMapping ();
    virtual ~CatmullRomSplineMapping();

    Data<unsigned int> SplittingLevel; ///< Number of recursive splits
    //Data<Real> Radius;

    //Both mesh topology must be Edge Mesh
    BaseMeshTopology* sourceMesh;
    BaseMeshTopology* targetMesh;

    helper::vector<defaulttype::Vec<4,ID> > m_index; // indices of primitives influencing each point.
    helper::vector<defaulttype::Vec<4,InReal> > m_weight;

public:
    void init() override;

    virtual void apply( const sofa::core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in) override;
    //void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    virtual void applyJ( const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in) override;
    //void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    virtual void applyJT( const sofa::core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in) override;
    //void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    virtual void applyJT( const sofa::core::ConstraintParams* cparams, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in) override;
    //void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );

    void draw(const core::visual::VisualParams* vparams) override;

};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_CATMULLROMSPLINEMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API CatmullRomSplineMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_MISC_MAPPING_API CatmullRomSplineMapping< defaulttype::Vec3dTypes, defaulttype::ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API CatmullRomSplineMapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_API CatmullRomSplineMapping< defaulttype::Vec3fTypes, defaulttype::ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API CatmullRomSplineMapping< defaulttype::Vec3dTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_API CatmullRomSplineMapping< defaulttype::Vec3fTypes, defaulttype::Vec3dTypes >;
#endif
#endif
#endif //defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_CatmullRomSplineMAPPING_CPP)



} // namespace mapping

} // namespace component

} // namespace sofa

#endif

