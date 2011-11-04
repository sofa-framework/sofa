/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_CATMULLROMSPLINEMAPPING_H
#define SOFA_COMPONENT_MAPPING_CATMULLROMSPLINEMAPPING_H

#include <sofa/core/Mapping.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/component.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/component/topology/EdgeSetTopologyModifier.h>

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

using namespace sofa::helper;
using namespace sofa::core::topology;

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
    typedef typename In::VecCoord VecInCoord;
    typedef typename In::VecDeriv VecInDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;

    // Output types
    typedef TOut Out;
    typedef typename Out::VecCoord VecOutCoord;
    typedef typename Out::VecDeriv VecOutDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real OutReal;

    typedef BaseMeshTopology::Edge       Edge;
    typedef BaseMeshTopology::SeqEdges   SeqEdges;
    typedef BaseMeshTopology::Triangle       Triangle;
    typedef BaseMeshTopology::SeqTriangles   SeqTriangles;
    typedef BaseMeshTopology::index_type ID;

protected:

    CatmullRomSplineMapping ();
    virtual ~CatmullRomSplineMapping();

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;

    Data<unsigned int> SplittingLevel;
    //Data<Real> Radius;

    //Both mesh topology must be Edge Mesh
    BaseMeshTopology* sourceMesh;
    BaseMeshTopology* targetMesh;

    vector<Vec<4,ID> > m_index; // indices of primitives influencing each point.
    vector<Vec<4,InReal> > m_weight;

public:
    void init();

    void apply(typename Out::VecCoord& out, const typename In::VecCoord& in);
    void applyJ(typename Out::VecDeriv& out, const typename In::VecDeriv& in);
    void applyJT(typename In::VecDeriv& out, const typename Out::VecDeriv& in);
    void applyJT(typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in);

    void draw(const core::visual::VisualParams* vparams);

};

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_CATMULLROMSPLINEMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_DEV_API CatmullRomSplineMapping< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_MISC_MAPPING_DEV_API CatmullRomSplineMapping< Vec3dTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_DEV_API CatmullRomSplineMapping< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_DEV_API CatmullRomSplineMapping< Vec3fTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_DEV_API CatmullRomSplineMapping< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_DEV_API CatmullRomSplineMapping< Vec3fTypes, Vec3dTypes >;
#endif
#endif
#endif //defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_CatmullRomSplineMAPPING_CPP)



} // namespace mapping

} // namespace component

} // namespace sofa

#endif

