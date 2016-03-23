/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_BEZIER2MESHMECHANICALMAPPING_H
#define SOFA_COMPONENT_MAPPING_BEZIER2MESHMECHANICALMAPPING_H

#include <sofa/core/Mapping.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa { namespace core { namespace topology { class BaseMeshTopology; } } }
namespace sofa { namespace component { namespace topology { class Bezier2MeshTopologicalMapping; } } }
namespace sofa { namespace component { namespace topology { template<typename  D>  class BezierTriangleSetGeometryAlgorithms; } } }
namespace sofa { namespace component { namespace topology {  class BezierTriangleSetTopologyContainer; } } }
namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
class Bezier2MeshMechanicalMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(Bezier2MeshMechanicalMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;

    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename InCoord::value_type Real;
protected:
    Bezier2MeshMechanicalMapping(core::State<In>* from = NULL, core::State<Out>* to = NULL);

    virtual ~Bezier2MeshMechanicalMapping();

public:

    void init();

    void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in);

    void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in);

    void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in);

    void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in);

protected:
	// the associated topological map 
	topology::Bezier2MeshTopologicalMapping* topoMap;
	// the input bezier triangle geometry algorithm object
	topology::BezierTriangleSetGeometryAlgorithms<TIn> *btsga;
	// the input bezier triangle geometry algorithm object
	topology::BezierTriangleSetTopologyContainer *btstc;
	// currently used bezier degree for the input Bezier triangulation or tetrahedral mesh
	size_t bezierDegree;
	// currently used tesselation degree for the output riangulation or tetrahedral mesh
	size_t tesselationDegree;

	/// precomputed coefficients to interpolate the positions of points.
	sofa::helper::vector< sofa::helper::vector<Real> > precomputedLinearBernsteinCoefficientArray; 
	sofa::helper::vector< sofa::helper::vector<Real> > precomputedTriangularBernsteinCoefficientArray; 
	sofa::helper::vector< sofa::helper::vector<Real> > precomputedDerivUTriangularBernsteinCoefficientArray; 
	sofa::helper::vector< sofa::helper::vector<Real> > precomputedDerivVTriangularBernsteinCoefficientArray;
	/// precompute weight array for tesselated mesh
	sofa::helper::vector< Real > bezierTesselationWeightArray; 
	// local indexing of points inside tessellated triangles
	sofa::helper::vector<sofa::defaulttype::Vec<3,unsigned char > > tesselatedTriangleIndices; 


};



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_BEZIER2MESHMECHANICALMAPPING_CPP)  //// ATTENTION PB COMPIL WIN3Z
#ifndef SOFA_FLOAT
extern template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< defaulttype::Vec3dTypes, defaulttype::ExtVec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< defaulttype::Vec3fTypes, defaulttype::ExtVec3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< defaulttype::Vec3dTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< defaulttype::Vec3fTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< defaulttype::Vec3fTypes, defaulttype::ExtVec3dTypes >;
extern template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< defaulttype::Vec3dTypes, defaulttype::ExtVec3fTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
