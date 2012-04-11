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
#ifndef SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_H
#define SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_H

#include <sofa/core/MultiMapping.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace mapping
{

/**
 * @class SubsetMapping
 * @brief Compute a subset of input points
 */
template <class TIn, class TOut>
class SubsetMultiMapping : public core::MultiMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SubsetMultiMapping, TIn, TOut), SOFA_TEMPLATE2(core::MultiMapping, TIn, TOut));

    typedef core::MultiMapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename InCoord::value_type Real;
    typedef typename helper::vector <const InVecCoord*> vecConstInVecCoord;
    typedef typename helper::vector<OutVecCoord*> vecOutVecCoord;
    /// Correspondance array
    typedef core::topology::BaseMeshTopology::SetIndex IndexArray;

    virtual void init();

    void addPoint(const core::BaseState* fromModel, int index);

    virtual void apply(const helper::vector<OutVecCoord*>& outPos, const vecConstInVecCoord& inPos);
    virtual void applyJ(const helper::vector<OutVecDeriv*>& outDeriv, const helper::vector<const  InVecDeriv*>& inDeriv);
    virtual void applyJT(const helper::vector< InVecDeriv*>& outDeriv, const helper::vector<const OutVecDeriv*>& inDeriv);
    virtual void applyDJT(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) {}

    /// @todo implement this
    virtual void applyJT( const helper::vector< typename In::MatrixDeriv* >& , const helper::vector< const typename Out::MatrixDeriv* >& );

#ifdef SOFA_HAVE_EIGEN2
    /// Experimental API used to handle multimappings in matrix assembly. Returns pointers to matrices associated with parent states, consistently with  getFrom().
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs();
#endif


protected :

    SubsetMultiMapping()
        : Inherit()
    {
    }

    virtual ~SubsetMultiMapping() {};


    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;

    vector<defaulttype::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector

    typedef std::pair<unsigned, unsigned> IndexPair;  ///< first is the parent, second is the index in the parent
    vector<IndexPair> indexPairs;                     ///< for each child, its parent and index in parent
};



#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_CPP)

using namespace sofa::defaulttype;

#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Vec3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Vec3fTypes, Vec3fTypes >;
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_H
