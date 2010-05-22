/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/core/behavior/MechanicalMultiMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/map.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/topology/PointSubset.h>

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
template <class BasicMapping>
class SubsetMultiMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SubsetMultiMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
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
    //typedef helper::vector<unsigned int> IndexArray;
    inline unsigned int computeTotalInputPoints() const
    {
        typename std::map<const  In* , IndexArray >::const_iterator iter;
        unsigned int total = 0;
        for ( iter = _indices.begin(); iter != _indices.end(); iter++)
        {
            total += (*iter).second.size();
        }
        return total;
    };

    virtual void init();

    virtual ~SubsetMultiMapping() {};

    void addPoint(const In* fromModel, int index);


    virtual void apply(const helper::vector<OutVecCoord*>& outPos, const vecConstInVecCoord& inPos);
    virtual void applyJ (const helper::vector<OutVecDeriv*>& outDeriv, const helper::vector<const  InVecDeriv*>& inDeriv);
    virtual void applyJT(const helper::vector< InVecDeriv*>& outDeriv, const helper::vector<const OutVecDeriv*>& inDeriv);

protected :
    typedef topology::PointSubset IndexArray;
    std::map<const In*,IndexArray>  _indices;
};



#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_CPP)
using namespace core::behavior;
using namespace sofa::defaulttype;
using namespace sofa::core;

#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API SubsetMultiMapping<MechanicalMultiMapping< MechanicalState< Vec3dTypes>, MechanicalState< Vec3dTypes> > > ;
extern template class SOFA_COMPONENT_MAPPING_API SubsetMultiMapping<MultiMapping< MechanicalState< Vec3dTypes>, MechanicalState< Vec3dTypes> > > ;

#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API SubsetMultiMapping<MechanicalMultiMapping< MechanicalState< Vec3fTypes>, MechanicalState< Vec3fTypes> > > ;
extern template class SOFA_COMPONENT_MAPPING_API SubsetMultiMapping<MultiMapping< MechanicalState< Vec3fTypes>, MechanicalState< Vec3fTypes> > > ;

#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_H
