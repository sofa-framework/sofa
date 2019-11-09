/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTETRAHEDRONSETTOPOLOGY_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTETRAHEDRONSETTOPOLOGY_H

#include <SofaBaseMechanics/BarycentricMappers/BarycentricMapperTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>

namespace sofa
{

namespace component
{

namespace mapping
{


using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::Mat3x3d;
using sofa::defaulttype::Vector3;
typedef typename sofa::core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;

/// Class allowing barycentric mapping computation on a TetrahedronSetTopology
template<class In, class Out>
class BarycentricMapperTetrahedronSetTopology : public BarycentricMapperTopologyContainer<In,Out,typename BarycentricMapper<In,Out>::MappingData3D,Tetrahedron>
{
    typedef typename BarycentricMapper<In,Out>::MappingData3D MappingData;

public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperTetrahedronSetTopology,In,Out),SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingData,Tetrahedron));
    typedef typename Inherit1::Real Real;
    typedef typename In::VecCoord VecCoord;
    using Inherit1::m_toTopology;

    int addPointInTetra(const int index, const SReal* baryCoords) override ;

protected:
    BarycentricMapperTetrahedronSetTopology(topology::TetrahedronSetTopologyContainer* fromTopology,
                                            topology::PointSetTopologyContainer* toTopology);
    ~BarycentricMapperTetrahedronSetTopology() override {}

    virtual helper::vector<Tetrahedron> getElements() override;
    virtual helper::vector<SReal> getBaryCoef(const Real* f) override;
    helper::vector<SReal> getBaryCoef(const Real fx, const Real fy, const Real fz);
    void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Tetrahedron& element) override;
    void computeCenter(Vector3& center, const typename In::VecCoord& in, const Tetrahedron& element) override;
    void computeDistance(double& d, const Vector3& v) override;
    void addPointInElement(const int elementIndex, const SReal* baryCoords) override;

    //handle topology changes depending on the topology
    void processTopologicalChanges(const typename Out::VecCoord& out, const typename In::VecCoord& in, core::topology::Topology* t) {
        using sofa::core::behavior::MechanicalState;
        
        if (t != m_toTopology) return;

        if ( m_toTopology->beginChange() == m_toTopology->endChange() )
            return;

        auto itBegin = m_toTopology->beginChange();
        auto itEnd = m_toTopology->endChange();

        helper::WriteAccessor < Data< helper::vector<MappingData > > > vectorData = d_map;
        vectorData.resize (out.size());

        for (auto changeIt = itBegin; changeIt != itEnd; ++changeIt ) {
            const core::topology::TopologyChangeType changeType = ( *changeIt )->getChangeType();
            switch ( changeType )
            {
            //TODO: implementation of BarycentricMapperHexahedronSetTopology<In,Out>::handleTopologyChange()
            case core::topology::POINTSINDICESSWAP:  ///< For PointsIndicesSwap.
            {
                const core::topology::PointsIndicesSwap * ps = static_cast<const core::topology::PointsIndicesSwap*>(*changeIt);
                MappingData copy = vectorData[ps->index[0]];
                vectorData[ps->index[0]] = vectorData[ps->index[1]];
                vectorData[ps->index[1]] = copy;
                break;
            }
            case core::topology::POINTSADDED:        ///< For PointsAdded.
            {
                const core::topology::PointsAdded * pa = static_cast<const core::topology::PointsAdded*>(*changeIt);
                auto& array = pa->getElementArray();

                for (unsigned i=0;i<array.size();i++) {
                    unsigned pid = array[i];
                    processAddPoint(Out::getCPos(out[pid]),
                                    in,
                                    vectorData[pid]);
                }

                break;
            }
            case core::topology::POINTSREMOVED:      ///< For PointsRemoved.
            {
                // nothing to do
                break;
            }
            case core::topology::POINTSRENUMBERING:  ///< For PointsRenumbering.
            {
                const core::topology::PointsRenumbering * pr = static_cast<const core::topology::PointsRenumbering*>(*changeIt);
                auto& array = pr->getIndexArray();
                auto& inv_array = pr->getinv_IndexArray();
                for (unsigned i=0;i<array.size();i++) {
                    MappingData copy = vectorData[array[i]];
                    vectorData[inv_array[i]] = vectorData[array[i]];
                    vectorData[array[i]] = copy;
                }
                break;
            }
            default:
            {
                break;
            }
            }
        }
    }

    void processAddPoint(const sofa::defaulttype::Vec3d & pos, const typename In::VecCoord& in, MappingData & vectorData){
        const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron>& tetrahedra = this->m_fromTopology->getTetrahedra();

        sofa::defaulttype::Vector3 coefs;
        int index = -1;
        double distance = std::numeric_limits<double>::max();
        for ( unsigned int t = 0; t < tetrahedra.size(); t++ )
        {
            sofa::defaulttype::Mat3x3d base;
            sofa::defaulttype::Vector3 center;
            computeBase(base,in,tetrahedra[t]);
            computeCenter(center,in,tetrahedra[t]);

            sofa::defaulttype::Vec3d v = base * ( pos - in[tetrahedra[t][0]] );
            double d = std::max ( std::max ( -v[0],-v[1] ),std::max ( -v[2],v[0]+v[1]+v[2]-1 ) );

            if ( d>0 ) d = ( pos-center ).norm2();

            if ( d<distance )
            {
                coefs = v;
                distance = d;
                index = t;
            }
        }

        vectorData.in_index = index;
        vectorData.baryCoords[0] = ( Real ) coefs[0];
        vectorData.baryCoords[1] = ( Real ) coefs[1];
        vectorData.baryCoords[2] = ( Real ) coefs[2];
    }

    topology::TetrahedronSetTopologyContainer*      m_fromContainer {nullptr};
    topology::TetrahedronSetGeometryAlgorithms<In>*	m_fromGeomAlgo  {nullptr};

    using Inherit1::d_map;
    using Inherit1::m_fromTopology;
    using Inherit1::m_matrixJ;
    using Inherit1::m_updateJ;
};

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTETRAHEDRONSETTOPOLOGY_CPP)
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3dTypes, Vec3dTypes >;


#endif

}}}

#endif
