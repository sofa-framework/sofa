/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <SofaBaseMechanics/config.h>

#include <SofaBaseMechanics/BarycentricMappers/TopologyBarycentricMapper.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include <sofa/core/Mapping.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>


namespace sofa::component::mapping
{

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;

template <class TIn, class TOut>
class BarycentricMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapping,TIn,TOut),
               SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef TIn In;
    typedef TOut Out;
    typedef In InDataTypes;
    typedef Out OutDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::Real Real;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    typedef typename OutDataTypes::Coord OutCoord;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef typename OutDataTypes::Real OutReal;

    typedef core::topology::BaseMeshTopology BaseMeshTopology;
    typedef TopologyBarycentricMapper<InDataTypes,OutDataTypes> Mapper;
    typedef typename Inherit1::ForceMask ForceMask;

public:
    Data< bool > useRestPosition; ///< Use the rest position of the input and output models to initialize the mapping    

    SingleLink<BarycentricMapping<In,Out>,Mapper,BaseLink::FLAG_STRONGLINK> d_mapper;
    SingleLink<BarycentricMapping<In,Out>,BaseMeshTopology,BaseLink::FLAG_STRONGLINK> d_input_topology;
    SingleLink<BarycentricMapping<In,Out>,BaseMeshTopology,BaseLink::FLAG_STRONGLINK> d_output_topology;

    void init() override;
    void reinit() override;
    void apply(const core::MechanicalParams *mparams, Data< typename Out::VecCoord >& out, const Data< typename In::VecCoord >& in) override;
    void applyJ(const core::MechanicalParams *mparams, Data< typename Out::VecDeriv >& out, const Data< typename In::VecDeriv >& in) override;
    void applyJT(const core::MechanicalParams *mparams, Data< typename In::VecDeriv >& out, const Data< typename Out::VecDeriv >& in) override;
    void applyJT(const core::ConstraintParams *cparams, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in) override;

    const sofa::defaulttype::BaseMatrix* getJ() override;
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs() override;
    void draw(const core::visual::VisualParams* vparams) override;
    void handleTopologyChange(core::topology::Topology* t) override;

    /// interface for continuous friction contact
    TopologyBarycentricMapper<InDataTypes,OutDataTypes> *getMapper()
    {
        return d_mapper.get();
    }

protected:
    typedef linearsolver::EigenSparseMatrix<InDataTypes, OutDataTypes> eigen_type;

    BarycentricMapping(core::State<In>* from, core::State<Out>* to,
                       typename Mapper::SPtr m_mapper);
    BarycentricMapping(core::State<In>* from=nullptr, core::State<Out>* to=nullptr,
                       BaseMeshTopology * from_topology=nullptr );

    ~BarycentricMapping() override {}
    void updateForceMask() override;

    /// eigen matrix for use with Compliant plugin
    eigen_type eigen;
    helper::vector< defaulttype::BaseMatrix* > js;

private:
    void createMapperFromTopology();
    void populateTopologies();
    void initMapper();
};

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_CPP)
extern template class SOFA_SOFABASEMECHANICS_API BarycentricMapping< Vec3dTypes, Vec3dTypes >;


#endif

} // namespace sofa::component::mapping
