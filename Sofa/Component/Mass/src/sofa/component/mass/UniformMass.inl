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
#include <sofa/component/mass/UniformMass.h>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/fwd.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/component/mass/AddMToMatrixFunctor.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>

namespace sofa::component::mass
{

using helper::WriteAccessor;
using helper::ReadAccessor;
using helper::WriteOnlyAccessor;
using type::vector;

using std::list;

using core::behavior::Mass;
using core::topology::BaseMeshTopology;
using core::topology::TopologyChange;
using core::MechanicalParams;
using core::behavior::MultiMatrixAccessor;
using core::visual::VisualParams;
using core::ConstVecCoordId;

using linearalgebra::BaseVector;
using type::Vec;
using type::Vec3d;
using defaulttype::DataTypeInfo;
using linearalgebra::BaseMatrix;

template <class DataTypes>
UniformMass<DataTypes>::UniformMass()
    : d_vertexMass ( initData ( &d_vertexMass, MassType ( 1.0f ), "vertexMass", "Specify one single, positive, real value for the mass of each particle. \n"
                                                                                "If unspecified or wrongly set, the totalMass information is used." ) )
    , d_totalMass ( initData ( &d_totalMass, SReal(1.0), "totalMass", "Specify the total mass resulting from all particles. \n"
                                                                      "If unspecified or wrongly set, the default value is used: totalMass = 1.0") )
    , d_filenameMass ( initData ( &d_filenameMass, "filename", "rigid file to load the mass parameters" ) )
    , d_showCenterOfGravity ( initData ( &d_showCenterOfGravity, false, "showGravityCenter", "display the center of gravity of the system" ) )
    , d_showAxisSize ( initData ( &d_showAxisSize, 1.0f, "showAxisSizeFactor", "factor length of the axis displayed (only used for rigids)" ) )
    , d_computeMappingInertia ( initData ( &d_computeMappingInertia, false, "compute_mapping_inertia", "to be used if the mass is placed under a mapping" ) )
    , d_showInitialCenterOfGravity ( initData ( &d_showInitialCenterOfGravity, false, "showInitialCenterOfGravity", "display the initial center of gravity of the system" ) )
    , d_showX0 ( initData ( &d_showX0, false, "showX0", "display the rest positions" ) )
    , d_localRange ( initData ( &d_localRange, Vec<2,int> ( -1,-1 ), "localRange", "optional range of local DOF indices. \n"
                                                                                   "Any computation involving only indices outside of this range \n"
                                                                                   "are discarded (useful for parallelization using mesh partitioning)" ) )
    , d_indices ( initData ( &d_indices, "indices", "optional local DOF indices. Any computation involving only indices outside of this list are discarded" ) )
    , d_preserveTotalMass( initData ( &d_preserveTotalMass, false, "preserveTotalMass", "Prevent totalMass from decreasing when removing particles."))
    , l_topology(initLink("topology", "link to the topology container"))
{
    constructor_message();


    sofa::core::objectmodel::Base::addUpdateCallback("updateFromTotalMass", {&d_totalMass}, [this](const core::DataTracker& )
    {
        if(m_isTotalMassUsed)
        {
            msg_info() << "dataInternalUpdate: data totalMass has changed";
            return updateFromTotalMass();
        }
        else
        {
            msg_info() << "vertexMass data is initially used, the callback associated with the totalMass is skipped";
            return updateFromVertexMass();
        }
    }, {});


    sofa::core::objectmodel::Base::addUpdateCallback("updateFromVertexMass", {&d_vertexMass}, [this](const core::DataTracker& )
    {
        if(!m_isTotalMassUsed)
        {
            msg_info() << "dataInternalUpdate: data vertexMass has changed";
            return updateFromVertexMass();
        }
        else
        {
            msg_info() << "totalMass data is initially used, the callback associated with the vertexMass is skipped";
            return updateFromTotalMass();
        }
    }, {});
}

template <class DataTypes>
UniformMass<DataTypes>::~UniformMass()
{}

template <class DataTypes>
void UniformMass<DataTypes>::constructor_message()
{
    d_filenameMass.setDisplayed(true) ;
    d_filenameMass.setReadOnly(true) ;
    d_filenameMass.setValue("unused") ;
    d_filenameMass.setHelp("File storing the mass parameters [rigid objects only].");
}

template <class DataTypes>
void UniformMass<DataTypes>::setMass ( const MassType& m )
{
    const MassType& currentVertexMass = d_vertexMass.getValue();
    d_vertexMass.setValue( m );
    if(!checkVertexMass())
    {
        msg_warning() << "Given value to setVertexMass() is not a strictly positive value\n"
                      << "Previous value is used: vertexMass = " << currentVertexMass;
        d_vertexMass.setValue(currentVertexMass);
    }
}

template <class DataTypes>
void UniformMass<DataTypes>::setTotalMass ( SReal m )
{
    Real currentTotalMass = d_totalMass.getValue();
    d_totalMass.setValue( m );
    if(!checkTotalMass())
    {
        msg_warning() << "Given value to setTotalMass() is not a strictly positive value\n"
                      << "Previous value is used: totalMass = " << currentTotalMass;
        d_totalMass.setValue(currentTotalMass);
    }
}


template <class DataTypes>
void UniformMass<DataTypes>::init()
{
    initDefaultImpl();
}


template <class DataTypes>
void UniformMass<DataTypes>::initDefaultImpl()
{
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);


    /// SingleStateAccessor checks the mstate pointer to a MechanicalObject
    Mass<DataTypes>::init();

        
    /// Check filename
    if ( d_filenameMass.isSet() && d_filenameMass.getValue() != "unused" )
    {
        loadRigidMass(d_filenameMass.getFullPath()) ;
    }


    /// Check indices
    WriteAccessor<Data<SetIndexArray > > indices = d_indices;

    //If d_localRange is set, update indices
    if (d_localRange.getValue()[0] >= 0
        && d_localRange.getValue()[1] > 0
        && d_localRange.getValue()[1] + 1 < int(mstate->getSize()))
    {
        indices.clear();
        for(int i=d_localRange.getValue()[0]; i<=d_localRange.getValue()[1]; i++)
            indices.push_back(i);
    }

    //If no given indices
    if(indices.size()==0)
    {
        indices.clear();
        for(int i=0; i<int(mstate->getSize()); i++)
            indices.push_back(i);
    }


    /// Check link to topology
    if (l_topology.empty())
    {
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    BaseMeshTopology* meshTopology = l_topology.get();

    if (meshTopology != nullptr && dynamic_cast<sofa::core::topology::TopologyContainer*>(meshTopology) != nullptr)
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        d_indices.createTopologyHandler(meshTopology);
        d_indices.supportNewTopologyElements(true);

        // Need to create a call back to assign index of new point into the topologySubsetData. Deletion is automatically handle.
        d_indices.setCreationCallback([](Index dataIndex, Index& valueIndex,
            const core::topology::BaseMeshTopology::Point& point,
            const sofa::type::vector< Index >& ancestors,
            const sofa::type::vector< SReal >& coefs)
        {
            SOFA_UNUSED(point);
            SOFA_UNUSED(ancestors);
            SOFA_UNUSED(coefs);
            valueIndex = dataIndex;
        });

        d_indices.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::ENDING_EVENT, [this](const core::topology::TopologyChange* eventTopo)
        {
            SOFA_UNUSED(eventTopo);
            updateMassOnResize(d_indices.getValue().size());
        });
    }


    /// Check on data isSet()
    if (d_vertexMass.isSet())
    {
        if(d_totalMass.isSet())
        {
            msg_warning(this) << "totalMass value overriding the value of the attribute vertexMass. \n"
                                 "vertexMass = totalMass / nb_dofs. \n"
                                 "To remove this warning you need to set either totalMass or vertexMass data field, but not both.";

            m_isTotalMassUsed = true;
            d_vertexMass.setReadOnly(true);
        }
        else
        {
            m_isTotalMassUsed = false;
            d_totalMass.setReadOnly(true);

            msg_info() << "Input vertexMass is used for initialization";
        }
    }
    else if (d_totalMass.isSet())
    {
        m_isTotalMassUsed = true;
        d_vertexMass.setReadOnly(true);

        msg_info() << "Input totalForce is used for initialization";
    }
    else
    {
        if(d_filenameMass.getValue() == "unused")
        {
            msg_error() << "No input mass information has been set. Please define one of both Data: "
                        << d_vertexMass.getName() << " or " << d_totalMass.getName()
                        << "\nFor your information, prior to #3927, default value was totalMass=\"1.0\".";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }


    /// Trigger callbacks to update data (see constructor)
    if(!this->isComponentStateValid())
        msg_error() << "Initialization process is invalid";

    /// Info post-init
    msg_info() << "totalMass  = " << d_totalMass.getValue() << " | "
                  "vertexMass = " << d_vertexMass.getValue();
}


template <class DataTypes>
bool UniformMass<DataTypes>::checkVertexMass()
{
    if(d_vertexMass.getValue() < 0.0 )
    {
        msg_error(this) << "vertexMass data can not have a negative value. \n"
                           "To remove this warning, you need to set one single, non-zero and positive value to the vertexMass data";
        return false;
    }
    else
    {
        return true;
    }
}


template <class DataTypes>
void UniformMass<DataTypes>::initFromVertexMass()
{
    //If the vertexMass attribute is set then the totalMass is computed from it
    //using the following formula: totalMass = vertexMass * number of particules
    const auto size = d_indices.getValue().size();
    const SReal vertexMass = SReal(d_vertexMass.getValue());
    const SReal totalMass = vertexMass * SReal(size);
    d_totalMass.setValue(totalMass);
    msg_info() << "vertexMass information is used";
}


template <class DataTypes>
bool UniformMass<DataTypes>::checkTotalMass()
{
    if(d_totalMass.getValue() < 0.0)
    {
        msg_error(this) << "totalMass data can not have a negative value. \n"
                           "To remove this warning, you need to set a non-zero positive value to the totalMass data";
        return false;
    }
    else
    {
        return true;
    }
}


template <class DataTypes>
void UniformMass<DataTypes>::initFromTotalMass()
{
    //If the totalMass attribute is set then the vertexMass is computed from it
    //using the following formula: vertexMass = totalMass / number of particules

    if(d_indices.getValue().size() > 0)
    {
        helper::WriteAccessor<Data<MassType>> m = d_vertexMass;
        *m = d_totalMass.getValue() / Real(d_indices.getValue().size());

        msg_info() << "totalMass information is used";
    }
    else
    {
        msg_warning() << "indices vector size is <= 0";
    }
}


template <class DataTypes>
sofa::core::objectmodel::ComponentState UniformMass<DataTypes>::updateFromTotalMass()
{
    if (checkTotalMass())
    {
        initFromTotalMass();
        return sofa::core::objectmodel::ComponentState::Valid;
    }
    else
    {
        msg_error() << "dataInternalUpdate: incorrect update from totalMass";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }
}


template <class DataTypes>
sofa::core::objectmodel::ComponentState UniformMass<DataTypes>::updateFromVertexMass()
{
    if(checkVertexMass())
    {
        initFromVertexMass();
        return sofa::core::objectmodel::ComponentState::Valid;
    }
    else
    {
        msg_error() << "dataInternalUpdate: incorrect update from vertexMass";
        return sofa::core::objectmodel::ComponentState::Invalid;
    }
}


template <class DataTypes>
void UniformMass<DataTypes>::updateMassOnResize(sofa::Size newSize)
{
    if (newSize == 0)
    {
        d_totalMass.setValue(Real(0));
        return;
    }

    if (d_preserveTotalMass.getValue())
    {
        Real newVertexMass = d_totalMass.getValue() / Real(newSize);
        d_vertexMass.setValue(static_cast<MassType>(newVertexMass));
    }
    else
    {
        d_totalMass.setValue(Real(newSize) * Real(d_vertexMass.getValue()));
    }
}


// -- Mass interface
template <class DataTypes>
void UniformMass<DataTypes>::addMDx ( const core::MechanicalParams*, DataVecDeriv& vres, const DataVecDeriv& vdx, SReal factor)
{
    if (!this->isComponentStateValid())
        return;

    helper::WriteAccessor<DataVecDeriv> res = vres;
    helper::ReadAccessor<DataVecDeriv> dx = vdx;

    WriteAccessor<Data<SetIndexArray > > indices = d_indices;

    MassType m = d_vertexMass.getValue();
    if ( factor != 1.0 )
        m *= typename DataTypes::Real(factor);

    for (const auto i : indices)
        res[i] += dx[i] * m;
}


template <class DataTypes>
void UniformMass<DataTypes>::accFromF ( const core::MechanicalParams*, DataVecDeriv& va, const DataVecDeriv& vf )
{
    if (!this->isComponentStateValid())
        return;

    WriteOnlyAccessor<DataVecDeriv> a = va;
    ReadAccessor<DataVecDeriv> f = vf;

    const ReadAccessor<Data<SetIndexArray > > indices = d_indices;

    MassType m = d_vertexMass.getValue();
    for (const auto i : indices)
        a[i] = f[i] / m;
}


template <class DataTypes>
void UniformMass<DataTypes>::addMDxToVector ( BaseVector * resVect, const VecDeriv* dx, SReal mFact, unsigned int& offset )
{
    SOFA_UNUSED(resVect);
    SOFA_UNUSED(dx);
    SOFA_UNUSED(mFact);
    SOFA_UNUSED(offset);
}


template <class DataTypes>
void UniformMass<DataTypes>::addGravityToV(const MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if (mparams)
    {
        helper::WriteAccessor<DataVecDeriv> v = d_v;

        const SReal* g = getContext()->getGravity().ptr();
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2] );
        Deriv hg = theGravity * Real(sofa::core::mechanicalparams::dt(mparams));

        dmsg_info()<< "addGravityToV hg = "<<theGravity<<"*"<<sofa::core::mechanicalparams::dt(mparams)<<"="<<hg ;

        for ( unsigned int i=0; i<v.size(); i++ )
        {
            v[i] += hg;
        }
    }
}

template <class DataTypes>
void UniformMass<DataTypes>::addForce ( const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/ )
{
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if ( this->m_separateGravity.getValue() )
        return;

    helper::WriteAccessor<DataVecDeriv> f = vf;

    // weight
    const SReal* g = getContext()->getGravity().ptr();
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2] );
    const MassType& m = d_vertexMass.getValue();
    Deriv mg = theGravity * m;

    dmsg_info() <<"addForce, mg = "<<d_vertexMass<<" * "<<theGravity<<" = "<<mg;

    const ReadAccessor<Data<SetIndexArray > > indices = d_indices;

    // add weight and inertia force
    for (const auto i : indices)
    {
        f[i] += mg;
    }
}

template <class DataTypes>
SReal UniformMass<DataTypes>::getKineticEnergy ( const MechanicalParams* params, const DataVecDeriv& d_v  ) const
{
    SOFA_UNUSED(params);

    ReadAccessor<DataVecDeriv> v = d_v;
    const ReadAccessor<Data<SetIndexArray > > indices = d_indices;

    SReal e = 0;
    const MassType& m = d_vertexMass.getValue();

    for (const auto i : indices)
        e += v[i] * m * v[i];

    return e/2;
}

template <class DataTypes>
SReal UniformMass<DataTypes>::getPotentialEnergy ( const MechanicalParams* params, const DataVecCoord& d_x  ) const
{
    SOFA_UNUSED(params);
    ReadAccessor<DataVecCoord> x = d_x;
    const ReadAccessor<Data<SetIndexArray > > indices = d_indices;

    SReal e = 0;
    const MassType& m = d_vertexMass.getValue();

    Vec3d g( getContext()->getGravity());
    Deriv gravity;
    DataTypes::set(gravity, g[0], g[1], g[2]);

    Deriv mg = gravity * m;

    for (const auto i : indices)
        e -= mg * x[i];

    return e;
}


// does nothing by default, need to be specialized in .cpp
template <class DataTypes>
type::Vec6
UniformMass<DataTypes>::getMomentum ( const core::MechanicalParams* params, const DataVecCoord& d_x, const DataVecDeriv& d_v  ) const
{
    SOFA_UNUSED(params);
    SOFA_UNUSED(d_x);
    SOFA_UNUSED(d_v);

    msg_warning(this) << "You are using the getMomentum function that has not been implemented"
                         "for the template '"<< this->getTemplateName() << "'.\n" ;

    return type::Vec6();
}


/// Add Mass contribution to global Matrix assembling
template <class DataTypes>
void UniformMass<DataTypes>::addMToMatrix (sofa::linearalgebra::BaseMatrix * mat, SReal mFact, unsigned int &offset)
{
    if (!this->isComponentStateValid())
        return;

    const MassType& m = d_vertexMass.getValue();

    static constexpr auto N = Deriv::total_size;

    AddMToMatrixFunctor<Deriv,MassType, linearalgebra::BaseMatrix> calc;

    const ReadAccessor<Data<SetIndexArray > > indices = d_indices;
    for (auto id : *indices)
    {
        calc ( mat, m, int(offset + N * id), mFact);
    }
}

template <class DataTypes>
void UniformMass<DataTypes>::buildMassMatrix(sofa::core::behavior::MassMatrixAccumulator* matrices)
{
    if (!this->isComponentStateValid())
    {
        return;
    }

    const MassType& m = d_vertexMass.getValue();
    static constexpr auto N = Deriv::total_size;

    AddMToMatrixFunctor<Deriv,MassType, core::behavior::MassMatrixAccumulator> calc;

    const ReadAccessor<Data<SetIndexArray > > indices = d_indices;
    for (const auto index : indices)
    {
        calc( matrices, m, N * index, 1.);
    }
}


template <class DataTypes>
SReal UniformMass<DataTypes>::getElementMass (sofa::Index ) const
{
    return (SReal ( d_vertexMass.getValue() ));
}


template <class DataTypes>
void UniformMass<DataTypes>::getElementMass (sofa::Index  index, BaseMatrix *m ) const
{
    SOFA_UNUSED(index);

    static const BaseMatrix::Index dimension = BaseMatrix::Index(DataTypeInfo<Deriv>::size());
    if ( m->rowSize() != dimension || m->colSize() != dimension )
        m->resize ( dimension, dimension );

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType, linearalgebra::BaseMatrix>() ( m, d_vertexMass.getValue(), 0, 1 );
}


template <class DataTypes>
void UniformMass<DataTypes>::draw(const VisualParams* vparams)
{
    if ( !vparams->displayFlags().getShowBehaviorModels() )
        return;

    if (!mstate.get())
        return;

    if (!d_showCenterOfGravity.getValue())
        return;

    ReadAccessor<VecCoord> x = mstate->read(core::vec_id::read_access::position)->getValue();
    const ReadAccessor<Data<SetIndexArray > > indices = d_indices;

    Coord gravityCenter;
    std::vector<  sofa::type::Vec3 > points;
    for (const auto i : indices)
    {
        const sofa::type::Vec3 p = toVec3(DataTypes::getCPos(x[i]));

        points.push_back ( p );
        gravityCenter += x[i];
    }
    vparams->drawTool()->drawSpheres(points, 0.01f, sofa::type::RGBAColor::yellow());
    
    {
        gravityCenter /= indices.size();
        const sofa::type::RGBAColor color = sofa::type::RGBAColor::yellow();

        Real axisSize = d_showAxisSize.getValue();
        sofa::type::Vec3 temp;

        for ( unsigned int i=0 ; i<3 ; i++ )
            if(i < Coord::spatial_dimensions )
                temp[i] = gravityCenter[i];

        vparams->drawTool()->drawCross(temp, float(axisSize), color);
    }
}


template<class DataTypes>
void UniformMass<DataTypes>::loadRigidMass( const std::string&  filename)
{
    msg_error(this) << "The attribute filename is set to ["<< filename << "] while " << msgendl <<
                         " the current object is not based on a Rigid template. It is thus ignored. " << msgendl <<
                         "To fix this error you can: " << msgendl <<
                         "  - remove the filename attribute from <UniformMass filename='"<< filename << "'/>." << msgendl <<
                         "  - use a Rigid mechanical object instead of a Vec one." ;
}

} // namespace sofa::component::mass
