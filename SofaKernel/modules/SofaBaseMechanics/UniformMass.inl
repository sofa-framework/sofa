/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MASS_UNIFORMMASS_INL
#define SOFA_COMPONENT_MASS_UNIFORMMASS_INL

#include <SofaBaseMechanics/UniformMass.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <SofaBaseMechanics/AddMToMatrixFunctor.h>
#include <sofa/simulation/Simulation.h>
#include <iostream>
#include <string.h>

#ifdef SOFA_SUPPORT_MOVING_FRAMES
#include <sofa/core/behavior/InertiaForce.h>
#endif



namespace sofa
{

namespace component
{

namespace mass
{

using helper::WriteAccessor;
using helper::ReadAccessor;
using helper::WriteOnlyAccessor;
using helper::vector;

using std::list;

using core::behavior::Mass;
using core::topology::BaseMeshTopology;
using core::topology::TopologyChange;
using core::MechanicalParams;
using core::behavior::MultiMatrixAccessor;
using core::visual::VisualParams;
using core::ConstVecCoordId;

using defaulttype::BaseVector;
using defaulttype::Vec;
using defaulttype::Vec3d;
using defaulttype::DataTypeInfo;
using defaulttype::BaseMatrix;



template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::UniformMass()
    : d_vertexMass ( initData ( &d_vertexMass, MassType ( 1.0f ), "vertexMass",
                                "Specify one single, positive, real value for the mass of each particle. \n"
                                "If unspecified or wrongly set, the totalMass information is used." ) )
    , d_totalMass ( initData ( &d_totalMass, (SReal)1.0, "totalMass",
                               "Specify the total mass resulting from all particles. \n"
                               "If unspecified or wrongly set, the default value is used: totalMass = 1.0") )
    , d_filenameMass ( initData ( &d_filenameMass, "filename",
                                  "rigid file to load the mass parameters" ) )
    , d_showCenterOfGravity ( initData ( &d_showCenterOfGravity, false, "showGravityCenter",
                                         "display the center of gravity of the system" ) )
    , d_showAxisSize ( initData ( &d_showAxisSize, 1.0f, "showAxisSizeFactor",
                                  "factor length of the axis displayed (only used for rigids)" ) )
    , d_computeMappingInertia ( initData ( &d_computeMappingInertia, false, "compute_mapping_inertia",
                                           "to be used if the mass is placed under a mapping" ) )
    , d_showInitialCenterOfGravity ( initData ( &d_showInitialCenterOfGravity, false, "showInitialCenterOfGravity",
                                                "display the initial center of gravity of the system" ) )
    , d_showX0 ( initData ( &d_showX0, false, "showX0",
                            "display the rest positions" ) )
    , d_localRange ( initData ( &d_localRange, Vec<2,int> ( -1,-1 ), "localRange",
                                "optional range of local DOF indices. \n"
                                "Any computation involving only indices outside of this range \n"
                                "are discarded (useful for parallelization using mesh partitionning)" ) )
    , d_indices ( initData ( &d_indices, "indices",
                             "optional local DOF indices. Any computation involving only indices outside of this list are discarded" ) )
    , d_handleTopoChange ( initData ( &d_handleTopoChange, false, "handleTopoChange",
                                      "The mass and totalMass are recomputed on particles add/remove." ) )
    , d_preserveTotalMass( initData ( &d_preserveTotalMass, false, "preserveTotalMass",
                                      "Prevent totalMass from decreasing when removing particles."))
{
    constructor_message();
}

template <class DataTypes, class MassType>
UniformMass<DataTypes, MassType>::~UniformMass()
{}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::constructor_message()
{
    d_filenameMass.setDisplayed(true) ;
    d_filenameMass.setReadOnly(true) ;
    d_filenameMass.setValue("unused") ;
    d_filenameMass.setHelp("File storing the mass parameters [rigid objects only].");
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setMass ( const MassType& m )
{
    d_vertexMass.setValue ( m );
}

template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::setTotalMass ( SReal m )
{
    d_totalMass.setValue ( m );
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::reinit()
{
    WriteAccessor<Data<vector<int> > > indices = d_indices;
    m_doesTopoChangeAffect = false;

    if(mstate==NULL){
        msg_warning(this) << "Missing mechanical state. \n"
                             "UniformMass need to be used with an object also having a MechanicalState. \n"
                             "To remove this warning: add a <MechanicalObject/> to the parent node of the one \n"
                             " containing this <UniformMass/>";
        return;
    }

    if ( d_filenameMass.isSet() && d_filenameMass.getValue() != "unused" ){
        loadRigidMass(d_filenameMass.getFullPath()) ;
    }

    //If localRange is set, update indices
    if (d_localRange.getValue()[0] >= 0
        && d_localRange.getValue()[1] > 0
        && d_localRange.getValue()[1] + 1 < (int)mstate->getSize())
    {
        indices.clear();
        for(int i=d_localRange.getValue()[0]; i<=d_localRange.getValue()[1]; i++)
            indices.push_back(i);
    }

    //If no given indices
    if(indices.size()==0)
    {
        indices.clear();
        for(int i=0; i<(int)mstate->getSize(); i++)
            indices.push_back(i);
        m_doesTopoChangeAffect = true;
    }

    //Select mass information
    bool useDefault = true;
    //If user defines the vertexMass, use this as the mass
    if (d_vertexMass.isSet())
    {
        //Check double definition : both totalMass and vertexMass are user-defined
        if (d_totalMass.isSet())
        {
            msg_warning(this) << "totalMass value overriding the value of the attribute vertexMass. \n"
                                 "vertexMass = totalMass / nb_dofs. \n"
                                 "To remove this warning you need to set either totalMass or vertexMass data field, but not both.";
            if(d_totalMass.getValue() <= 0.0)
            {
                msg_warning(this) << "totalMass data can not have a negative value. \n"
                                     "Switching back to default value: totalMass = 1.0 \n"
                                     "To remove this warning, you need to set a positive value to the totalMass data";
                d_totalMass.setValue(1.0) ;
            }
            if(d_vertexMass.getValue() <= 0.0 )
            {
                msg_warning(this) << "vertexMass data can not have a negative value. \n"
                                     "To remove this warning, you need to set one single and positive value to the vertexMass data";
            }
            //By default use the totalMass
            useDefault = true;
        }
        //Check for negative or null value, by default use the totalMass
        else if(d_vertexMass.getValue() <= 0.0 )
        {
            msg_warning(this) << "vertexMass data can not have a negative value. \n"
                                 "Switching back to default value: totalMass = 1.0 \n"
                                 "To remove this warning, you need to set one single and positive value to the vertexMass data";
            d_totalMass.setValue(1.0) ;
            useDefault = true;
        }
        //If no problem detected, then use the vertexMass information
        //totalMass will be computed from it using the formulat : totalMass = vertexMass * number of particules
        else
        {
            SReal totalMass = d_vertexMass.getValue() * indices.size();
            d_totalMass.setValue(totalMass);
            useDefault = false;
            msg_info() << "vertexMass information is used";
        }
    }
    //else totalMass is used
    else
    {
        if(!d_totalMass.isSet())
        {
            msg_info() << "No information about the mass is given. Default totatMass is used as reference.";
        }
        //Check for negative or null value, by default use totalMass = 1.0
        if(d_totalMass.getValue() <= 0.0)
        {
            msg_warning(this) << "totalMass data can not have a negative value. \n"
                                 "Switching back to default value: totalMass = 1.0 \n"
                                 "To remove this warning, you need to set a positive value to the totalMass data";
            d_totalMass.setValue(1.0) ;
        }

        //If the totalMass attribute is set then the vertexMass is computed from it
        //using the following formula: vertexMass = totalMass / number of particules
        useDefault = true;
    }

    //If the mass is based on the totalMass information
    if(useDefault)
    {
        MassType *m = d_vertexMass.beginEdit();
        *m = ( ( typename DataTypes::Real ) d_totalMass.getValue() / indices.size() );
        d_vertexMass.endEdit();
    }

    //Info post-reinit
    msg_info() << "totalMass  = " << d_totalMass.getValue() << " \n"
                  "vertexMass = " << d_vertexMass.getValue();
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::init()
{
    Mass<DataTypes>::init();
    reinit();
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::handleTopologyChange()
{
    BaseMeshTopology *meshTopology = getContext()->getMeshTopology();
    WriteAccessor<Data<vector<int> > > indices = d_indices;

    if(m_doesTopoChangeAffect)
    {
        indices.clear();
        for(size_t i=0; i<mstate->getSize(); i++)
            indices.push_back((int)i);
    }

    if ( meshTopology != 0 )
    {
        list< const TopologyChange * >::const_iterator it = meshTopology->beginChange();
        list< const TopologyChange * >::const_iterator itEnd = meshTopology->endChange();

        while ( it != itEnd )
        {
            switch ( ( *it )->getChangeType() )
            {
            case core::topology::POINTSADDED:
                if ( d_handleTopoChange.getValue() && m_doesTopoChangeAffect)
                {
                    MassType* m = d_vertexMass.beginEdit();
                    *m = ( ( typename DataTypes::Real ) d_totalMass.getValue() / mstate->getSize() );
                    d_vertexMass.endEdit();
                }
                break;

            case core::topology::POINTSREMOVED:
                if ( d_handleTopoChange.getValue() && m_doesTopoChangeAffect)
                {
                    if (!d_preserveTotalMass.getValue())
                        d_totalMass.setValue (mstate->getSize() * (Real)d_vertexMass.getValue() );
                    else
                        d_vertexMass.setValue( static_cast< MassType >( ( typename DataTypes::Real ) d_totalMass.getValue() / mstate->getSize()) );
                }
                break;

            default:
                break;
            }

            ++it;
        }
    }
}


// -- Mass interface
template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMDx ( const core::MechanicalParams*,
                                                DataVecDeriv& vres,
                                                const DataVecDeriv& vdx,
                                                SReal factor)
{
    helper::WriteAccessor<DataVecDeriv> res = vres;
    helper::ReadAccessor<DataVecDeriv> dx = vdx;

    WriteAccessor<Data<vector<int> > > indices = d_indices;

    MassType m = d_vertexMass.getValue();
    if ( factor != 1.0 )
        m *= ( typename DataTypes::Real ) factor;

    for ( unsigned int i=0; i<indices.size(); i++ )
        res[indices[i]] += dx[indices[i]] * m;
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::accFromF ( const core::MechanicalParams*,
                                                  DataVecDeriv& va,
                                                  const DataVecDeriv& vf )
{
    WriteOnlyAccessor<DataVecDeriv> a = va;
    ReadAccessor<DataVecDeriv> f = vf;

    ReadAccessor<Data<vector<int> > > indices = d_indices;

    MassType m = d_vertexMass.getValue();
    for ( unsigned int i=0; i<indices.size(); i++ )
        a[indices[i]] = f[indices[i]] / m;
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMDxToVector ( BaseVector * resVect,
                                                        const VecDeriv* dx,
                                                        SReal mFact,
                                                        unsigned int& offset )
{
    SOFA_UNUSED(resVect);
    SOFA_UNUSED(dx);
    SOFA_UNUSED(mFact);
    SOFA_UNUSED(offset);
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addGravityToV(const MechanicalParams* mparams,
                                                     DataVecDeriv& d_v)
{
    if (mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        const SReal* g = getContext()->getGravity().ptr();
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2] );
        Deriv hg = theGravity * ( Real ) (mparams->dt());

        dmsg_info()<< " addGravityToV hg = "<<theGravity<<"*"<<mparams->dt()<<"="<<hg ;

        for ( unsigned int i=0; i<v.size(); i++ )
        {
            v[i] += hg;
        }

        d_v.endEdit();
    }
}

template <class DataTypes, class MassType>
#ifdef SOFA_SUPPORT_MAPPED_MASS
void UniformMass<DataTypes, MassType>::addForce ( const core::MechanicalParams* mparams, DataVecDeriv& vf, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/)
#else
#ifdef SOFA_SUPPORT_MOVING_FRAMES
void UniformMass<DataTypes, MassType>::addForce ( const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& x, const DataVecDeriv& v )
#else
void UniformMass<DataTypes, MassType>::addForce ( const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/ )
#endif
#endif
{
#ifndef SOFA_SUPPORT_MOVING_FRAMES
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if ( this->m_separateGravity.getValue() )
        return;
#endif

    helper::WriteAccessor<DataVecDeriv> f = vf;

    // weight
    const SReal* g = getContext()->getGravity().ptr();
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2] );
    const MassType& m = d_vertexMass.getValue();
    Deriv mg = theGravity * m;

    dmsg_info() <<" addForce, mg = "<<d_vertexMass<<" * "<<theGravity<<" = "<<mg;


#ifdef SOFA_SUPPORT_MOVING_FRAMES
    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = getContext()->getPositionInWorld() / vframe;
    aframe = getContext()->getPositionInWorld().backProjectVector ( aframe );
#endif

    ReadAccessor<Data<vector<int> > > indices = d_indices;

    // add weight and inertia force
    if (this->m_separateGravity.getValue()) for ( unsigned int i=0; i<indices.size(); i++ )
    {
#ifdef SOFA_SUPPORT_MOVING_FRAMES
        f[indices[i]] += core::behavior::inertiaForce ( vframe,aframe,m,x[indices[i]],v[indices[i]] );
#endif
    }
    else for ( unsigned int i=0; i<indices.size(); i++ )
    {
#ifdef SOFA_SUPPORT_MOVING_FRAMES
        f[indices[i]] += mg + core::behavior::inertiaForce ( vframe,aframe,m,x[indices[i]],v[indices[i]] );
#else
        f[indices[i]] += mg;
#endif
    }


#ifdef SOFA_SUPPORT_MAPPED_MASS
    if ( compute_mapping_inertia.getValue() )
    {
        helper::ReadAccessor< Data<VecDeriv> > acc = *mparams->readDx(mstate);
        // add inertia force due to acceleration from the motion of the mapping (coriolis type force)
        if ( acc.size() != f.size() )
            return;

        for ( unsigned int i=0; i<f.size(); i++ )
        {
            Deriv coriolis = -acc[i]*m;
            f[i] += coriolis;
        }
    }
#endif
}

template <class DataTypes, class MassType>
SReal UniformMass<DataTypes, MassType>::getKineticEnergy ( const MechanicalParams* params,
                                                           const DataVecDeriv& d_v  ) const
{
    SOFA_UNUSED(params);

    ReadAccessor<DataVecDeriv> v = d_v;
    ReadAccessor<Data<vector<int> > > indices = d_indices;

    SReal e = 0;
    const MassType& m = d_vertexMass.getValue();

    for ( unsigned int i=0; i<indices.size(); i++ )
        e+= v[indices[i]]*m*v[indices[i]];

    return e/2;
}

template <class DataTypes, class MassType>
SReal UniformMass<DataTypes, MassType>::getPotentialEnergy ( const MechanicalParams* params,
                                                             const DataVecCoord& d_x  ) const
{
    SOFA_UNUSED(params);
    ReadAccessor<DataVecCoord> x = d_x;
    ReadAccessor<Data<vector<int> > > indices = d_indices;

    SReal e = 0;
    const MassType& m = d_vertexMass.getValue();

    Vec3d g( getContext()->getGravity());
    Deriv gravity;
    DataTypes::set(gravity, g[0], g[1], g[2]);

    Deriv mg = gravity * m;

    for ( unsigned int i=0; i<indices.size(); i++ )
        e -= mg*x[indices[i]];

    return e;
}


// does nothing by default, need to be specialized in .cpp
template <class DataTypes, class MassType>
defaulttype::Vector6
UniformMass<DataTypes, MassType>::getMomentum ( const core::MechanicalParams* params,
                                                const DataVecCoord& d_x,
                                                const DataVecDeriv& d_v  ) const
{
    SOFA_UNUSED(params);
    SOFA_UNUSED(d_x);
    SOFA_UNUSED(d_v);

    msg_warning(this) << "You are using the getMomentum function that has not been implemented"
                         "for the template '"<< this->getTemplateName() << "'.\n" ;

    return defaulttype::Vector6();
}


/// Add Mass contribution to global Matrix assembling
template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::addMToMatrix (const MechanicalParams *mparams,
                                                     const MultiMatrixAccessor* matrix)
{
    const MassType& m = d_vertexMass.getValue();

    const size_t N = DataTypeInfo<Deriv>::size();

    AddMToMatrixFunctor<Deriv,MassType> calc;
    MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(mstate);

    Real mFactor = (Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());

    ReadAccessor<Data<vector<int> > > indices = d_indices;
    for ( unsigned int i=0; i<indices.size(); i++ )
        calc ( r.matrix, m, r.offset + N*indices[i], mFactor);
}


template <class DataTypes, class MassType>
SReal UniformMass<DataTypes, MassType>::getElementMass ( unsigned int ) const
{
    return ( SReal ) ( d_vertexMass.getValue() );
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::getElementMass ( unsigned int  index ,
                                                        BaseMatrix *m ) const
{
    SOFA_UNUSED(index);

    static const BaseMatrix::Index dimension = (BaseMatrix::Index) DataTypeInfo<Deriv>::size();
    if ( m->rowSize() != dimension || m->colSize() != dimension )
        m->resize ( dimension, dimension );

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType>() ( m, d_vertexMass.getValue(), 0, 1 );
}


template <class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::draw(const VisualParams* vparams)
{
    if ( !vparams->displayFlags().getShowBehaviorModels() )
        return;

    ReadAccessor<VecCoord> x = mstate->read(ConstVecCoordId::position())->getValue();
    ReadAccessor<Data<vector<int> > > indices = d_indices;

    Coord gravityCenter;
    std::vector<  sofa::defaulttype::Vector3 > points;
    for ( unsigned int i=0; i<indices.size(); i++ )
    {
        sofa::defaulttype::Vector3 p;
        p = DataTypes::getCPos(x[indices[i]]);

        points.push_back ( p );
        gravityCenter += x[indices[i]];
    }


    if ( d_showCenterOfGravity.getValue() )
    {
        gravityCenter /= x.size();
        const sofa::defaulttype::Vec4f color(1.0,1.0,0.0,1.0);

        Real axisSize = d_showAxisSize.getValue();
        sofa::defaulttype::Vector3 temp;

        for ( unsigned int i=0 ; i<3 ; i++ )
            if(i < Coord::spatial_dimensions )
                temp[i] = gravityCenter[i];

        vparams->drawTool()->drawCross(temp, (float)axisSize, color);
    }
}

template<class DataTypes, class MassType>
void UniformMass<DataTypes, MassType>::loadRigidMass( const std::string&  filename)
{
    msg_warning(this) << "The attribute filename is set to ["<< filename << "] while " << msgendl <<
                         " the current object is not based on a Rigid template. It is thus ignored. " << msgendl <<
                         "To remove this warning you can: " << msgendl <<
                         "  - remove the filename attribute from <UniformMass filename='"<< filename << "'/>." << msgendl <<
                         "  - use a Rigid mechanical object instead of a VecXX one." ;
}

} // namespace mass

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MASS_UNIFORMMASS_INL
