/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_INL
#define SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_INL

#include <sofa/core/visual/VisualParams.h>
#include "FrameBlendingMapping.h"
#include "LinearBlending.inl"
#include "DualQuatBlending.inl"
#include "GridMaterial.inl"
#include <sofa/core/Mapping.inl>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/gl/glText.inl>
#include <sofa/helper/gl/template.h>
#include <sofa/simulation/Simulation.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseMechanics/MechanicalObject.inl>
#include <iostream>
#include <SofaSimulationTree/GNode.h>
#include <SofaLoader/MeshObjLoader.h>
namespace sofa
{

namespace defaulttype
{

inline const Vec<3,double>& center(const DeformationGradientTypes<3, 3, 1, double>::Coord& c)
{
    return c.getCenter();
}

inline const Vec<3,double>& center(const DeformationGradientTypes<3, 3, 2, double>::Coord& c)
{
    return c.getCenter();
}

inline const Vec<3,float>& center(const DeformationGradientTypes<3, 3, 1, float>::Coord& c)
{
    return c.getCenter();
}

inline const Vec<3,float>& center(const DeformationGradientTypes<3, 3, 2, float>::Coord& c)
{
    return c.getCenter();
}

template<class Real>
inline const Vec<3,Real>& center(const Vec<3,Real>& c)
{
    return c;
}

template<class _Real>
inline Vec<3,_Real>& center(Vec<3,_Real>& c)
{
    return c;
}

inline const Vec<3,double>& center(const StdAffineTypes<3,double>::Coord& c)
{
    return c.getCenter();
}

inline const Vec<3,double>& center(const StdRigidTypes<3,double>::Coord& c)
{
    return c.getCenter();
}

inline const Vec<3,double>& center(const StdQuadraticTypes<3,double>::Coord& c)
{
    return c.getCenter();
}
}

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::FramePointHandler::applyCreateFunction(unsigned int, OutCoord &, const Point &,
        const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (m_map)
    {
        if(!m_map->useAdaptivity.getValue()) return;

        m_map->updateMapping();
    }
}


using helper::WriteAccessor;
using helper::ReadAccessor;

template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::FrameBlendingMapping (core::State<In>* from, core::State<Out>* to )
    : Inherit ( from, to ), FData(), SData()
    , useLinearWeights ( initData ( &useLinearWeights, false, "useLinearWeights","use linearly interpolated weights between the two closest frames." ) )
    , blending ( initData ( &blending,"inout","Data specific to the conversion between the types" ) )
    , dq_blending ( initData ( &dq_blending,"dqinout","Data specific to the conversion between the types" ) )
    , useDQ ( initData ( &useDQ, false, "useDQ","use dual quaternion blending instead of linear blending ." ) )
    , useAdaptivity ( initData ( &useAdaptivity, false, "useAdaptivity","use automatic frame and sample adaptation." ) )
    , maskFrom(NULL)
    , maskTo(NULL)
    , f_initPos ( initData ( &f_initPos,"initPos","initial child coordinates in the world reference frame" ) )
    , f_index ( initData ( &f_index,"indices","parent indices for each child" ) )
    , f_groups ( initData ( &f_groups,"groups","child group (initialized from trianglegroupes)" ) )
    , weight ( initData ( &weight,"weights","influence weights of the Dofs" ) )
    , weightDeriv ( initData ( &weightDeriv,"weightGradients","weight gradients" ) )
    , weightDeriv2 ( initData ( &weightDeriv2,"weightHessians","weight Hessians" ) )
    , showBlendedFrame ( initData ( &showBlendedFrame, false, "showBlendedFrame","weights list for the influences of the references Dofs" ) )
    , showFromIndex ( initData ( &showFromIndex, ( unsigned int ) 0, "showFromIndex","Displayed From Index." ) )
    , showWeights ( initData ( &showWeights, false, "showWeights","Show coeficients." ) )
    , showGammaCorrection ( initData ( &showGammaCorrection, 1.0, "showGammaCorrection","Correction of the Gamma by a power" ) )
    , showWeightsValues ( initData ( &showWeightsValues, false, "showWeightsValues","Show coeficients values." ) )
    , showReps ( initData ( &showReps, false, "showReps","Show repartition." ) )
    , showValuesNbDecimals ( initData ( &showValuesNbDecimals, 0, "showValuesNbDecimals","Multiply floating point by 10^n." ) )
    , showTextScaleFactor ( initData ( &showTextScaleFactor, 0.00005, "showTextScaleFactor","Text Scale Factor." ) )
    , showGradients ( initData ( &showGradients, false, "showGradients","Show gradients." ) )
    , showGradientsValues ( initData ( &showGradientsValues, false, "showGradientsValues","Show Gradients Values." ) )
    , showGradientsScaleFactor ( initData ( &showGradientsScaleFactor, 0.0001, "showGradientsScaleFactor","Gradients Scale Factor." ) )
    , showStrain ( initData ( &showStrain, false, "showStrain","Show Computed Strain Tensors." ) )
    , showStrainScaleFactor ( initData ( &showStrainScaleFactor, 1.0, "showStrainScaleFactor","Strain Tensors Scale Factor." ) )
    , showDetF ( initData ( &showDetF, false, "showDetF","Show Computed Det F." ) )
    , showDetFScaleFactor ( initData ( &showDetFScaleFactor, 1.0, "showDetFScaleFactor","Det F Scale Factor." ) )
    , targetFrameNumber ( initData ( &targetFrameNumber, ( unsigned int ) 0, "targetFrameNumber","Desired number of frames resulting from the automatic discretization of the material. Use 0 to use user-defined frames." ) )
    , initializeFramesInRigidParts ( initData ( &initializeFramesInRigidParts, false, "initializeFramesInRigidParts","Automatically initialize frames in rigid parts if stiffness>15E6." ) )
    , targetSampleNumber ( initData ( &targetSampleNumber, ( unsigned int ) 0, "targetSampleNumber","Desired number of integration points resulting from the automatic discretization of the material. Use 0 to use user-defined integration points." ) )
    , restrictInterpolationToLabel ( initData ( &restrictInterpolationToLabel, "restrictInterpolationToLabel","Restrict interpolation to a label in gridmaterial." ) )
{
    pointHandler = new FramePointHandler(this, &f_initPos);

    // These sout are here to check if the template interface work well
    sout << "In VSize: " << defaulttype::InDataTypesInfo<In>::VSize << sendl;
    sout << "Out order: " << defaulttype::OutDataTypesInfo<TOut>::primitive_order << sendl;
}

template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::~FrameBlendingMapping ()
{
    if(pointHandler) delete pointHandler;
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::init()
{
    if (core::behavior::BaseMechanicalState* stateFrom = dynamic_cast<core::behavior::BaseMechanicalState*>(this->fromModel.get()))
        maskFrom = &stateFrom->forceMask;
    if (core::behavior::BaseMechanicalState* stateTo = dynamic_cast<core::behavior::BaseMechanicalState*>(this->toModel.get()))
        maskTo = &stateTo->forceMask;

    // init samples and frames according to target numbers
    gridMaterial=NULL;
    this->getContext()->get( gridMaterial, core::objectmodel::BaseContext::SearchRoot);
    if ( !gridMaterial )
    {
        sout << "GridMaterial component not found -> use model vertices as Gauss point and 1/d^2 as weights." << sendl;
    }
    else
    {
        initFrames();
        initSamples();
    }

    //   unsigned int numParents = this->fromModel->getSize();
    unsigned int numChildren = this->toModel->getSize();
    ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    // should be position0.. but does not work with extvec3f

    //WriteAccessor<PointData<OutCoord> > initPos(this->f_initPos);
    vector<OutCoord>& initPos = *(f_initPos.beginEdit());

    if( this->f_initPos.getValue().size() != numChildren )
    {
        initPos.resize(out.size());
        for(unsigned int i=0; i<out.size(); i++ )
            initPos[i] = out[i];
    }
    f_initPos.endEdit();


    // Get the topology of toModel
    sofa::component::loader::MeshObjLoader *meshobjLoader;
    this->getContext()->get( meshobjLoader, core::objectmodel::BaseContext::Local);
    if (meshobjLoader)
    {
        triangles.assign(meshobjLoader->triangles.getValue().begin(),meshobjLoader->triangles.getValue().end());
        trianglesGroups.assign(meshobjLoader->trianglesGroups.getValue().begin(),meshobjLoader->trianglesGroups.getValue().end());
        std::cout<<"FrameBlendingMapping: Import triangle groups: ";
        for(unsigned int i=0; i<	trianglesGroups.size(); i++) std::cout<<		trianglesGroups[i]<<",";
        std::cout<<std::endl;
    }

    if(triangles.size() && trianglesGroups.size())
    {
        vector<unsigned int>& groups = *(f_groups.beginEdit());
        groups.resize(out.size()); groups.fill(0);
        for(unsigned int i=0; i<trianglesGroups.size(); i++)
            for(unsigned int j=0; j<(unsigned int)trianglesGroups[i].nbp; j++)
                if(j+trianglesGroups[i].p0<triangles.size())
                    for(unsigned int k=0; k<3; k++)
                        groups[triangles[j+trianglesGroups[i].p0][k]]=i;
        f_groups.endEdit();
    }
    this->getToModel()->getContext()->get(to_topo); // Get the output model topology to manage eventualy changes


    // init weights and sample info (mass, moments) todo: ask the Material
    updateWeights();

    // init jacobians for mapping
    if(useDQ.getValue())
    {
        vector<DQBlending>& dqInOut = *(dq_blending.beginEdit());
        dqInOut.resize( out.size() );
        for(unsigned int i=0; i<out.size(); i++ )
        {
            dqInOut[i].init(
                this->f_initPos.getValue()[i],
                f_index.getValue()[i],
                this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),
                weight.getValue()[i],
                weightDeriv.getValue()[i],
                weightDeriv2.getValue()[i]
            );
        }
        dq_blending.endEdit();
    }
    else
    {
        vector<Blending>& inOut = *(blending.beginEdit());
        inOut.resize( out.size() );
        for(unsigned int i=0; i<out.size(); i++ )
        {
            inOut[i].init(
                this->f_initPos.getValue()[i],
                f_index.getValue()[i],
                this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),
                weight.getValue()[i],
                weightDeriv.getValue()[i],
                weightDeriv2.getValue()[i]
            );
        }
        blending.endEdit();
    }

    // Create specific handler for the different PointData
    if (!useDQ.getValue())
    {
        blending.createTopologicalEngine(to_topo);
        blending.registerTopologicalData();
    }
    //else
    //{
    //    dqinout.createTopologicalEngine(to_topo);
    //    dqinout.registerTopologicalData();
    //}

    f_initPos.createTopologicalEngine(to_topo, pointHandler);
    f_initPos.registerTopologicalData();

    f_index.createTopologicalEngine(to_topo);
    f_index.registerTopologicalData();

    f_groups.createTopologicalEngine(to_topo);
    f_groups.registerTopologicalData();

    weight.createTopologicalEngine(to_topo);
    weight.registerTopologicalData();

    weightDeriv.createTopologicalEngine(to_topo);
    weightDeriv.registerTopologicalData();

    weightDeriv2.createTopologicalEngine(to_topo);
    weightDeriv2.registerTopologicalData();


    static_cast<simulation::tree::GNode*>(static_cast<simulation::tree::GNode*>(this->getContext())->getParent())->get ( physicalMapping, core::objectmodel::BaseContext::SearchDown );
    //      if (!physicalMapping) serr << "Unable to get physical mapping" << sendl;

    Inherit::init();
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::apply( InCoord& coord, const InCoord& restCoord)
{
    if(!gridMaterial) { serr << "No GridMaterial !! on single point apply call" << sendl; return;}

    Vec<nbRef,InReal> w;
    Vec<nbRef,unsigned int> reps;
    MaterialCoord restPos = restCoord.getCenter();
    int hexaID = gridMaterial->getIndex( restPos);
    if (hexaID == -1) return;

    gridMaterial->getWeights( w, hexaID );
    gridMaterial->getIndices( reps, hexaID );

    // Allocates and initialises mapping data.
    defaulttype::LinearBlending<In,In,GridMat,nbRef, defaulttype::OutDataTypesInfo<In>::type > map;
    defaulttype::DualQuatBlending<In,In,GridMat,nbRef, defaulttype::OutDataTypesInfo<In>::type > dqmap;
    if(useDQ.getValue()) dqmap.init(restCoord,reps,this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),w,Vec<nbRef,MaterialDeriv>(),Vec<nbRef,MaterialMat>());
    else map.init(restCoord,reps,this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),w,Vec<nbRef,MaterialDeriv>(),Vec<nbRef,MaterialMat>());

    // Transforms the point depending of the current 'in' position.
    ReadAccessor<Data<VecInCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
    if(useDQ.getValue()) {coord = dqmap.apply( in.ref());}
    else {coord = map.apply( in.ref());}
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::apply( typename SData::MaterialCoord& coord, const typename SData::MaterialCoord& restCoord)
{
    if(!gridMaterial) { serr << "No GridMaterial !! on single point apply call" << sendl; return;}

    Vec<nbRef,InReal> w;
    Vec<nbRef,unsigned int> reps;
    unsigned int hexaID = gridMaterial->getIndex( (MaterialCoord)restCoord);

    gridMaterial->getWeights( w, hexaID );
    gridMaterial->getIndices( reps, hexaID );

    // Allocates and initialises mapping data.
    defaulttype::LinearBlending<In,Vec3dTypes,GridMat,nbRef, 0 > map;
    defaulttype::DualQuatBlending<In,Vec3dTypes,GridMat,nbRef, 0 > dqmap;
    if(useDQ.getValue()) dqmap.init(restCoord,reps,this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),w,Vec<nbRef,MaterialDeriv>(),Vec<nbRef,MaterialMat>());
    else map.init(restCoord,reps,this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),w,Vec<nbRef,MaterialDeriv>(),Vec<nbRef,MaterialMat>());

    // Transforms the point depending of the current 'in' position.
    ReadAccessor<Data<VecInCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));
    if(useDQ.getValue()) {coord = dqmap.apply( in.ref());}
    else {coord = map.apply( in.ref());}
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    checkForChanges();

    if( this->f_printLog.getValue() ) std::cerr<<"FrameBlendingMapping<TIn, TOut>::apply, in = "<< in << std::endl;
    if(useDQ.getValue())
        for ( unsigned int i = 0 ; i < out.size(); i++ )
        {
            out[i] = dq_blending[i].apply( in );
            //                    if( this->f_printLog.getValue() )   std::cerr<<"FrameBlendingMapping<TIn, TOut>::apply, out = "<< out[i] << std::endl;
        }
    else
        for ( unsigned int i = 0 ; i < out.size(); i++ )
        {
            out[i] = blending[i].apply( in );
            if( this->f_printLog.getValue() )     std::cerr<<"FrameBlendingMapping<TIn, TOut>::apply, out = "<< out[i] << std::endl;
        }

}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    //                 std::cout<<"maskto="<<   maskTo<<std::endl;
    //                if (this->maskTo) std::cout<<"masktoUse="<<   this->maskTo->isInUse()<<std::endl;

    if ((!this->maskTo)||(this->maskTo&& !(this->maskTo->isInUse())) )
    {
        if( this->f_printLog.getValue() ) std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJ, in = "<< in << std::endl;
        if(useDQ.getValue())
            for ( unsigned int i=0; i<out.size(); i++ )
            {
                out[i] = dq_blending[i].mult( in );
                //if( this->f_printLog.getValue() ) std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJ, out = "<< out[i] << std::endl;
            }
        else
            for ( unsigned int i=0; i<out.size(); i++ )
            {
                out[i] = blending[i].mult( in );
                //if( this->f_printLog.getValue() )     std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJ, out = "<< out[i] << std::endl;
            }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();
        ParticleMask::InternalStorage::const_iterator it;
        if(useDQ.getValue())
            for ( it=indices.begin(); it!=indices.end(); it++ )
            {
                unsigned int i= ( unsigned int ) ( *it );
                out[i] = dq_blending[i].mult( in );
            }
        else
            for ( it=indices.begin(); it!=indices.end(); it++ )
            {
                unsigned int i= ( unsigned int ) ( *it );
                out[i] = blending[i].mult( in );
            }
    }
}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    if((!this->maskTo)||(this->maskTo&& !(this->maskTo->isInUse())) )
    {
        if( this->f_printLog.getValue() )  std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, parent values before = "<< out << std::endl;
        if(useDQ.getValue())
            for ( unsigned int i=0; i<in.size(); i++ ) // VecType
            {
                dq_blending[i].addMultTranspose( out, in[i] );
                //if( this->f_printLog.getValue() )     std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, child value = "<< in[i] << std::endl;
            }
        else
            for ( unsigned int i=0; i<in.size(); i++ ) // VecType
            {
                blending[i].addMultTranspose( out, in[i] );
                //if( this->f_printLog.getValue() )    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, child value = "<< in[i] << std::endl;
            }

        if( this->f_printLog.getValue() )  std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, parent values after = "<< out << std::endl;
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();
        ReadAccessor<Data<vector<Vec<nbRef,unsigned int> > > > index ( f_index );

        ParticleMask::InternalStorage::const_iterator it;
        if( this->f_printLog.getValue() )     std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, use mask, parent values before = "<< out << std::endl;

        if(useDQ.getValue())
            for ( it=indices.begin(); it!=indices.end(); it++ ) // VecType
            {
                const int i= ( int ) ( *it );
                dq_blending[i].addMultTranspose( out, in[i] );
                // if( this->f_printLog.getValue() )      std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, child value = "<< in[i] << std::endl;
                for (unsigned int j = 0; j < nbRef; ++j)
                    maskFrom->insertEntry ( index[i][j] );
            }
        else
            for ( it=indices.begin(); it!=indices.end(); it++ ) // VecType
            {
                const int i= ( int ) ( *it );
                blending[i].addMultTranspose( out, in[i] );
                // if( this->f_printLog.getValue() )    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, child value = "<< in[i] << std::endl;
                for (unsigned int j = 0; j < nbRef; ++j)
                    maskFrom->insertEntry ( index[i][j] );
            }

        if( this->f_printLog.getValue() )    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, parent values after = "<< out << std::endl;
    }
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::MatrixDeriv& parentJacobians, const typename Out::MatrixDeriv& childJacobians )
{

    for (typename Out::MatrixDeriv::RowConstIterator childJacobian = childJacobians.begin(); childJacobian != childJacobians.end(); ++childJacobian)
    {
        typename In::MatrixDeriv::RowIterator parentJacobian = parentJacobians.writeLine(childJacobian.index());

        if(useDQ.getValue())
            for (typename Out::MatrixDeriv::ColConstIterator childParticle = childJacobian.begin(); childParticle != childJacobian.end(); ++childParticle)
            {
                unsigned int childIndex = childParticle.index();
                const OutDeriv& childJacobianVec = childParticle.val();

                dq_blending[childIndex].addMultTranspose( parentJacobian, childJacobianVec );
            }
        else
            for (typename Out::MatrixDeriv::ColConstIterator childParticle = childJacobian.begin(); childParticle != childJacobian.end(); ++childParticle)
            {
                unsigned int childIndex = childParticle.index();
                const OutDeriv& childJacobianVec = childParticle.val();

                blending[childIndex].addMultTranspose( parentJacobian, childJacobianVec );
            }
    }
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::initFrames( const bool& setFramePos, const bool& updateFramePosFromOldOne)
{
    if( targetFrameNumber.getValue() == 0) return; // use user-defined frames

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Init_Frames");
#endif
    // Get references
    WriteAccessor<Data<VecInCoord> > xfrom0 = *this->fromModel->write(core::VecCoordId::restPosition());
    WriteAccessor<Data<VecInCoord> >  xfrom = *this->fromModel->write(core::VecCoordId::position());
    WriteAccessor<Data<VecInCoord> >  xfromReset = *this->fromModel->write(core::VecCoordId::resetPosition());
    unsigned int num_points=xfrom0.size();

    if (setFramePos)
    {
        // ignore if one frame initialized at 0 (done by the mechanical object, not the user)
        if(num_points==1)
        {
            unsigned int i=0; while(i!=num_spatial_dimensions && xfrom0[0][i]==0) i++;
            if(i==num_spatial_dimensions) num_points=0;
        }
    }

    // retrieve initial frames
    vector<SpatialCoord> points(num_points);
    for ( unsigned int i=0; i<num_points; i++ )
        for ( unsigned int j=0; j<num_spatial_dimensions; j++ )
            points[i][j]= xfrom0[i][j];

    // Insert new frames and compute associated voxel weights
    if(num_points>=targetFrameNumber.getValue()) std::cout<<"Inserting 0 frames..."<<std::endl;
    else std::cout<<"Inserting "<<targetFrameNumber.getValue()-num_points<<" frames..."<<std::endl;
    if(initializeFramesInRigidParts.getValue()) gridMaterial->rigidPartsSampling(points);
    gridMaterial->computeUniformSampling(points,targetFrameNumber.getValue());
    std::cout<<"Computing weights in grid..."<<std::endl;
    gridMaterial->computeWeights(points);

    if (setFramePos)
    {
        if(updateFramePosFromOldOne)
        {
#ifdef SOFA_DUMP_VISITOR_INFO
            simulation::Visitor::printNode("Map_new_Frame_State_From_Old_One");
#endif
            vector<InCoord> newRestStates, newStates;
            newRestStates.resize(points.size());
            newStates.resize(points.size());
            // Init restStates from points ( = new pos after lloyd)
            for ( unsigned int i=0; i<points.size(); i++ )
                for ( unsigned int j=0; j<num_spatial_dimensions; j++ )
                    newRestStates[i][j] = points[i][j];
            // Transform these points to obtain the new deformation
            for ( unsigned int i=0; i<points.size(); i++ )
                apply( newStates[i], newRestStates[i]);
            // Store the new DOFs
            for ( unsigned int i=0; i<points.size(); i++ )
            {
                xfrom0[i] = xfromReset[i] = newRestStates[i];
                xfrom[i] = newStates[i];
            }
#ifdef SOFA_DUMP_VISITOR_INFO
            simulation::Visitor::printCloseNode("Map_new_Frame_State_From_Old_One");
#endif
        }
        else
        {
            //// copy the position only
            this->fromModel->resize(points.size());
            for ( unsigned int i=num_points; i<points.size(); i++ )
                for ( unsigned int j=0; j<num_spatial_dimensions; j++ )
                    xfrom[i][j] = xfrom0[i][j] = xfromReset[i][j]=  points[i][j];
        }

    }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Init_Frames");
#endif
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::initSamples()
{
    if(!this->isPhysical)  return; // no gauss point -> use visual/collision or used-define points

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Init_Samples");
#endif
    WriteAccessor<Data<VecOutCoord> >  xto0 = *this->toModel->write(core::VecCoordId::restPosition());
    WriteAccessor<Data<typename defaulttype::OutDataTypesInfo<Out>::VecMaterialCoord> >  points(this->f_materialPoints);

    if(this->targetSampleNumber.getValue()==0)   // use user-defined samples
    {
        // update voronoi in gridmaterial
        vector<MaterialCoord> p(xto0.size());
        for(unsigned int i=0; i<xto0.size(); i++ )
            for ( unsigned int j=0; j<num_spatial_dimensions; j++ )
                p[i][j] = xto0[i][j];
        gridMaterial->computeGeodesicalDistances ( p );
    }
    else  // use automatic sampling
    {
        // Get references
        WriteAccessor<Data<VecOutCoord> >  xto = *this->toModel->write(core::VecCoordId::position());
        WriteAccessor<Data<VecOutCoord> >  xtoReset = *this->toModel->write(core::VecCoordId::resetPosition());

        core::behavior::MechanicalState< Out >* mstateto = this->toModel ? dynamic_cast<core::behavior::MechanicalState< Out >* >( this->toModel.get()) : 0;
        if ( !mstateto)
        {
            serr << "Error: try to insert new samples, which are not mechanical states !" << sendl;
#ifdef SOFA_DUMP_VISITOR_INFO
            simulation::Visitor::printCloseNode("Init_Samples");
#endif
            return;
        }

        vector<MaterialCoord> p;

        // Insert new samples

        //gridMaterial->computeRegularSampling(p,3);
        //gridMaterial->computeUniformSampling(p,targetSampleNumber.getValue());
        gridMaterial->computeLinearRegionsSampling(p,targetSampleNumber.getValue());

        std::cout<<"Inserting "<<p.size()<<" gauss points..."<<std::endl;

        // copy to out
        this->toModel->resize(p.size());
        for ( unsigned int i=0; i<p.size(); i++ )
        {
            xto[i].clear(); xto0[i].clear(); xtoReset[i].clear();
            for ( unsigned int j=0; j<num_spatial_dimensions; j++ )
                xto[i][j] = xto0[i][j] = xtoReset[i][j]= p[i][j];
        }
    }

    gridMaterial->updateSampleMaterialProperties();
    // copy to sampledata
    points.resize(xto0.size());
    for(unsigned int i=0; i<xto0.size(); i++ )
        for ( unsigned int j=0; j<num_spatial_dimensions; j++ )
            points[i][j] = xto0[i][j];

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Init_Samples");
#endif
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::updateWeights ()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Update_Weights");
#endif
    const vector<OutCoord>& xto = f_initPos.getValue();
    ReadAccessor<Data<VecInCoord> > xfrom = *this->fromModel->read(core::ConstVecCoordId::restPosition());
    WriteAccessor<Data<vector<Vec<nbRef,InReal> > > >       m_weights  ( weight );
    WriteAccessor<Data<vector<Vec<nbRef,MaterialDeriv> > > > m_dweight  ( weightDeriv );
    WriteAccessor<Data<vector<Vec<nbRef,MaterialMat> > > >   m_ddweight ( weightDeriv2 );
    WriteAccessor<Data<vector<Vec<nbRef,unsigned int> > > > index ( f_index );

    index.resize( xto.size() );
    m_weights.resize ( xto.size() );
    //if(primitiveorder > 0)
    m_dweight.resize ( xto.size() );
    //if(primitiveorder > 1)
    m_ddweight.resize( xto.size() );

    for (unsigned int i = 0; i < xto.size(); ++i)
    {
        index[i].clear();
        m_weights[i].clear();
        m_dweight[i].clear();
        m_ddweight[i].clear();
    }

    if(useLinearWeights.getValue())  // linear weights based on 2 closest primitives
    {
        for (unsigned int i=0; i<xto.size(); i++ )
        {
            Vec<3,InReal> cto; Out::get( cto[0],cto[1],cto[2], xto[i] ); // OutReal??

            // get the 2 closest primitives
            for (unsigned int j=0; j<nbRef; j++ )
            {
                m_weights[i][j]=0; index[i][j]=0;
                m_dweight[i][j].fill(0);
                m_ddweight[i][j].fill(0);
            }
            m_weights[i][0]=std::numeric_limits<InReal>::max();
            m_weights[i][1]=std::numeric_limits<InReal>::max();
            for (unsigned int j=0; j<xfrom.size(); j++ )
            {
                Vec<3,InReal> cfrom; In::get( cfrom[0],cfrom[1],cfrom[2], xfrom[j] );
                InReal d=(cto-cfrom).norm();
                if(m_weights[i][0]>d) {m_weights[i][1]=m_weights[i][0]; index[i][1]=index[i][0]; m_weights[i][0]=d; index[i][0]=j; }
                else if(m_weights[i][1]>d) {m_weights[i][1]=d; index[i][1]=j;}
            }
            // compute weight
            Vec<3,InReal> cfrom1; In::get( cfrom1[0],cfrom1[1],cfrom1[2], xfrom[index[i][0]] );
            Vec<3,InReal> cfrom2; In::get( cfrom2[0],cfrom2[1],cfrom2[2], xfrom[index[i][1]] );
            Vec<3,InReal> u=cfrom2-cfrom1;
            InReal d=u.norm2(); u=u/d;
            InReal w2=dot(cto-cfrom1,u),w1=-dot(cto-cfrom2,u);
            if(w1<=0) {m_weights[i][0]=0; m_weights[i][1]=1;}
            else if(w2<=0) {m_weights[i][0]=1; m_weights[i][1]=0;}
            else
            {
                m_weights[i][0]=w1; m_weights[i][1]=w2;
                m_dweight[i][0]=-u; m_dweight[i][1]=u;
            }
        }
    }
    else if(gridMaterial)
    {
        if(this->isPhysical) std::cout<<"Lumping weights to gauss points..."<<std::endl;
        SpatialCoord point;

        for (unsigned int i=0; i<xto.size(); i++ )
        {
            Out::get(point[0],point[1],point[2], xto[i]);

            if(!this->isPhysical)  // no gauss point here -> interpolate weights in the grid
            {
                if(restrictInterpolationToLabel.getValue().size()==0) // general case=no restriction
                    gridMaterial->interpolateWeightsRepartition(point,index[i],m_weights[i]);
                else if(restrictInterpolationToLabel.getValue().size()==1) // global restriction for all points
                    gridMaterial->interpolateWeightsRepartition(point,index[i],m_weights[i],restrictInterpolationToLabel.getValue()[0]);
                else if(restrictInterpolationToLabel.getValue().size()==xto.size()) // restriction defined for each point
                    gridMaterial->interpolateWeightsRepartition(point,index[i],m_weights[i],restrictInterpolationToLabel.getValue()[i]);
                else if(restrictInterpolationToLabel.getValue().size()>=trianglesGroups.size() && f_groups.getValue().size()!=0) // restriction defined for each group
                    gridMaterial->interpolateWeightsRepartition(point,index[i],m_weights[i],restrictInterpolationToLabel.getValue()[f_groups.getValue()[i]]);
                else // global restriction for all points
                    gridMaterial->interpolateWeightsRepartition(point,index[i],m_weights[i],restrictInterpolationToLabel.getValue()[0]);
            }
            else // gauss points generated -> approximate weights over a set of voxels by least squares fitting
                gridMaterial->lumpWeightsRepartition(i,point,index[i],m_weights[i],&m_dweight[i],&m_ddweight[i]);
        }
    }

    else	// 1/d^2 weights with Euclidean distance
    {
        for (unsigned int i=0; i<xto.size(); i++ )
        {
            Vec<3,InReal> cto; Out::get( cto[0],cto[1],cto[2], xto[i] ); // OutReal??
            // get the nbRef closest primitives
            for (unsigned int j=0; j<nbRef; j++ )
            {
                m_weights[i][j]=0;
                index[i][j]=0;
            }
            for (unsigned int j=0; j<xfrom.size(); j++ )
            {
                Vec<3,InReal> cfrom; In::get( cfrom[0],cfrom[1],cfrom[2], xfrom[j] );
                InReal w=(cto-cfrom)*(cto-cfrom);
                if(w!=0) w=1./w;
                else w=std::numeric_limits<InReal>::max();
                unsigned int m=0; while (m!=nbRef && m_weights[i][m]>w) m++;
                if(m!=nbRef)
                {
                    for (unsigned int k=nbRef-1; k>m; k--)
                    {
                        m_weights[i][k]=m_weights[i][k-1];
                        index[i][k]=index[i][k-1];
                    }
                    m_weights[i][m]=w;
                    index[i][m]=j;
                }
            }
            // compute weight gradients
            for (unsigned int j=0; j<nbRef; j++ )
            {
                InReal w=m_weights[i][j];
                m_dweight[i][j].fill(0);
                m_ddweight[i][j].fill(0);
                if (w)
                {
                    InReal w2=w*w,w3=w2*w;
                    Vec<3,InReal> u;
                    for(unsigned int k=0; k<3; k++)
                        u[k]=(xto[i][k]-xfrom[index[i][j]][k]);
                    m_dweight[i][j] = - u * w2* 2.0;
                    for(unsigned int k=0; k<num_spatial_dimensions; k++)
                        m_ddweight[i][j][k][k]= - w2* 2.0;
                    for(unsigned int k=0; k<num_spatial_dimensions; k++)
                        for(unsigned int m=0; m<num_spatial_dimensions; m++)
                            m_ddweight[i][j][k][m]+=u[k]*u[m]*w3* 8.0;
                }
            }
        }
    }

    normalizeWeights();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Update_Weights");
#endif
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::normalizeWeights()
{
    const unsigned int xtoSize = this->toModel->getX()->size();
    WriteAccessor<Data<vector<Vec<nbRef,InReal> > > >       m_weights  ( weight );
    WriteAccessor<Data<vector<Vec<nbRef,MaterialDeriv> > > > m_dweight  ( weightDeriv );
    WriteAccessor<Data<vector<Vec<nbRef,MaterialMat> > > >   m_ddweight ( weightDeriv2 );

    for (unsigned int i = 0; i < xtoSize; ++i)
    {
        InReal sumWeights = 0,wn;
        MaterialDeriv sumGrad,dwn;			sumGrad.fill(0);
        MaterialMat sumGrad2,ddwn;				sumGrad2.fill(0);

        // Compute norm
        for (unsigned int j = 0; j < nbRef; ++j)
        {
            sumWeights += m_weights[i][j];
            sumGrad += m_dweight[i][j];
            sumGrad2 += m_ddweight[i][j];
        }

        // Normalise
        if(sumWeights!=0)
        {
            for (unsigned int j = 0; j < nbRef; ++j)
            {
                wn=m_weights[i][j]/sumWeights;
                dwn=(m_dweight[i][j] - sumGrad*wn)/sumWeights;
                for(unsigned int o=0; o<num_material_dimensions; o++)
                {
                    for(unsigned int p=0; p<num_material_dimensions; p++)
                    {
                        ddwn[o][p]=(m_ddweight[i][j][o][p] - wn*sumGrad2[o][p] - sumGrad[o]*dwn[p] - sumGrad[p]*dwn[o])/sumWeights;
                    }
                }
                m_ddweight[i][j]=ddwn;
                m_dweight[i][j]=dwn;
                m_weights[i][j] =wn;
            }
        }
    }
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::LumpMassesToFrames (MassVector& f_mass0, MassVector& f_mass)
{

    ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::restPosition()));
    ReadAccessor<Data<VecInCoord> > in (*this->fromModel->read(core::ConstVecCoordId::restPosition()));

    if(!this->isPhysical) return; // no gauss point here -> no need for lumping

    MassVector& massVector = f_mass0;
    massVector.resize(in.size());
    for(unsigned int i=0; i<in.size(); i++) { massVector[i].clear(); massVector[i].mass = 1.0;}

    vector<Vec<nbRef,unsigned int> > reps;
    vector<Vec<nbRef,InReal> > w;
    vector<SpatialCoord> pts;
    vector<InReal> masses;

    VecInDeriv d(in.size()),m(in.size());

    defaulttype::LinearBlending<In,Vec3dTypes,GridMat,nbRef, 0 > map;
    defaulttype::DualQuatBlending<In,Vec3dTypes,GridMat,nbRef, 0 > dqmap;

    for(unsigned int i=0; i<out.size(); i++) // treat each sample
    {
        if(gridMaterial)  gridMaterial->getWeightedMasses(i,reps,w,pts,masses);
        else
        {
            SpatialCoord point;
            Out::get(point[0],point[1],point[2], out[i]) ;
            pts.clear(); pts.push_back(point);
            w.clear(); w.push_back(weight.getValue()[i]);
            reps.clear(); reps.push_back(f_index.getValue()[i]);
            masses.clear();  masses.push_back(1);	// default value for the mass when model vertices are used as gauss points
        }

        for(unsigned int j=0; j<pts.size(); j++) // treat each voxel j of the sample
        {
            if(useDQ.getValue()) dqmap.init(pts[j],reps[j],this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),w[j],Vec<nbRef,MaterialDeriv>(),Vec<nbRef,MaterialMat>());
            else map.init(pts[j],reps[j],this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),w[j],Vec<nbRef,MaterialDeriv>(),Vec<nbRef,MaterialMat>());

            for(unsigned int k=0; k<nbRef && w[j][k]>0 ; k++) // treat each primitive influencing the voxel
            {
                unsigned int findex=reps[j][k];
                for(unsigned int l=0; l<InVSize; l++)	// treat each dof of the primitive
                {
                    d[findex][l]=1;
                    m[findex].clear();
                    if(useDQ.getValue()) dqmap.addMultTranspose( m , dqmap.mult(d) );  // get the contribution of j to each column l of the mass = mass(j).J^T.J.[0...1...0]^T
                    else map.addMultTranspose( m , map.mult(d) );  // get the contribution of j to each column l of the mass = mass(j).J^T.J.[0...1...0]^T
                    m[findex]*=masses[j];
                    for(unsigned int col=0; col<InVSize; col++)  massVector[findex].inertiaMatrix[col][l]+= m[findex][col];
                    d[reps[j][k]][l]=0;
                }
            }
        }
    }

    for(unsigned int i=0; i<in.size(); i++)  massVector[i].recalc();

    // copy mass0 to current mass
    f_mass = massVector;

    //for(unsigned int i=0;i<in.size();i++) std::cout<<"mass["<<i<<"]="<<massVector[i].inertiaMassMatrix<<std::endl;
}





template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    const typename Out::VecCoord& xto = *this->toModel->getX();
    const typename In::VecCoord& xfrom = *this->fromModel->getX();
    ReadAccessor<Data<vector<Vec<nbRef,unsigned int> > > > index = this->f_index;
    ReadAccessor<Data<vector<Vec<nbRef,InReal> > > > m_weights = weight ;
    ReadAccessor<Data<vector<Vec<nbRef,MaterialDeriv> > > >  m_dweights = weightDeriv ;
    const int valueScale = showValuesNbDecimals.getValue();
    int scale = 1;
    for (int i = 0; i < valueScale; ++i) scale *= 10;
    const double textScale = showTextScaleFactor.getValue();

    glDisable ( GL_LIGHTING );

    if ( vparams->displayFlags().getShowMappings() )
    {
        // Display mapping links between in and out elements
        glDisable ( GL_LIGHTING );
        glPointSize ( 1 );
        glColor4f ( 1,1,0,1 );
        glBegin ( GL_LINES );

        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int m=0 ; m<nbRef; m++ )
            {
                const int idxReps=index[i][m];
                double coef = m_weights[i][m];
                if ( coef > 0.0 )
                {
                    glColor4d ( coef,coef,0,1 );
                    glColor4d ( 1,1,1,1 );
                    helper::gl::glVertexT ( xfrom[idxReps].getCenter() );
                    helper::gl::glVertexT ( defaulttype::center(xto[i]) );
                }
            }
        }
        glEnd();
    }

    // Display index for each points
    if ( showReps.getValue())
    {
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            SpatialCoord p;
            Out::get(p[0],p[1],p[2],xto[i]);
            sofa::helper::gl::GlText::draw ( index[i][0]*scale, p, textScale );
        }
    }

    // Display distance gradients values for each points
    if ( showGradientsValues.getValue())
    {
        char txt[100];
        glColor3f( 0.5, 0.5, 0.5);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            bool influenced;
            unsigned int refIndex;
            findIndexInRepartition(influenced, refIndex, i, showFromIndex.getValue()%nbRef);
            if ( influenced)
            {
                const MaterialDeriv& grad = m_dweights[i][refIndex];
                sprintf( txt, "( %i, %i, %i)", (int)(grad[0]*scale), (int)(grad[1]*scale), (int)(grad[2]*scale));
                SpatialCoord p;
                Out::get(p[0],p[1],p[2],xto[i]);
                sofa::helper::gl::GlText::draw ( txt, p, textScale );
            }
        }
    }

    // Display weights for each points
    if ( showWeightsValues.getValue())
    {
        glColor3f( 1.0, 1.0, 1.0);
        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            bool influenced;
            unsigned int indexRep;
            findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);
            if ( influenced)
            {
                SpatialCoord p;
                Out::get(p[0],p[1],p[2],xto[i]);
                sofa::helper::gl::GlText::draw ( (int)(m_weights[i][indexRep]*scale), p, textScale );
            }
        }
    }

    // Display weights gradients for each points
    if ( showGradients.getValue())
    {
        glColor3f ( 0.0, 1.0, 0.3 );
        glBegin ( GL_LINES );
        for ( unsigned int i = 0; i < xto.size(); i++ )
        {
            bool influenced;
            unsigned int indexRep;
            findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);
            if (influenced)
            {
                const MaterialDeriv& gradMap = m_dweights[i][indexRep];
                SpatialCoord point;
                Out::get(point[0],point[1],point[2],xto[i]);
                glVertex3f ( point[0], point[1], point[2] );
                glVertex3f ( point[0] + gradMap[0] * showGradientsScaleFactor.getValue(), point[1] + gradMap[1] * showGradientsScaleFactor.getValue(), point[2] + gradMap[2] * showGradientsScaleFactor.getValue() );
            }
        }
        glEnd();
    }
    //

    // Show weights
    if ( showWeights.getValue())
    {
        // Compute min and max values.
        InReal minValue = std::numeric_limits<InReal>::max();
        InReal maxValue = -std::numeric_limits<InReal>::min();
        for ( unsigned int i = 0; i < xto.size(); i++)
        {
            bool influenced;
            unsigned int indexRep;
            findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);
            if (influenced)
            {
                const InReal& weight = m_weights[i][indexRep];
                if ( weight < minValue && weight != 0xFFF) minValue = weight;
                if ( weight > maxValue && weight != 0xFFF) maxValue = weight;
            }
        }

        if ( ! triangles.empty())
        {
            glPushAttrib( GL_LIGHTING_BIT | GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT);
            std::vector< defaulttype::Vector3 > points;
            std::vector< defaulttype::Vector3 > normals;
            std::vector< defaulttype::Vec<4,float> > colors;
            for ( unsigned int i = 0; i < triangles.size(); i++)
            {
                for ( unsigned int j = 0; j < 3; j++)
                {
                    bool influenced;
                    unsigned int indexRep;
                    //                                findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);  FF
                    findIndexInRepartition(influenced, indexRep, triangles[i][j], showFromIndex.getValue()%nbRef);
                    if (influenced)
                    {
                        const unsigned int& indexPoint = triangles[i][j];
                        float color = (float)(m_weights[indexPoint][indexRep] - minValue) / (maxValue - minValue);
                        color = (float)pow((float)color, (float)showGammaCorrection.getValue());
                        points.push_back(defaulttype::Vector3(xto[indexPoint][0],xto[indexPoint][1],xto[indexPoint][2]));
                        colors.push_back(defaulttype::Vec<4,float>(color, 0.0, 0.0,1.0));
                    }
                }
            }
            vparams->drawTool()->drawTriangles(points, normals, colors);
            glPopAttrib();
        }
        else // Show by points
        {
            glPointSize( 10);
            glBegin( GL_POINTS);
            for ( unsigned int i = 0; i < xto.size(); i++)
            {
                bool influenced;
                unsigned int indexRep;
                findIndexInRepartition(influenced, indexRep, i, showFromIndex.getValue()%nbRef);
                if (influenced)
                {
                    float color = (float)(m_weights[i][indexRep] - minValue) / (maxValue - minValue);
                    color = (float)pow((float)color, (float)showGammaCorrection.getValue());
                    glColor3f( color, 0.0, 0.0);
                    glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
                }
            }
            glEnd();
            glPointSize( 1);
        }
    }

    /*
                // Display def tensor values for each points
                if ( this->showDefTensorsValues.getValue())
                {
                    char txt[100];
                    glColor3f( 0.5, 0.5, 0.5);
                    for ( unsigned int i=0;i<xto.size();i++ )
                    {
                        const Vec6& e = this->deformationTensors[i];
                        sprintf( txt, "( %i, %i, %i)", (int)(e[0]*scale), (int)(e[1]*scale), (int)(e[2]*scale));
                        sofa::helper::gl::GlText::draw ( txt, xto[i], textScale );
                    }
                }
                */

    // Deformation tensor show
    if ( this->showStrain.getValue())
    {
        //               if (!this->isPhysical)
        //               {
        //                   serr << "The Frame Blending Mapping must be physical to display the strain tensors." << sendl;
        //}
        //               else
        {
            glPushAttrib( GL_LIGHTING_BIT | GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT);
            glDisable( GL_LIGHTING);

            typedef Vec<3,double> Vec3;
            if ( ! triangles.empty())
            {
                glBegin( GL_TRIANGLES);
                for ( unsigned int i = 0; i < triangles.size(); i++)
                {
                    for ( unsigned int j = 0; j < 3; j++)
                    {
                        const unsigned int& indexP = triangles[i][j];
                        Vec3 e(0,0,0);
                        // (Ft * F - I) /2.0

                        if (!this->isPhysical) // create and map a def gradient
                        {
                            typedef typename defaulttype::DeformationGradientTypes<num_spatial_dimensions,num_material_dimensions,1,InReal> DefGrad1;
                            typename DefGrad1::Coord out0,out;
                            DefGrad1::set( out0, this->f_initPos.getValue()[indexP][0], this->f_initPos.getValue()[indexP][1], this->f_initPos.getValue()[indexP][2]);

                            if(useDQ.getValue())
                            {
                                defaulttype::DualQuatBlending<In,DefGrad1,GridMat,nbRef, 1 > map;
                                map.init(out0,f_index.getValue()[indexP],this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),weight.getValue()[indexP],weightDeriv.getValue()[indexP],weightDeriv2.getValue()[indexP]);
                                out = map.apply( xfrom );
                            }
                            else
                            {
                                defaulttype::LinearBlending<In,DefGrad1,GridMat,nbRef, 1 > map;
                                map.init(out0,f_index.getValue()[indexP],this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),weight.getValue()[indexP],weightDeriv.getValue()[indexP],weightDeriv2.getValue()[indexP]);
                                out = map.apply( xfrom );
                            }

                            for (unsigned int k = 0; k < 3; ++k)
                            {
                                for (unsigned int l = 0; l < 3; ++l)
                                    e[k] += out[3+3*l+k]*out[3+3*l+k];
                                e[k] -= 1.0;
                                e[k] /= 2.0;
                            }
                        }
                        else
                            for (unsigned int k = 0; k < 3; ++k)
                            {
                                for (unsigned int l = 0; l < 3; ++l)
                                    e[k] += xto[indexP][3+3*l+k]*xto[indexP][3+3*l+k];
                                e[k] -= 1.0;
                                e[k] /= 2.0;
                            }
                        float color = ( e[0] + e[1] + e[2])/3.0;
                        if (color<0) color=2*color/(color+1.);
                        color*=1000 * this->showStrainScaleFactor.getValue();
                        color+=120;
                        if (color<0) color=0;
                        if (color>240) color=240;
                        sofa::helper::gl::Color::setHSVA(color,1.,.8,1.);
                        glVertex3f( xto[indexP][0], xto[indexP][1], xto[indexP][2]);
                    }
                }
                glEnd();
            }
            else // Show by points
            {
                glPointSize( 10);
                glBegin( GL_POINTS);
                for ( unsigned int i = 0; i < xto.size(); i++)
                {
                    Vec3 e(0,0,0);
                    // (Ft * F - I) /2.0

                    if (!this->isPhysical) // create and map a def gradient
                    {
                        typedef typename defaulttype::DeformationGradientTypes<num_spatial_dimensions,num_material_dimensions,1,InReal> DefGrad1;
                        typename DefGrad1::Coord out0,out;
                        DefGrad1::set( out0, this->f_initPos.getValue()[i][0], this->f_initPos.getValue()[i][1], this->f_initPos.getValue()[i][2]);

                        if(useDQ.getValue())
                        {
                            defaulttype::DualQuatBlending<In,DefGrad1,GridMat,nbRef, 1 > map;
                            map.init(out0,f_index.getValue()[i],this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),weight.getValue()[i],weightDeriv.getValue()[i],weightDeriv2.getValue()[i]);
                            out = map.apply( xfrom );
                        }
                        else
                        {
                            defaulttype::LinearBlending<In,DefGrad1,GridMat,nbRef, 1 > map;
                            map.init(out0,f_index.getValue()[i],this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),weight.getValue()[i],weightDeriv.getValue()[i],weightDeriv2.getValue()[i]);
                            out = map.apply( xfrom );
                        }

                        for (unsigned int k = 0; k < 3; ++k)
                        {
                            for (unsigned int l = 0; l < 3; ++l)
                                e[k] += out[3+3*l+k]*out[3+3*l+k];
                            e[k] -= 1.0;
                            e[k] /= 2.0;
                        }
                    }
                    else
                        for (unsigned int k = 0; k < 3; ++k)
                        {
                            for (unsigned int l = 0; l < 3; ++l)
                                e[k] += xto[i][3+3*l+k]*xto[i][3+3*l+k];
                            e[k] -= 1.0;
                            e[k] /= 2.0;
                        }

                    float color = ( e[0] + e[1] + e[2])/3.0;
                    if (color<0) color=2*color/(color+1.);
                    color*=1000 * this->showStrainScaleFactor.getValue();
                    color+=120;
                    if (color<0) color=0;
                    if (color>240) color=240;
                    sofa::helper::gl::Color::setHSVA(color,1.,.8,1.);
                    glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
                }
                glEnd();
                glPointSize( 1);
            }
            glPopAttrib();
        }
    }

    // Det F show
    if ( this->showDetF.getValue())
    {
        if (!this->isPhysical)
        {
            serr << "The Frame Blending Mapping must be physical to display the strain tensors." << sendl;
        }
        else
        {
            glPushAttrib( GL_LIGHTING_BIT | GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT);
            glDisable( GL_LIGHTING);
            typedef Vec<3,double> Vec3;
            if ( ! triangles.empty())
            {
                glBegin( GL_TRIANGLES);
                for ( unsigned int i = 0; i < triangles.size(); i++)
                {
                    for ( unsigned int j = 0; j < 3; j++)
                    {
                        const unsigned int& indexP = triangles[i][j];
                        // Det( F )
                        float color = xto[indexP][3]*(xto[indexP][ 7]*xto[indexP][11]-xto[indexP][10]*xto[indexP][ 8])
                                -xto[indexP][6]*(xto[indexP][10]*xto[indexP][ 5]-xto[indexP][ 4]*xto[indexP][11])
                                +xto[indexP][9]*(xto[indexP][ 4]*xto[indexP][ 8]-xto[indexP][ 7]*xto[indexP][ 5]);
                        color=(color-1.)*100 * this->showDetFScaleFactor.getValue();
                        color+=120;
                        if (color<0) color=0;
                        if (color>240) color=240;
                        sofa::helper::gl::Color::setHSVA(color,1.,.8,1.);
                        glVertex3f( xto[indexP][0], xto[indexP][1], xto[indexP][2]);
                    }
                }
                glEnd();
            }
            else // Show by points
            {
                glPointSize( 10);
                glBegin( GL_POINTS);
                for ( unsigned int i = 0; i < xto.size(); i++)
                {
                    // Det( F )
                    float color = xto[i][3]*(xto[7]*xto[11]-xto[10]*xto[8])
                            -xto[i][6]*(xto[10]*xto[5]-xto[4]*xto[11])
                            +xto[i][9]*(xto[4]*xto[8]-xto[7]*xto[5]);
                    color=(color-1.)*100 * this->showDetFScaleFactor.getValue();
                    color+=120;
                    if (color<0) color=0;
                    if (color>240) color=240;
                    sofa::helper::gl::Color::setHSVA(color,1.,.8,1.);
                    glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
                }
                glEnd();
                glPointSize( 1);
            }
            glPopAttrib();
        }
    }
}




template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::findIndexInRepartition( bool& influenced, unsigned int& realIndex, const unsigned int& pointIndex, const unsigned int& frameIndex)
{
    ReadAccessor<Data<vector<Vec<nbRef,unsigned int> > > >  index( f_index );
    influenced = false;
    for ( unsigned int j = 0; j < nbRef; ++j)
    {
        if ( index[pointIndex][j] == frameIndex)
        {
            influenced = true;
            realIndex = j;
            return;
        }
    }
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::updateMapping(const bool& computeWeights)
{
    if(!useAdaptivity.getValue()) return;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Update_Mapping");
#endif
    if (!gridMaterial) return;

    if (this->isPhysical || computeWeights)
    {
        //initFrames( true, true); // With lloyd on frames

        initFrames( false); // Without lloyd on frames


        initSamples();

        gridMaterial->voxelsHaveChanged.setValue (false);
    }

    //ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::restPosition()));
    const VecOutCoord& out = *this->toModel->getX0();
    vector<OutCoord>& initPos = *(f_initPos.beginEdit());
    initPos.resize(out.size());
    for(unsigned int i=0; i<out.size(); i++ )
        initPos[i] = out[i];
    f_initPos.endEdit();

    // init weights and sample info (mass, moments) todo: ask the Material
    updateWeights();

    // init jacobians for mapping
    if(useDQ.getValue())
    {
        vector<DQBlending>& dqInOut = *(dq_blending.beginEdit());
        dqInOut.resize( out.size() );
        for(unsigned int i=0; i<out.size(); i++ )
        {
            dqInOut[i].init(
                this->f_initPos.getValue()[i],
                f_index.getValue()[i],
                this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),
                weight.getValue()[i],
                weightDeriv.getValue()[i],
                weightDeriv2.getValue()[i]
            );
        }
        dq_blending.endEdit();
    }
    else
    {
        vector<Blending>& inOut = *(blending.beginEdit());
        inOut.resize( out.size() );
        for(unsigned int i=0; i<out.size(); i++ )
        {
            inOut[i].init(
                this->f_initPos.getValue()[i],
                f_index.getValue()[i],
                this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),
                weight.getValue()[i],
                weightDeriv.getValue()[i],
                weightDeriv2.getValue()[i]
            );
        }
        blending.endEdit();
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Update_Mapping");
#endif

}



// This method is equivalent to handleTopologyChange() for physical mappings.
template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::checkForChanges()
{
    if(!useAdaptivity.getValue()) return;

    if (this->mappingHasChanged) this->mappingHasChanged = false;
    ReadAccessor<Data<VecInCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));

    bool dofModified = false;
    if (this->isPhysical)
    {
        // Insert a frame depending on the gradient of the deformation tensor.
        ReadAccessor<Data<VecOutCoord> > out  = *this->toModel->read(core::ConstVecCoordId::position());
        for (unsigned int i = 0; i < out.size(); ++i)
        {
            double criteria = 0;
            for (unsigned int k = num_spatial_dimensions+9; k < OutCoord::total_size; ++k) // Check if the second order terms are < epsilon
                criteria += out[i][k] * out[i][k];
            criteria = sqrt( criteria) * pow (gridMaterial->getVolumeForVoronoi(i), 1/3.0);

            serr << "inserting criteria[" << i << "]: " << criteria << sendl;

            if (criteria > this->adaptativeCriteria.getValue()) insertFrame( center(out[i]));
        }

        //* // Remove after a given time
        for (unsigned int i = 0; i < this->frameLife.size();)
        {
            if (this->getContext()->getTime() > this->frameLife[i])
            {
                removeFrame (i);
                dofModified = true;
            }
            else ++i;
        }

        /*/
                    // Remove a frame depending on the gradient of the deformation tensor.
                    ReadAccessor<Data<vector<Vec<nbRef,unsigned int> > > > index ( f_index );
                    ReadAccessor<Data<vector<Vec<nbRef,InReal> > > > m_weights ( weight );
                    for (unsigned int i = 0; i < this->addedFrameIndices.size();)
                    {
                        // Check all the elastons mapped by this frame.
                        double criteria = 0;
                        unsigned int nbElastons = 0;
                        for (unsigned int j = 0; j < index.size(); ++j)
                        {
                            for (unsigned int ref = 0; ref < nbRef; ++ref)
                            {
                                if ( index[j][ref] == this->addedFrameIndices[i] && m_weights[j][ref] != 0) // If this elaston is mapped by the frame
                                {
                                    double elastonCriteria = 0;
                                    for (unsigned int k = num_spatial_dimensions+9; k < OutCoord::total_size; ++k) // Check if the second order terms are < epsilon
                                        elastonCriteria += out[i][k] * out[i][k];
                                    criteria += sqrt( elastonCriteria) * pow (gridMaterial->getVolumeForVoronoi(j), 1/3.0);
                                    nbElastons++;
                                }
                            }
                        }
                        criteria /= nbElastons;

                        serr << "removal criteria[" << i << "]: " << criteria << sendl;

                        if (criteria < this->adaptativeCriteria.getValue())
                        {
                            removeFrame (i);
                            dofModified = true;
                        }
                        else ++i;
                    }
                    //*/
    }


    // Mapping has to be updated
    if ( (in.size() != targetFrameNumber.getValue()) || // In DOFs have changed
            (gridMaterial && gridMaterial->voxelsHaveChanged.getValue()) || // Voxels have changed
            (dofModified))
    {
        targetFrameNumber.setValue(in.size());
        this->mappingHasChanged = true;
        updateMapping (dofModified);
    }
}



template <class TIn, class TOut>
bool FrameBlendingMapping<TIn, TOut>::insertFrame (const Vec3d& pos)
{
    core::behavior::MechanicalState< In >* mstatefrom = this->fromModel ? static_cast<core::behavior::MechanicalState< In >* >( this->fromModel.get()) : 0;
    unsigned int indexFrom = mstatefrom->getSize();

    WriteAccessor<Data<VecInCoord> > xfrom0 = *this->fromModel->write(core::VecCoordId::restPosition());
    WriteAccessor<Data<VecInCoord> >  xfrom = *this->fromModel->write(core::VecCoordId::position());
    WriteAccessor<Data<VecInCoord> >  xfromReset = *this->fromModel->write(core::VecCoordId::resetPosition());

    // Avoid to insert a frame to close from another one (preliminary test to avoid inverseSkinning).
    for (unsigned int i = 0; i < xfrom.size(); ++i)
    {
        const SReal dist=(xfrom[i].getCenter()-pos).norm();
        if (dist < this->newFrameMinDist.getValue()) return false;
    }

    // Compute the rest position of the frame.
    InCoord newX, newX0;
    InCoord targetDOF;
    In::set( targetDOF, pos[0], pos[1], pos[2]);
    if (!inverseApply( newX0, newX, targetDOF)) return false;

    // Test if the frame to insert is not too close of an existing frame.
    for (unsigned int i = 0; i < xfrom0.size(); ++i)
    {
        const SReal dist=(xfrom0[i].getCenter()-newX0.getCenter()).norm();
        if (dist < this->newFrameMinDist.getValue()) return false;
    }

    // Insert a new DOF
    this->fromModel->resize(indexFrom+1);
    xfrom0[indexFrom] = newX0;
    xfrom[indexFrom] = newX;
    xfromReset[indexFrom] = newX0;

    // register new frame in the corresponding physical mapping
    physicalMapping->addedFrameIndices.push_back( indexFrom);
    physicalMapping->frameLife.push_back(this->getContext()->getTime()+100);

    return true;
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::removeFrame (const unsigned int index)
{
    component::container::MechanicalObject< In >* mstatefrom = this->fromModel ? static_cast<component::container::MechanicalObject< In >* >( this->fromModel.get()) : 0;
    mstatefrom->replaceValue (mstatefrom->getSize()-1,this->addedFrameIndices[index]);
    mstatefrom->resize(mstatefrom->getSize()-1);
    this->addedFrameIndices[index] = this->addedFrameIndices[this->addedFrameIndices.size()-1];
    this->addedFrameIndices.resize(this->addedFrameIndices.size()-1);
    this->frameLife[index] = this->frameLife[this->frameLife.size()-1];
    this->frameLife.resize(this->frameLife.size()-1);
}



template <class TIn, class TOut>
bool FrameBlendingMapping<TIn, TOut>::inverseApply( InCoord& restCoord, InCoord& coord, const InCoord& targetCoord)
{
    ReadAccessor<Data<VecInCoord> > xfrom = *this->fromModel->read(core::VecCoordId::position());
    ReadAccessor<Data<VecOutCoord> > xto0 = *this->toModel->read(core::VecCoordId::resetPosition());
    ReadAccessor<Data<VecOutCoord> > xto = *this->toModel->read(core::VecCoordId::position());

    typedef typename defaulttype::DeformationGradientTypes<num_spatial_dimensions,num_material_dimensions,1,InReal> DefGrad1;
    // Get closest material point
    Vec<3,OutReal> t;
    double d,dmin=1E5;
    for (unsigned int i = 0; i < xto.size(); ++i)
    {
        t = targetCoord.getCenter() - center (xto[i]);
        d = t * t;
        if (d<dmin)
        {
            dmin = d;
            restCoord.getCenter() = center (xto0[i]);
        }
    }

    typename GridMat::VRef index;
    Vec<nbRef,InReal> weights;
    double eps=1E-5;
    bool stop=false;
    int count=0;
    typename DefGrad1::MaterialFrame Uinv;
    typename DefGrad1::Coord out, out0;
    DefGrad1::set( out0, restCoord.getCenter()[0], restCoord.getCenter()[1], restCoord.getCenter()[2]);
    while (!stop)
    {
        defaulttype::LinearBlending<In,DefGrad1,GridMat,nbRef, 1 > map;
        defaulttype::DualQuatBlending<In,DefGrad1,GridMat,nbRef, 1 > dqmap;

        if (!gridMaterial->interpolateWeightsRepartition(out0.getCenter(),index,weights)) return false;

        if(useDQ.getValue()) dqmap.init(out0,index,this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),weights,Vec<nbRef,MaterialDeriv>(),Vec<nbRef,MaterialMat>());
        else map.init(out0,index,this->fromModel->read(core::ConstVecCoordId::restPosition())->getValue(),weights,Vec<nbRef,MaterialDeriv>(),Vec<nbRef,MaterialMat>());

        if(useDQ.getValue()) out = dqmap.apply (xfrom.ref());
        else out = map.apply (xfrom.ref());

        //update skinned points
        t = targetCoord.getCenter()- out.getCenter();

        if ( t*t < eps || count >= 10) stop = true;
        count++;

        if (!stop)
        {
            const typename DefGrad1::MaterialFrame& U = out.getMaterialFrame();
            invertMatrix(Uinv,U);
            out0.getCenter() += Uinv * t;
        }
    }

    // If the new frame is out of the grid, displace it to the nearest voxel position.
    int nearestVoxel = gridMaterial->getIndex (out0.getCenter());
    while (gridMaterial->grid.data()[nearestVoxel] == 0)
    {
        vector<unsigned int> neighbors;
        gridMaterial->get26Neighbors( nearestVoxel, neighbors);
        double dist = gridMaterial->getDistance(nearestVoxel);
        for (vector<unsigned int>::const_iterator it = neighbors.begin(); it != neighbors.end(); ++it)
        {
            if (gridMaterial->getDistance(*it) < dist)
            {
                nearestVoxel = *it;
                dist = gridMaterial->getDistance(nearestVoxel);
            }
        }
    }
    gridMaterial->getCoord (nearestVoxel, restCoord.getCenter());

    apply( coord, restCoord);

    return true;
}



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
