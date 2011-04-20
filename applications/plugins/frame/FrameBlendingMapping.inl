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
#ifndef SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_INL
#define SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_INL

#include "FrameBlendingMapping.h"
#include "LinearBlendTypes.inl"
#include "DualQuatBlendTypes.inl"
#include "GridMaterial.inl"
#include <sofa/core/Mapping.inl>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/gl/glText.inl>
#include <sofa/helper/gl/template.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/topology/PointSetTopologyChange.h>
#include <sofa/component/topology/PointData.inl>
#include <iostream>

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

using helper::WriteAccessor;
using helper::ReadAccessor;

template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::FrameBlendingMapping (core::State<In>* from, core::State<Out>* to )
    : Inherit ( from, to ), FData(), SData()
    , useLinearWeights ( initData ( &useLinearWeights, false, "useLinearWeights","use linearly interpolated weights between the two closest frames." ) )
    , useDQ ( initData ( &useDQ, false, "useDQ","use dual quaternion blending instead of linear blending ." ) )
    , f_initPos ( initData ( &f_initPos,"initPos","initial child coordinates in the world reference frame" ) )
    , f_index ( initData ( &f_index,"indices","parent indices for each child" ) )
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
    , targetFrameNumber ( initData ( &targetFrameNumber, ( unsigned int ) 0, "targetFrameNumber","Target frames number" ) )
    , initializeFramesInRigidParts ( initData ( &initializeFramesInRigidParts, false, "initializeFramesInRigidParts","Automatically initialize frames in rigid parts if stiffness>15E6." ) )
    , targetSampleNumber ( initData ( &targetSampleNumber, ( unsigned int ) 0, "targetSampleNumber","Target samples number" ) )
    , restrictInterpolationToLabel ( initData ( &restrictInterpolationToLabel, "restrictInterpolationToLabel","Restrict interpolation to a label in gridmaterial." ) )
{
    maskFrom = NULL;
    if ( core::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *> ( from ) )
        maskFrom = &stateFrom->forceMask;
    maskTo = NULL;
    if ( core::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::behavior::BaseMechanicalState *> ( to ) )
        maskTo = &stateTo->forceMask;

    // These sout are here to check if the template interface work well
    sout << "In VSize: " << defaulttype::InDataTypesInfo<In>::VSize << sendl;
    sout << "Out order: " << defaulttype::OutDataTypesInfo<TOut>::primitive_order << sendl;
}

template <class TIn, class TOut>
FrameBlendingMapping<TIn, TOut>::~FrameBlendingMapping ()
{
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::init()
{
    // init samples and frames according to target numbers
    gridMaterial=NULL;
    this->getContext()->get( gridMaterial, core::objectmodel::BaseContext::SearchRoot);
    if ( !gridMaterial )
    {
        serr << "GridMaterial component not found -> use model vertices as Gauss point and 1/d^2 as weights." << sendl;
    }
    else
    {
        initFrames();
        initSamples();
    }

    //   unsigned int numParents = this->fromModel->getSize();
    unsigned int numChildren = this->toModel->getSize();
    ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::restPosition()));
    //WriteAccessor<PointData<OutCoord> > initPos(this->f_initPos);
    vector<OutCoord>& initPos = *(f_initPos.beginEdit());

    if( this->f_initPos.getValue().size() != numChildren )
    {
        initPos.resize(out.size());
        for(unsigned int i=0; i<out.size(); i++ )
            initPos[i] = out[i];
    }
    f_initPos.endEdit();


    // init weights and sample info (mass, moments) todo: ask the Material
    updateWeights();

    // init jacobians for mapping
    if(useDQ.getValue())
    {
        vector<DQInOut>& dqInOut = *(dqinout.beginEdit());
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
        dqinout.endEdit();
    }
    else
    {
        vector<InOut>& inOut = *(inout.beginEdit());
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
        inout.endEdit();
    }

    // Get the topology of toModel to display the weights and the strain tensors on the triangular model.
    MeshLoader *meshLoader;
    this->getContext()->get( meshLoader, core::objectmodel::BaseContext::Local);
    if (meshLoader)
    {
        meshLoader->getTriangles(triangles);
    }

    this->getToModel()->getContext()->get(to_topo); // Get the output model topology to manage eventualy changes

    Inherit::init();
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::apply( InCoord& coord, const InCoord& restCoord)
{
    if(!gridMaterial) { serr << "No GridMaterial !! on single point apply call" << sendl; return;}

    Vec<nbRef,InReal> w;
    Vec<nbRef,unsigned int> reps;
    MaterialCoord restPos = restCoord.getCenter();
    unsigned int hexaID = gridMaterial->getIndex( restPos);

    gridMaterial->getWeights( w, hexaID );
    gridMaterial->getIndices( reps, hexaID );

    // Allocates and initialises mapping data.
    defaulttype::LinearBlendTypes<In,In,GridMat,nbRef, defaulttype::OutDataTypesInfo<In>::type > map;
    defaulttype::DualQuatBlendTypes<In,In,GridMat,nbRef, defaulttype::OutDataTypesInfo<In>::type > dqmap;
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
    defaulttype::LinearBlendTypes<In,Vec3dTypes,GridMat,nbRef, 0 > map;
    defaulttype::DualQuatBlendTypes<In,Vec3dTypes,GridMat,nbRef, 0 > dqmap;
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

    //if( this->f_printLog.getValue() ){
    //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::apply, in = "<< in << std::endl;
    //}
    if(useDQ.getValue())
        for ( unsigned int i = 0 ; i < out.size(); i++ )
        {
            out[i] = dqinout[i].apply( in );
            //if( this->f_printLog.getValue() ){
            //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::apply, out = "<< out[i] << std::endl;
            //}
        }
    else
        for ( unsigned int i = 0 ; i < out.size(); i++ )
        {
            out[i] = inout[i].apply( in );
            //if( this->f_printLog.getValue() ){
            //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::apply, out = "<< out[i] << std::endl;
            //}
        }

}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    if ( ! ( this->maskTo->isInUse() ) )
    {
        //if( this->f_printLog.getValue() ){
        //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJ, in = "<< in << std::endl;
        //}
        if(useDQ.getValue())
            for ( unsigned int i=0; i<out.size(); i++ )
            {
                out[i] = dqinout[i].mult( in );
                //if( this->f_printLog.getValue() ){
                //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJ, out = "<< out[i] << std::endl;
                //}
            }
        else
            for ( unsigned int i=0; i<out.size(); i++ )
            {
                out[i] = inout[i].mult( in );
                //if( this->f_printLog.getValue() ){
                //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJ, out = "<< out[i] << std::endl;
                //}
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
                out[i] = dqinout[i].mult( in );
            }
        else
            for ( it=indices.begin(); it!=indices.end(); it++ )
            {
                unsigned int i= ( unsigned int ) ( *it );
                out[i] = inout[i].mult( in );
            }
    }
}

template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    if ( ! ( this->maskTo->isInUse() ) )
    {
        this->maskFrom->setInUse ( false );
        //if( this->f_printLog.getValue() ){
        //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, parent values before = "<< out << std::endl;
        //}
        if(useDQ.getValue())
            for ( unsigned int i=0; i<in.size(); i++ ) // VecType
            {
                dqinout[i].addMultTranspose( out, in[i] );
                //if( this->f_printLog.getValue() ){
                //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, child value = "<< in[i] << std::endl;
                // }
            }
        else
            for ( unsigned int i=0; i<in.size(); i++ ) // VecType
            {
                inout[i].addMultTranspose( out, in[i] );
                //if( this->f_printLog.getValue() ){
                //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, child value = "<< in[i] << std::endl;
                // }
            }

        //if( this->f_printLog.getValue() ){
        //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, parent values after = "<< out << std::endl;
        //}
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();
        ReadAccessor<Data<vector<Vec<nbRef,unsigned int> > > > index ( f_index );

        ParticleMask::InternalStorage::const_iterator it;
        //if( this->f_printLog.getValue() ){
        //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, use mask, parent values before = "<< out << std::endl;
        //}
        if(useDQ.getValue())
            for ( it=indices.begin(); it!=indices.end(); it++ ) // VecType
            {
                const int i= ( int ) ( *it );
                dqinout[i].addMultTranspose( out, in[i] );
                //  if( this->f_printLog.getValue() ){
                //      std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, child value = "<< in[i] << std::endl;
                // }
                for (unsigned int j = 0; j < nbRef; ++j)
                    maskFrom->insertEntry ( index[i][j] );
            }
        else
            for ( it=indices.begin(); it!=indices.end(); it++ ) // VecType
            {
                const int i= ( int ) ( *it );
                inout[i].addMultTranspose( out, in[i] );
                //  if( this->f_printLog.getValue() ){
                //      std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, child value = "<< in[i] << std::endl;
                // }
                for (unsigned int j = 0; j < nbRef; ++j)
                    maskFrom->insertEntry ( index[i][j] );
            }

        //if( this->f_printLog.getValue() ){
        //    std::cerr<<"FrameBlendingMapping<TIn, TOut>::applyJT, parent values after = "<< out << std::endl;
        //}
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

                dqinout[childIndex].addMultTranspose( parentJacobian, childJacobianVec );
            }
        else
            for (typename Out::MatrixDeriv::ColConstIterator childParticle = childJacobian.begin(); childParticle != childJacobian.end(); ++childParticle)
            {
                unsigned int childIndex = childParticle.index();
                const OutDeriv& childJacobianVec = childParticle.val();

                inout[childIndex].addMultTranspose( parentJacobian, childJacobianVec );
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

        core::behavior::MechanicalState< Out >* mstateto = dynamic_cast<core::behavior::MechanicalState< Out >* >( this->toModel);
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
                else if(restrictInterpolationToLabel.getValue().size()==xto.size()) // restriction defined on each point
                    gridMaterial->interpolateWeightsRepartition(point,index[i],m_weights[i],restrictInterpolationToLabel.getValue()[i]);
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

    defaulttype::LinearBlendTypes<In,Vec3dTypes,GridMat,nbRef, 0 > map;
    defaulttype::DualQuatBlendTypes<In,Vec3dTypes,GridMat,nbRef, 0 > dqmap;

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
void FrameBlendingMapping<TIn, TOut>::draw()
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

    if ( this->getShow() )
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
            glPushAttrib( GL_LIGHTING_BIT || GL_COLOR_BUFFER_BIT || GL_ENABLE_BIT);
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
            simulation::getSimulation()->DrawUtility().drawTriangles(points, normals, colors);
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
        if (!this->isPhysical)
        {
            serr << "The Frame Blending Mapping must be physical to display the strain tensors." << sendl;
        }
        else
        {
            glPushAttrib( GL_LIGHTING_BIT || GL_COLOR_BUFFER_BIT || GL_ENABLE_BIT);
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
            glPushAttrib( GL_LIGHTING_BIT || GL_COLOR_BUFFER_BIT || GL_ENABLE_BIT);
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
void FrameBlendingMapping<TIn, TOut>::updateMapping()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Update_Mapping");
#endif

    serr << this->getName() << " (" << this->isPhysical << ") Update Mapping" << sendl;

    if (this->isPhysical)
    {
        serr << "Update Frames" << sendl;

        //*
        initFrames( true, true); // With lloyd on frames
        /*/
        initFrames( false); // Without lloyd on frames
        //*/

        serr << "Update Samples" << sendl;

        initSamples();

        gridMaterial->voxelsHaveChanged.setValue (false);
    }

    ReadAccessor<Data<VecOutCoord> > out  = *this->toModel->read(core::ConstVecCoordId::restPosition());
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
        vector<DQInOut>& dqInOut = *(dqinout.beginEdit());
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
        dqinout.endEdit();
    }
    else
    {
        vector<InOut>& inOut = *(inout.beginEdit());
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
        inout.endEdit();
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Update_Mapping");
#endif
}



// This method is equivalent to handleTopologyChange() for physical mappings.
template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::checkForChanges()
{
    if (this->mappingHasChanged) this->mappingHasChanged = false;
    ReadAccessor<Data<VecInCoord> > in (*this->fromModel->read(core::ConstVecCoordId::position()));

    // Mapping has to be updated
    if ( (in.size() != targetFrameNumber.getValue()) || // In DOFs have changed
            (gridMaterial && gridMaterial->voxelsHaveChanged.getValue())) // Voxels have changed
    {
        targetFrameNumber.setValue(in.size());
        this->mappingHasChanged = true;
        updateMapping();
    }
}



// This method is only called on mappings handling collision and visual models. Physical ones are not based on a topology (see the method checkForChanges()).
template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::handleTopologyChange(core::topology::Topology* t)
{
    if (t == to_topo)
    {
        // Handle topological changes on the mapped topology
        std::list<const core::topology::TopologyChange *>::const_iterator itBegin=to_topo->beginChange();
        std::list<const core::topology::TopologyChange *>::const_iterator itEnd=to_topo->endChange();

        // Handle topological changes for PointData
        if (!useDQ.getValue()) inout.handleTopologyEvents(itBegin, itEnd);
        //else dqinout.handleTopologyEvents(itBegin, itEnd);
        f_initPos.handleTopologyEvents(itBegin, itEnd);
        f_index.handleTopologyEvents(itBegin, itEnd);
        weight.handleTopologyEvents(itBegin, itEnd);
        weightDeriv.handleTopologyEvents(itBegin, itEnd);
        weightDeriv2.handleTopologyEvents(itBegin, itEnd);

        while ( itBegin != itEnd )
        {
            core::topology::TopologyChangeType changeType = ( *itBegin )->getChangeType();

            switch ( changeType )
            {

            case core::topology::ENDING_EVENT:
            {
                break;
            }
            case core::topology::POINTSADDED:
            {
                //const unsigned int& nbNewVertices = ( static_cast< const typename component::topology::PointsAdded *> ( *itBegin ) )->getNbAddedVertices();
                updateMapping();
                break;
            }
            case core::topology::POINTSREMOVED:
            {
                //const sofa::helper::vector<core::topology::BaseMeshTopology::PointID> &tab = ( static_cast< const typename component::topology::PointsRemoved *> ( *itBegin ) )->getArray();
                //removeSamples( tab);
                break;
            }
            case core::topology::POINTSRENUMBERING:
            default:
                break;
            };

            ++itBegin;
        }
    }
    return;
}



template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::insertFrame (const Vec3d& pos)
{
    core::behavior::MechanicalState< In >* mstatefrom = static_cast<core::behavior::MechanicalState< In >* >( this->fromModel);
    mstatefrom->resize( mstatefrom->getSize()+1);

    WriteAccessor<Data<VecInCoord> > xfrom0 = *this->fromModel->write(core::VecCoordId::restPosition());
    WriteAccessor<Data<VecInCoord> >  xfrom = *this->fromModel->write(core::VecCoordId::position());
    WriteAccessor<Data<VecInCoord> >  xfromReset = *this->fromModel->write(core::VecCoordId::resetPosition());

    //TODO inverseSkinning to obtain restPos from pos

    for ( unsigned int j=0; j<num_spatial_dimensions; j++ ) xfrom0[xfrom0.size()-1][j] = pos[j];
    for ( unsigned int j=0; j<num_spatial_dimensions; j++ ) xfrom[xfrom.size()-1][j] = pos[j];
    for ( unsigned int j=0; j<num_spatial_dimensions; j++ ) xfromReset[xfromReset.size()-1][j] = pos[j];
}


template <class TIn, class TOut>
void FrameBlendingMapping<TIn, TOut>::removeFrame (const unsigned int /*index*/)
{
    serr << "removeFrame call !!" << sendl;
}




} // namespace mapping

} // namespace component

} // namespace sofa

#endif
