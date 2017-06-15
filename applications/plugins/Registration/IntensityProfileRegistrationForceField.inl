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
#ifndef INTENSITYPROFILEREGISTRATIONFORCEFIELD_INL
#define INTENSITYPROFILEREGISTRATIONFORCEFIELD_INL


#include "IntensityProfileRegistrationForceField.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include "float.h"
#include <sofa/helper/IndexOpenMP.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace forcefield
{


using namespace helper;
using namespace cimg_library;


template<class DataTypes, class ImageTypes>
IntensityProfileRegistrationForceField<DataTypes, ImageTypes>::IntensityProfileRegistrationForceField(core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , refImage(initData(&refImage,ImageTypes(),"refImage",""))
    , image(initData(&image,ImageTypes(),"image",""))
    , refTransform(initData(&refTransform,TransformType(),"refTransform",""))
    , transform(initData(&transform,TransformType(),"transform",""))
    , refDirections(initData(&refDirections,VecCoord(),"refDirections","Profile reference directions."))
    , directions(initData(&directions,VecCoord(),"directions","Profile directions."))
    , refProfiles(initData(&refProfiles,ImageTypes(),"refProfiles","reference intensity profiles"))
    , profiles(initData(&profiles,ImageTypes(),"profiles","computed intensity profiles"))
    , similarity(initData(&similarity,similarityTypes(),"similarity","similarity image"))
    , maskOutside(initData(&maskOutside,false,"maskOutside","discard profiles outside images"))
    , useAnisotropicStiffness(initData(&useAnisotropicStiffness,false,"useAnisotropicStiffness", "use more accurate but non constant stiffness matrix."))
    , Sizes(initData(&Sizes,defaulttype::Vec<2,unsigned int>(5,5),"sizes","Inwards/outwards profile size."))
    , Step(initData(&Step,(Real)1E-2,"step","Spacing of the profile discretization."))
    , Interpolation( initData ( &Interpolation,"interpolation","Interpolation method." ) )
    , SimilarityMeasure( initData ( &SimilarityMeasure,"measure","Similarity measure." ) )
    , threshold(initData(&threshold,(Real)1.,"threshold","threshold for the distance minimization."))
    , searchRange(initData(&searchRange,(unsigned int)10,"searchRange","Number of inwards/outwards steps for searching the most similar profiles."))
    , ks(initData(&ks,(Real)100.0,"stiffness","uniform stiffness for the all springs"))
    , kd(initData(&kd,(Real)5.0,"damping","uniform damping for the all springs"))
    , showArrowSize(initData(&showArrowSize,0.01f,"showArrowSize","size of the axis"))
    , drawMode(initData(&drawMode,0,"drawMode","The way springs will be drawn:\n- 0: Line\n- 1:Cylinder\n- 2: Arrow"))
{
    helper::OptionsGroup InterpolationOptions(3,"Nearest", "Linear", "Cubic");
    InterpolationOptions.setSelectedItem(INTERPOLATION_LINEAR);
    Interpolation.setValue(InterpolationOptions);

    helper::OptionsGroup MeasureOptions(2,"Sum of square differences", "Normalized cross correlation");
    MeasureOptions.setSelectedItem(SIMILARITY_NCC);
    SimilarityMeasure.setValue(MeasureOptions);

    refImage.setReadOnly(true);           refImage.setGroup("Inputs");
    image.setReadOnly(true);              image.setGroup("Inputs");
    refTransform.setReadOnly(true);       refTransform.setGroup("Inputs");
    transform.setReadOnly(true);          transform.setGroup("Inputs");
    refDirections.setReadOnly(true);      refDirections.setGroup("Inputs");
    directions.setReadOnly(true);         directions.setGroup("Inputs");

    refProfiles.setReadOnly(true);       refProfiles.setGroup("Outputs");
    profiles.setReadOnly(true);          profiles.setGroup("Outputs");
    similarity.setReadOnly(true);        similarity.setGroup("Outputs");
}


template<class DataTypes,class ImageTypes>
void IntensityProfileRegistrationForceField<DataTypes,ImageTypes>::reinit()
{
    this->udpateProfiles(true);
}

template<class DataTypes,class ImageTypes>
void IntensityProfileRegistrationForceField<DataTypes,ImageTypes>::init()
{
    this->Inherit::init();
    core::objectmodel::BaseContext* context = this->getContext();

    if(!(this->mstate)) this->mstate = dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes> *>(context->getMechanicalState());

    if(this->mstate)
    {
        const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
        //RDataRefVecCoord x(*this->getMState()->read(core::ConstVecCoordId::position()));

        RDataRefVecCoord dir(this->directions);
        if(dir.size() != x.size()) serr<<"IntensityProfileRegistrationForceField: invalid 'directions' size "<< endl;

        VecCoord& refDir = *this->refDirections.beginEdit();
        if(refDir.size() != x.size())
            if(dir.size() == x.size())
                refDir.assign(dir.begin(),dir.end());
        this->refDirections.endEdit();
    }

    // check inputs
    raImage im(this->image);    if( im->isEmpty() ) serr<<"IntensityProfileRegistrationForceField: Target data not found"<< endl;
    raImage rim(this->refImage),rp(this->refProfiles);    if( rim->isEmpty() && rp->isEmpty() ) serr<<"IntensityProfileRegistrationForceField: Reference data not found"<< endl;

    this->udpateProfiles(true);
}



/// compute intensity profile image by sampling the input image along 'direction'.
/// can be done for the current or reference position/image
/// Inward and outward profile sizes are defined by data 'Sizes' (+ searchRange, if done on current position)

template<class DataTypes,class ImageTypes>
void IntensityProfileRegistrationForceField<DataTypes,ImageTypes>::udpateProfiles(bool ref)
{
    raImage in(ref?this->refImage:this->image);
    if( in->isEmpty() || !this->mstate)  return;
    raTransform inT(ref?this->refTransform:this->transform);

    // get current time modulo dimt
    const unsigned int dimt=in->getDimensions()[4];
    Real t=inT->toImage(this->getContext()->getTime()) ;
    t-=(Real)((int)((int)t/dimt)*dimt);
    t=(t-floor(t)>0.5)?ceil(t):floor(t); // nearest
    if(t<0) t=0.0; else if(t>=(Real)dimt) t=(Real)dimt-1.0; // clamp

    // get current data
    const CImg<T>& img = in->getCImg(t);
    const VecCoord& pos = (ref?
                               this->mstate->read(core::ConstVecCoordId::restPosition())->getValue():
                               this->mstate->read(core::ConstVecCoordId::position())->getValue());
    const RDataRefVecCoord dir(ref?this->refDirections:this->directions);
    if(dir.size() != pos.size()) return;

    // allocate
    unsigned int nbpoints=pos.size();
    defaulttype::Vec<2,unsigned int> sizes(this->Sizes.getValue()[0]+ (ref?0:searchRange.getValue()) ,this->Sizes.getValue()[1]+ (ref?0:searchRange.getValue()));
    defaulttype::Vec<4,unsigned int> dims(sizes[0]+sizes[1]+1,nbpoints,1,img.spectrum());

    waImage out(ref?this->refProfiles:this->profiles);
    if( out->isEmpty() )  out->getCImgList().push_back(CImg<T>());
    CImg<T> &prof = out->getCImg(0);

    if(dims[0]!= (unsigned int)prof.width() || dims[1]!=(unsigned int)prof.height() || dims[2]!=(unsigned int)prof.depth()  || dims[3]!=(unsigned int)prof.spectrum())
        prof.assign(dims[0],dims[1],dims[2],dims[3]);

    CImg<bool> &msk= ref?this->refMask:this->mask;
    if(dims[0]!= (unsigned int)msk.width() || dims[1]!=(unsigned int)msk.height() || dims[2]!=(unsigned int)msk.depth()  || dims[3]!=(unsigned int)msk.spectrum())
        msk.assign(dims[0],dims[1],dims[2],dims[3]);
    msk.fill(0);

    // sampling
    static const T OutValue = (T)0; // value assigned to profiles when outside image

    if(Interpolation.getValue().getSelectedId()==INTERPOLATION_NEAREST)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(sofa::helper::IndexOpenMP<unsigned int>::type i=0;i<dims[1];i++)
        {
            Coord dp=dir[i]*this->Step.getValue();
            Coord p=pos[i]-dp*(Real)sizes[0];
            for(unsigned int j=0;j<dims[0];j++)
            {
                Coord Tp = inT->toImage(p);
                if(!in->isInside(Tp[0],Tp[1],Tp[2])) for(unsigned int k=0;k<dims[3];k++) { prof(j,i,0,k) = OutValue; msk(j,i,0,k) = 1; }
                else for(unsigned int k=0;k<dims[3];k++) prof(j,i,0,k) = img.atXYZ(sofa::helper::round((double)Tp[0]),sofa::helper::round((double)Tp[1]),sofa::helper::round((double)Tp[2]),k);
                p+=dp;
            }
        }
    }
    else if(Interpolation.getValue().getSelectedId()==INTERPOLATION_LINEAR)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(sofa::helper::IndexOpenMP<unsigned int>::type i=0;i<dims[1];i++)
        {
            Coord dp=dir[i]*this->Step.getValue();
            Coord p=pos[i]-dp*(Real)sizes[0];
            for(unsigned int j=0;j<dims[0];j++)
            {
                Coord Tp = inT->toImage(p);
                if(!in->isInside(Tp[0],Tp[1],Tp[2])) for(unsigned int k=0;k<dims[3];k++) { prof(j,i,0,k) = OutValue; msk(j,i,0,k) = 1; }
                else for(unsigned int k=0;k<dims[3];k++) prof(j,i,0,k) = (T)img.linear_atXYZ(Tp[0],Tp[1],Tp[2],k,OutValue);
                p+=dp;
            }
        }
    }
    else    // INTERPOLATION_CUBIC
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(sofa::helper::IndexOpenMP<unsigned int>::type i=0;i<dims[1];i++)
        {
            Coord dp=dir[i]*this->Step.getValue();
            Coord p=pos[i]-dp*(Real)sizes[0];
            for(unsigned int j=0;j<dims[0];j++)
            {
                Coord Tp = inT->toImage(p);
                if(!in->isInside(Tp[0],Tp[1],Tp[2])) for(unsigned int k=0;k<dims[3];k++) { prof(j,i,0,k) = OutValue; msk(j,i,0,k) = 1; }
                else for(unsigned int k=0;k<dims[3];k++) prof(j,i,0,k) = (T)img.cubic_atXYZ(Tp[0],Tp[1],Tp[2],k,OutValue,cimg::type<T>::min(),cimg::type<T>::max());
                p+=dp;
            }
        }
    }
}

/// compute silarity image by convoluing current and reference profiles
/// the width of the resulting image is 2*searchRange

template<class DataTypes,class ImageTypes>
void IntensityProfileRegistrationForceField<DataTypes,ImageTypes>::udpateSimilarity()
{
    raImage IPref(this->refProfiles);
    raImage IP(this->profiles);

    if( IP->isEmpty() || IPref->isEmpty() || !this->mstate)  return;

    const VecCoord& pos = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const CImg<T>& prof = IP->getCImg(0);
    const CImg<T>& profref = IPref->getCImg(0);

    // allocate
    unsigned int nbpoints=pos.size();
    defaulttype::Vec<4,unsigned int> dims(2*searchRange.getValue()+1,nbpoints,1,1);

    waSimilarity out(this->similarity);
    if( out->isEmpty() )  out->getCImgList().push_back(CImg<Ts>());
    CImg<Ts> &simi = out->getCImg(0);

    if(dims[0]!= (unsigned int)simi.width() || dims[1]!=(unsigned int)simi.height() || dims[2]!=(unsigned int)simi.depth()  || dims[3]!=(unsigned int)simi.spectrum())
        simi.assign(dims[0],dims[1],dims[2],dims[3]);

    if(dims[0]!= (unsigned int)similarityMask.width() || dims[1]!=(unsigned int)similarityMask.height() || dims[2]!=(unsigned int)similarityMask.depth()  || dims[3]!=(unsigned int)similarityMask.spectrum())
        similarityMask.assign(dims[0],dims[1],dims[2],dims[3]);
    similarityMask.fill(0);

    // check lengths
    unsigned int ipdepth=this->Sizes.getValue()[0]+this->Sizes.getValue()[1]+1;
    unsigned int nbchannels =  prof.spectrum();
    if(prof.height() != (int)nbpoints || prof.width()!=(int)(ipdepth+2*searchRange.getValue())) { serr<<"IntensityProfileRegistrationForceField: invalid profile size"<<sendl; return; }
    if(/*profref.height() != (int)nbpoints ||*/ profref.width()!=(int)ipdepth) { serr<<"IntensityProfileRegistrationForceField: invalid profile ref length"<<sendl; return; }
    if(profref.spectrum() != (int)nbchannels) { serr<<"IntensityProfileRegistrationForceField: invalid profile ref channel number"<<sendl; return; }

    // convolve
    simi.fill((Ts)0);


    typedef sofa::helper::IndexOpenMP<unsigned int>::type IndexOpenMPType;

    if(this->SimilarityMeasure.getValue().getSelectedId()==SIMILARITY_SSD)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexOpenMPType i=0;i<dims[1];i++)
        {
            unsigned int iref=i<(IndexOpenMPType)profref.height()?i:(IndexOpenMPType)profref.height()-1;
            for(unsigned int j=0;j<dims[0];j++)
            {
                Ts& s = simi(j,i);

                for(unsigned int k=0;k<nbchannels;k++)
                    for(unsigned int x=0;x<ipdepth;x++)
                        if(!similarityMask(j,i))
                        {
                            if(maskOutside.getValue() && (refMask (x,i,0,k) || mask (x+j,i,0,k))) similarityMask(j,i)=1;
                            else
                            {
                                T vref= profref (x,iref,0,k) , v = prof (x+j,i,0,k);
                                s+=((Ts)v-(Ts)vref)*((Ts)v-(Ts)vref);
                            }
                        }
            }
        }
    }
    else // SIMILARITY_NCC
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(IndexOpenMPType i=0;i<dims[1];i++)
        {
            unsigned int iref=i<(IndexOpenMPType)profref.height()?i:(IndexOpenMPType)profref.height()-1;
            for(unsigned int j=0;j<dims[0];j++)
            {
                Ts& s = simi(j,i);
                Ts refmean=(Ts)0 , mean=(Ts)0 , norm=(Ts)0 , refnorm=(Ts)0;

                for(unsigned int k=0;k<nbchannels;k++)
                    for(unsigned int x=0;x<ipdepth;x++)
                        if(!similarityMask(j,i))
                        {
                            if(maskOutside.getValue() && (refMask (x,i,0,k) || mask (x+j,i,0,k)))  similarityMask(j,i)=1;
                            else
                            {
                                T vref= profref (x,iref,0,k) , v = prof (x+j,i,0,k);
                                refmean+=(Ts)vref; mean+=(Ts)v;
                            }
                        }
                if(!similarityMask(j,i))
                {
                    refmean/=(Ts)(ipdepth*nbchannels); mean/=(Ts)(ipdepth*nbchannels);

                    for(unsigned int k=0;k<nbchannels;k++)
                        for(unsigned int x=0;x<ipdepth;x++)
                        {
                            Ts vref= (Ts)profref (x,iref,0,k) , v = (Ts)prof (x+j,i,0,k);
                            s       -= (vref-refmean)*(v-mean);
                            refnorm += (vref-refmean)*(vref-refmean);
                            norm    += (v-mean)*(v-mean);
                        }
                    if(refnorm && norm) s/=sqrt(refnorm*norm);
                    else similarityMask(j,i)=1;
                }
            }
        }
    }


    // assign outside values to max similarity and normalize btw 0 and 1
    Ts vmax=cimg::type<Ts>::min();
    cimg_foroff(simi,off) if(!similarityMask[off]) if(vmax<simi[off]) vmax=simi[off];
    cimg_foroff(simi,off)  if(similarityMask[off]) simi[off]=vmax;
    simi.normalize(0,1);
}




template<class DataTypes,class ImageTypes>
void IntensityProfileRegistrationForceField<DataTypes,ImageTypes>::addForce(const core::MechanicalParams* /*mparams*/,DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v )
{
    VecDeriv&        f = *_f.beginEdit();           //WDataRefVecDeriv f(_f);
    const VecCoord&  x = _x.getValue();			//RDataRefVecCoord x(_x);
    const VecDeriv&  v = _v.getValue();			//RDataRefVecDeriv v(_v);
    RDataRefVecCoord dirs ( directions );

    unsigned int nb = x.size();

    if(this->useAnisotropicStiffness.getValue())    this->dfdx.resize(nb);
    this->targetPos.resize(nb);
    Real k = this->ks.getValue(),kd=this->kd.getValue();

    udpateProfiles();
    udpateSimilarity();

    m_potentialEnergy = 0;
    //serr<<"addForce()"<<sendl;
    for (unsigned int i=0; i<nb; i++)
    {

        raSimilarity out(this->similarity);
        if( out->isEmpty())  return;
        const CImg<Ts> &simi = out->getCImg(0);

        int L=(int)searchRange.getValue();

        const Ts* ptr=simi.data(0,i);
        Ts minimum=(Ts)threshold.getValue();
        int lmin=0;
        for (int l=-L; l<=L; l++)
        {
            if(*ptr<minimum)
            {
                minimum=*ptr;
                lmin=l;
            }
            ptr++;
        }
        targetPos[i]=x[i]+dirs[i]*(Real)lmin*Step.getValue();

        // add rest spring force
        //serr<<"addForce() between "<<i<<" and "<<closestPos[i]<<sendl;
        Coord u = this->targetPos[i]-x[i];
        Real nrm2 = u.norm2();
        f[i]+=k*u;
        if(this->useAnisotropicStiffness.getValue())
        {
            if(nrm2) this->dfdx[i] = defaulttype::dyad(u,u)/u.norm2();
            else this->dfdx[i].identity(); // use to stabilize points with no force
        }
        m_potentialEnergy += nrm2 * k * 0.5;
        if(kd && nrm2) f[i]-=kd*u*dot(u,v[i])/u.norm2();
    }
    _f.endEdit();
}


template<class DataTypes,class ImageTypes>
void IntensityProfileRegistrationForceField<DataTypes,ImageTypes>::addDForce(const core::MechanicalParams* mparams,DataVecDeriv& _df , const DataVecDeriv&  _dx )
{
    sofa::helper::WriteAccessor< DataVecDeriv > df = _df;
    sofa::helper::ReadAccessor< DataVecDeriv > dx = _dx;
    Real k = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue())  * this->ks.getValue() ;
    if(!k) return;

    if(this->useAnisotropicStiffness.getValue())
        for (unsigned int i=0; i<dx.size(); i++)        df[i] -= this->dfdx[i]*dx[i]  * k;
    else
        for (unsigned int i=0; i<dx.size(); i++)        df[i] -= dx[i] * k;
}



template<class DataTypes,class ImageTypes>
void IntensityProfileRegistrationForceField<DataTypes,ImageTypes>::addKToMatrix(const core::MechanicalParams* mparams,const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    Real k = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()) * this->ks.getValue();
    if(!k) return;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix *mat = mref.matrix;
    const int offset = (int)mref.offset;
    const int N = Coord::total_size;
    const int nb = this->targetPos.size();

    if(this->useAnisotropicStiffness.getValue())
    {
        for (int index = 0; index <nb; index++)
            for(int i = 0; i < N; i++)
                for(int j = 0; j < N; j++)
                    mat->add(offset + N * index + i, offset + N * index + j, -k*this->dfdx[index][i][j]);
    }
    else
    {
        for (int index = 0; index <nb; index++)
            for(int i = 0; i < N; i++)
                mat->add(offset + N * index + i, offset + N * index + i, -k);
    }

}

template<class DataTypes,class ImageTypes>
void IntensityProfileRegistrationForceField<DataTypes,ImageTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(ks.getValue()==0) return;

    if (!vparams->displayFlags().getShowForceFields()) return;

    RDataRefVecCoord x(*this->getMState()->read(core::ConstVecCoordId::position()));
    //const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    unsigned int nb = this->targetPos.size();
    if (vparams->displayFlags().getShowForceFields())
    {
        std::vector< defaulttype::Vector3 > points;
        for (unsigned int i=0; i<nb; i++)
        {
            defaulttype::Vector3 point1 = DataTypes::getCPos(x[i]);
            defaulttype::Vector3 point2 = DataTypes::getCPos(this->targetPos[i]);
            points.push_back(point1);
            points.push_back(point2);
        }

        const defaulttype::Vec<4,float> c(0,1,0.5,1);
        if (showArrowSize.getValue()==0 || drawMode.getValue() == 0)	vparams->drawTool()->drawLines(points, 1, c);
        else if (drawMode.getValue() == 1)	for (unsigned int i=0;i<points.size()/2;++i) vparams->drawTool()->drawCylinder(points[2*i+1], points[2*i], showArrowSize.getValue(), c);
        else if (drawMode.getValue() == 2)	for (unsigned int i=0;i<points.size()/2;++i) vparams->drawTool()->drawArrow(points[2*i+1], points[2*i], showArrowSize.getValue(), c);
        else serr << "No proper drawing mode found!" << sendl;
    }
}


}
}
} // namespace sofa

#endif
