/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_IMAGE_ImageToRigidMassEngine_H
#define SOFA_IMAGE_ImageToRigidMassEngine_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/decompose.h>

namespace sofa
{
namespace component
{
namespace engine
{

/**
 * Compute rigid mass from a density image
 */


template <class _ImageTypes>
class ImageToRigidMassEngine : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageToRigidMassEngine,_ImageTypes),Inherited);

    typedef SReal Real;

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    typedef defaulttype::RigidCoord<3,Real> RigidCoord;
    Data< RigidCoord > d_position; ///< position

    /** @name  Outputs */
    //@{
    Data< Real > d_mass; ///< mass
    Data< Coord > d_inertia; ///< axis-aligned inertia tensor

    typedef defaulttype::RigidMass<3,Real> RigidMass;
    typedef typename RigidMass::Mat3x3 Mat3x3;
    Data< RigidMass > d_rigidMass; ///< rigidMass
    //@}

    /** @name  Inputs */
    //@{
    Data< Real > d_density; ///< density (in kg/m^3)
    Data< bool > d_mult; ///< multiply density by image intensity?
    //@}


    virtual std::string getTemplateName() const    override { return templateName(this);    }
    static std::string templateName(const ImageToRigidMassEngine<ImageTypes>* = NULL) { return ImageTypes::Name();    }

    ImageToRigidMassEngine()    :   Inherited()
      , image(initData(&image,ImageTypes(),"image",""))
      , transform(initData(&transform,TransformType(),"transform",""))
      , d_position(initData(&d_position,RigidCoord(),"position","position"))
      , d_mass(initData(&d_mass,(Real)0,"mass","mass"))
      , d_inertia(initData(&d_inertia,Coord(),"inertia","axis-aligned inertia tensor"))
      , d_rigidMass(initData(&d_rigidMass,RigidMass(),"rigidMass","rigidMass"))
      , d_density(initData(&d_density,(Real)1000.,"density","density (in kg/m^3)"))
      , d_mult(initData(&d_mult,false,"multiply","multiply density by image intensity?"))
      , time((unsigned int)0)
    {
        image.setReadOnly(true);
        transform.setReadOnly(true);
        f_listening.setValue(true);
    }

    virtual void init() override
    {
        addInput(&image);
        addInput(&transform);
        addInput(&d_density);
        addInput(&d_mult);
        addOutput(&d_position);
        addOutput(&d_rigidMass);
        addOutput(&d_inertia);
        setDirtyValue();
    }

    virtual void reinit() override { update(); }

protected:

    unsigned int time;

    virtual void update() override
    {
        raTransform inT(this->transform);
        raImage in(this->image);
        if(in->isEmpty()) return;
        const cimg_library::CImg<T>& img = in->getCImg(this->time);

        Real d = d_density.getValue();
        bool mult = d_mult.getValue();


        cleanDirty();

        helper::WriteOnlyAccessor<Data< RigidCoord > > pos(this->d_position);
        helper::WriteOnlyAccessor<Data< RigidMass > > rigidMass(this->d_rigidMass);
        helper::WriteOnlyAccessor<Data< Coord > > inertia(this->d_inertia);

        pos->clear();
        rigidMass->mass=0;
        rigidMass->volume=0;
        rigidMass->inertiaMatrix.clear();

        Real voxelVol = inT->getScale()[0]*inT->getScale()[1]*inT->getScale()[2];
        Mat3x3 C;

        cimg_forXYZ(img,x,y,z)
                if(img(x,y,z)!=(T)0)
        {
            Real density = mult ? (Real)img(x,y,z)*d : d;
            Real m = density*voxelVol;
            rigidMass->volume+=voxelVol;
            rigidMass->mass+=m;
            Coord p = inT->fromImage(Coord(x,y,z));
            pos->getCenter()+=p*m;
            C+=dyad(p,p)*m; // covariance matrix
        }

        if(rigidMass->mass)
        {
            d_mass.setValue(rigidMass->mass);

            pos->getCenter()/=rigidMass->mass;
            C-=dyad(pos->getCenter(),pos->getCenter())*rigidMass->mass; // recenter covariance matrix around mean
            rigidMass->inertiaMatrix = Mat3x3::s_identity*trace(C) - C;   // covariance matrix to inertia matrix

            typename RigidMass::Mat3x3 R;
            helper::Decompose<Real>::eigenDecomposition(rigidMass->inertiaMatrix, R, inertia.wref());

            pos->getOrientation().fromMatrix(R);
            rigidMass->inertiaMatrix.clear();
            for(size_t i=0;i<3;i++) rigidMass->inertiaMatrix[i][i]=inertia.ref()[i]/rigidMass->mass;

            rigidMass->recalc();
        }

    }

    void handleEvent(sofa::core::objectmodel::Event *event) override
    {
        if (simulation::AnimateEndEvent::checkEventType(event))
        {
            raImage in(this->image);
            raTransform inT(this->transform);

            // get current time modulo dimt
            const unsigned int dimt=in->getDimensions()[ImageTypes::DIMENSION_T];
            if(!dimt) return;
            Real t=inT->toImage(this->getContext()->getTime()) ;
            t-=(Real)((int)((int)t/dimt)*dimt);
            t=(t-floor(t)>0.5)?ceil(t):floor(t); // nearest
            if(t<0) t=0.0; else if(t>=(Real)dimt) t=(Real)dimt-1.0; // clamp

            if(this->time!=(unsigned int)t) { this->time=(unsigned int)t; update(); }
        }
    }

};


} // namespace engine
} // namespace component
} // namespace sofa

#endif // SOFA_IMAGE_ImageToRigidMassEngine_H
