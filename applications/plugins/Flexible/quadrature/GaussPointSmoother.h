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
#ifndef SOFA_GaussPointSmoother_H
#define SOFA_GaussPointSmoother_H

#include <Flexible/config.h>
#include "../quadrature/BaseGaussPointSampler.h"
#include "../shapeFunction/BaseShapeFunction.h"
#include <sofa/helper/SVector.h>


namespace sofa
{
namespace component
{
namespace engine
{

/**
 * Smooth gauss points from another sampler
 */


class SOFA_Flexible_API GaussPointSmoother : public BaseGaussPointSampler
{
public:
    typedef BaseGaussPointSampler Inherited;
    SOFA_CLASS(GaussPointSmoother,Inherited);

    /** @name  GaussPointSampler types */
    //@{
    typedef Inherited::Real Real;
    typedef Inherited::waVolume waVolume;
    enum { spatial_dimensions = Inherited::spatial_dimensions };
    //@}

    /** @name  Shape Function types    */
    //@{
    typedef core::behavior::ShapeFunctionTypes<spatial_dimensions,Real> ShapeFunctionType;
    typedef core::behavior::BaseShapeFunction<ShapeFunctionType> BaseShapeFunction;
    typedef BaseShapeFunction::VReal VReal;
    typedef BaseShapeFunction::VecVReal VecVReal;
    typedef BaseShapeFunction::VRef VRef;
    typedef BaseShapeFunction::VecVRef VecVRef;
    //@}

    Data< VTransform > d_inputTransforms;   ///< parent transforms from another GP sampler
    Data< helper::vector<volumeIntegralType> > d_inputVolume;  ///< parent volumes from another GP sampler

    Data<VecVRef > d_index;      ///< computed child to parent relationship using local shape function. index[i][j] is the index of the j-th parent influencing child i.
    Data<VecVReal > d_w;      ///< Influence weights

    virtual void init()
    {
        Inherited::init();
        addInput(&f_position);
        addInput(&d_inputTransforms);
        addInput(&d_inputVolume);
        addOutput(&d_index);
        addOutput(&d_w);
        setDirtyValue();
    }

    virtual void reinit() { update(); }


protected:
    GaussPointSmoother()    :   Inherited()
      , d_inputTransforms(initData(&d_inputTransforms,VTransform(),"inputTransforms","sample orientations"))
      , d_inputVolume(initData(&d_inputVolume,helper::vector<volumeIntegralType>(),"inputVolume","sample volume"))
      , d_index ( initData ( &d_index,"indices","parent indices for each child" ) )
      , d_w ( initData ( &d_w,"weights","influence weights" ) )
    {

    }

    virtual ~GaussPointSmoother()
    {
    }


    virtual void update()
    {
        this->updateAllInputsIfDirty();
        cleanDirty();

        BaseShapeFunction* _shapeFunction=NULL;
        this->getContext()->get(_shapeFunction,core::objectmodel::BaseContext::SearchUp);
        if( !_shapeFunction ) { serr<<"Shape function not found"<< sendl; return;}

        //        engine::BaseGaussPointSampler* sampler=NULL;
        //        this->getContext()->get(sampler,core::objectmodel::BaseContext::SearchUp);
        //        if( !sampler ) { serr<<"Gauss point sampler not found"<< sendl; }
        //        helper::ReadAccessor<Data< VTransform > > inputTransforms(sampler->f_transforms);
        //        helper::ReadAccessor< Data< helper::vector<volumeIntegralType> > > inputVolumes(sampler->f_volume);

        helper::ReadAccessor<Data< VTransform > > inputTransforms(this->d_inputTransforms);
        helper::ReadAccessor< Data< helper::vector<volumeIntegralType> > > inputVolumes(this->d_inputVolume);
        const unsigned int volumeDim = inputVolumes[0].size();

        raPositions positions(this->f_position);
        const unsigned int childSize = positions.size();

        helper::WriteOnlyAccessor<Data< VecVRef > > indices(this->d_index);
        helper::WriteOnlyAccessor<Data< VecVReal > > weights(this->d_w);

        waVolume volumes(this->f_volume);
        helper::WriteOnlyAccessor<Data< VTransform > > transforms(this->f_transforms);

        // interpolate weights at sample positions
        indices.resize(childSize);
        weights.resize(childSize);
        for( size_t i=0 ; i<childSize ; ++i)
        {
            _shapeFunction->computeShapeFunction(positions[i],indices[i],weights[i]);
        }

        // generate transforms as weighted sums
        transforms.resize(childSize);
        for( size_t i=0 ; i<childSize ; ++i)
        {
            transforms[i] = Transform();
            for( size_t j=0 ; j<indices[i].size() ; ++j) transforms[i]+=inputTransforms[indices[i][j]]*weights[i][j];
        }

        // partition volumes using parent to child normalized weights
        volumes.resize(childSize);
        std::vector<Real> W(inputVolumes.size(),0.);
        for(size_t i=0; i<childSize; ++i)
            for(size_t j=0; j< indices[i].size(); j++ )
                W[indices[i][j]]+=weights[i][j];

        for(size_t i=0; i<childSize; ++i)
        {
            volumes[i].resize(volumeDim);
            for( size_t k=0 ; k<volumeDim ; ++k) volumes[i][k]=0;
            for(size_t j=0; j< indices[i].size(); j++ )
            {
                Real w = weights[i][j]/W[indices[i][j]];
                for( size_t k=0 ; k<volumeDim ; ++k) volumes[i][k]+=inputVolumes[indices[i][j]][k]*w;
            }
        }
    }

};

}
}
}

#endif
