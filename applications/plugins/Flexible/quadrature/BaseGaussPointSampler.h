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
#ifndef SOFA_BaseGaussPointSAMPLER_H
#define SOFA_BaseGaussPointSAMPLER_H

#include <Flexible/config.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/OptionsGroup.h>

#define GAUSSLEGENDRE 0
#define NEWTONCOTES 1
#define ELASTON 2


namespace sofa
{
namespace component
{
namespace engine
{

///Abstract class for sampling integration points according to a specific quadrature method.
/**
 * Samplers provide:
 * - Sample positions according to integration method of order 'order'
 * - Initial value of deformation gradient according to underlying geometry (eg. surface mesh) and/or user input (to set anisotopy and residual deformation)
 * - Weighted volume associated to each sample (for elastons, this is a vector of volume moments)
 * - sample region (can be used in shape function component to provide an averaged version of the shape function)
 */


class SOFA_Flexible_API BaseGaussPointSampler : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_ABSTRACT_CLASS(BaseGaussPointSampler,Inherited);

    typedef SReal Real;

    /** @name quadrature method */
    //@{
    Data<helper::OptionsGroup> f_method;    ///< Quadrature method <br>
    /**< \arg \c 0 Gauss-Legendre http://en.wikipedia.org/wiki/Gaussian_quadrature
                                                 \arg \c 1 Newton-Cotes http://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas
                                                 \arg \c 2 Elaston [Martin-2010] http://graphics.ethz.ch/publications/papers/paperMar10.php
                                            */
    //@}

    /** @name position data */
    //@{
    static const unsigned int spatial_dimensions = 3;
    typedef defaulttype::Vec<spatial_dimensions,Real> Coord;   // TODO: put as a template to handle the 2D case (currently the conversion 3D-> 2D is done in the deformation mapping) ?
    typedef helper::vector<Coord> SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    typedef helper::WriteOnlyAccessor<Data< SeqPositions > > waPositions;
    Data< SeqPositions > f_position; ///< Samples position
    //@}

    /** @name orientation data */
    //@{
    typedef defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> Transform;
    typedef helper::vector<Transform> VTransform;
    Data< VTransform > f_transforms;        ///< linear transformation in world space to orient samples
    //@}


    /** @name volume integral data */
    //@{
    Data< unsigned int > f_order; ///< Order of quadrature method
    typedef helper::vector<Real> volumeIntegralType;
    Data< helper::vector<volumeIntegralType> > f_volume; ///< Weighted volumes associated to samples
    typedef helper::WriteOnlyAccessor< Data< helper::vector<volumeIntegralType> > > waVolume;
    //@}

    /** @name visu data */
    //@{
    Data< float > showSamplesScale; ///< Samples scale
    Data< int > drawMode; ///< Drawing mode: 0. Green points; 1. Green spheres
    Data< float > showIndicesScale; ///< Indices samples scale
    //@}

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const BaseGaussPointSampler* = NULL) { return std::string();    }

    BaseGaussPointSampler()    :   Inherited()
      , f_method ( initData ( &f_method,"method","quadrature method" ) )
      , f_position(initData(&f_position,SeqPositions(),"position","output sample positions"))
      , f_transforms(initData(&f_transforms,VTransform(),"transforms","output sample orientations"))
      , f_order(initData(&f_order,(unsigned int)1,"order","order of quadrature method"))
      , f_volume(initData(&f_volume,helper::vector<volumeIntegralType>(),"volume","output weighted volume"))
      , showSamplesScale(initData(&showSamplesScale,0.0f,"showSamplesScale","show samples scale"))
      , drawMode(initData(&drawMode,0,"drawMode","0: Green points; 1: Green spheres"))
      , showIndicesScale(initData(&showIndicesScale,0.0f,"showIndicesScale", "show indices scale"))
    {
        helper::OptionsGroup methodOptions(3,"0 - Gauss-Legendre"
                                           ,"1 - Newton-Cotes"
                                           ,"2 - Elastons"
                                           );
        methodOptions.setSelectedItem(GAUSSLEGENDRE);
        f_method.setValue(methodOptions);
    }

    virtual void init()
    {
        addInput(&f_method);
        addInput(&f_order);
//        addInput(&f_position); // can be either input or output depending on the sampler
        addOutput(&f_volume);
        addOutput(&f_transforms);
    }

    ///@brief Get the number of samples
    unsigned int getNbSamples() {return this->f_position.getValue().size(); }
    ///@brief Get all samples as a const reference
    const SeqPositions& getSamples(){return this->f_position.getValue();}
    ///@brief Get all orientations as a const reference
    const VTransform& getTransforms(){return this->f_transforms.getValue();}

protected:

    virtual void draw(const core::visual::VisualParams* vparams)
    {
        if (!vparams->displayFlags().getShowVisualModels()) return;

        vparams->drawTool()->saveLastState();

        if (showSamplesScale.getValue()>0) {
            switch( drawMode.getValue() ) {
            case 1:
                vparams->drawTool()->setLightingEnabled(true);
                vparams->drawTool()->drawSpheres(this->f_position.getValue(),showSamplesScale.getValue(),defaulttype::Vec<4,float>(0.1f, 0.7f, 0.1f, 1.0f));
                break;
            default:
                vparams->drawTool()->drawPoints(this->f_position.getValue(),showSamplesScale.getValue(),defaulttype::Vec<4,float>(0.2f, 1.0f, 0.2f, 1.0f));
                break;
            }
        }
        if (showIndicesScale.getValue()>0) {
            vparams->drawTool()->draw3DText_Indices(this->f_position.getValue(), showIndicesScale.getValue(), defaulttype::Vec<4,float>(0.1f, 0.7f, 0.1f, 1.0f));
        }

        vparams->drawTool()->restoreLastState();
    }


};

}
}
}

#endif
