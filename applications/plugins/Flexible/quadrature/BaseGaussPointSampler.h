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
#ifndef SOFA_BaseGaussPointSAMPLER_H
#define SOFA_BaseGaussPointSAMPLER_H

#include "../initFlexible.h"
#include <sofa/core/DataEngine.h>
#include <sofa/component/component.h>
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

using helper::vector;
using defaulttype::Vec;

/**
 * Abstract class for sampling integration points according to a specific quadrature method
 * Samplers provide:
 * - Sample positions according to integration method of order @param order
 * - Weighted volume associated to each sample (for elastons, this is a vector of volume moments)
 */


class BaseGaussPointSampler : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_ABSTRACT_CLASS(BaseGaussPointSampler,Inherited);

    typedef SReal Real;

    /** @name quadrature method */
    //@{
    Data<helper::OptionsGroup> f_method;
    //@}

    /** @name position data */
    //@{
    typedef Vec<3,Real> Coord;
    typedef vector<Coord> SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    typedef helper::WriteAccessor<Data< SeqPositions > > waPositions;
    Data< SeqPositions > f_position;
    //@}

    /** @name volume integral data */
    //@{
    Data< unsigned int > f_order;
    typedef vector<Real> volumeIntegralType;
    Data< vector<volumeIntegralType> > f_volume;
    typedef helper::WriteAccessor< Data< vector<volumeIntegralType> > > waVolume;
    //@}

    /** @name visu data */
    //@{
    Data< bool > showSamples;
    //@}

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const BaseGaussPointSampler* = NULL) { return std::string();    }

    BaseGaussPointSampler()    :   Inherited()
        , f_method ( initData ( &f_method,"method","quadrature method" ) )
        , f_position(initData(&f_position,SeqPositions(),"position","output sample positions"))
        , f_order(initData(&f_order,(unsigned int)1,"order","order of quadrature method"))
        , f_volume(initData(&f_volume,vector<volumeIntegralType>(),"volume","output weighted volume"))
        , showSamples(initData(&showSamples,false,"showSamples","show samples"))
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
        addOutput(&f_position);
        addOutput(&f_volume);
        setDirtyValue();
    }

    /** functions to initialize mechanical objects (e.g. deformation gradients) */
    unsigned int getNbSamples() {return this->f_position.getValue().size(); }
    const Coord& getSample(unsigned int index) {return this->f_position.getValue()[index]; }

protected:

    virtual void draw(const core::visual::VisualParams* vparams)
    {
        if (!vparams->displayFlags().getShowVisualModels()) return;
        raPositions pos(this->f_position);
        if (this->showSamples.getValue()) vparams->drawTool()->drawPoints(this->f_position.getValue(),5.0,Vec<4,float>(0.2,1,0.2,1));
    }



};

}
}
}

#endif
