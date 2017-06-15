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
#ifndef SOFA_GaussPointContainer_H
#define SOFA_GaussPointContainer_H

#include <Flexible/config.h>
#include "../quadrature/BaseGaussPointSampler.h"


namespace sofa
{
namespace component
{
namespace engine
{

/**
 * This class is empty. It is just used to contain custom Gauss points and provide interface with mappings
 */


class SOFA_Flexible_API GaussPointContainer : public BaseGaussPointSampler
{
public:
    typedef BaseGaussPointSampler Inherited;
    SOFA_CLASS(GaussPointContainer,Inherited);

    /** @name  GaussPointSampler types */
    //@{
    typedef Inherited::Real Real;
    typedef Inherited::waVolume waVolume;
    //@}

    Data< unsigned int > f_volumeDim;
    Data< helper::vector<Real> > f_inputVolume;

    virtual void init()
    {
        Inherited::init();
        addInput(&f_position);
        addInput(&f_inputVolume);
        addInput(&f_volumeDim);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:
    GaussPointContainer()    :   Inherited()
      , f_volumeDim(initData(&f_volumeDim,(unsigned int)1,"volumeDim","dimension of quadrature weight vectors"))
      , f_inputVolume(initData(&f_inputVolume,"inputVolume","weighted volumes (=quadrature weights)"))
    {

    }

    virtual ~GaussPointContainer()
    {
    }

    virtual void update()
    {
        this->updateAllInputsIfDirty();
        cleanDirty();

        helper::ReadAccessor< Data< helper::vector<Real> > > invol(f_inputVolume);
        if(!invol.size()) serr<<"no volume provided -> use unit default volume"<<sendl;
        waVolume vol(this->f_volume);
        unsigned int dim = this->f_volumeDim.getValue();

        helper::WriteOnlyAccessor<Data< VTransform > > transforms(this->f_transforms);

        vol.resize(this->f_position.getValue().size());
        transforms.resize(this->f_position.getValue().size());
        for(unsigned int i=0;i<vol.size();i++)
        {
            vol[i].resize(dim);
            for(unsigned int j=0;j<dim;j++)
            {
                if(!invol.size()) vol[i][j]=(j==0)?1.:0.;
                else if(invol.size()==dim) vol[i][j] = invol[j]; // the same quadrature weights are repeated
                else vol[i][j] = invol[i*dim + j];
            }
            transforms[i].identity();
        }
    }

};

}
}
}

#endif
