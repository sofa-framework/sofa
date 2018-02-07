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
#define FLEXIBLE_DeformationGradientTYPES_CPP

#include <Flexible/config.h>
#include "../types/DeformationGradientTypes.h"
#include <sofa/core/ObjectFactory.h>

#include <SofaBaseMechanics/MechanicalObject.inl>
#include <sofa/core/State.inl>

namespace sofa
{

using namespace sofa::defaulttype;


namespace core
{

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API State<F331dTypes>;
template class SOFA_Flexible_API State<F321dTypes>;
template class SOFA_Flexible_API State<F311dTypes>;
template class SOFA_Flexible_API State<F332dTypes>;
template class SOFA_Flexible_API State<F221dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API State<F331fTypes>;
template class SOFA_Flexible_API State<F321fTypes>;
template class SOFA_Flexible_API State<F311fTypes>;
template class SOFA_Flexible_API State<F332fTypes>;
template class SOFA_Flexible_API State<F221fTypes>;
#endif

} // namespace core


namespace component
{
namespace container
{

// ==========================================================================
// Init Specializations (initialization from GaussPointSampler)
/*
template <>
void MechanicalObject<F331Types>::init()
{
    engine::BaseGaussPointSampler* sampler=NULL;
    this->getContext()->get(sampler,core::objectmodel::BaseContext::Local);
    if(sampler)
    {
        unsigned int nbp=sampler->getNbSamples();
        this->resize(nbp);

        Data<VecCoord>* x_wAData = this->write(VecCoordId::position());
        VecCoord& x_wA = *x_wAData->beginWriteOnly();
        for(unsigned int i=0;i<nbp;i++) DataTypes::set(x_wA[i], sampler->getSample(i)[0], sampler->getSample(i)[1], sampler->getSample(i)[2]);

        VecCoord *x0_edit = x0.beginEdit();
        x0.setValue(x.getValue());
        if (restScale.getValue() != (Real)1) { Real s = (Real)restScale.getValue(); for (unsigned int i=0; i<x0_edit->size(); i++) (*x0_edit)[i] *= s;        }
        x0.endEdit();

        if(this->f_printLog.getValue())  std::cout<<this->getName()<<" : "<< nbp <<" gauss points imported"<<std::endl;
        reinit();
    }
}

template <>
void MechanicalObject<F332Types>::init()
{
    engine::BaseGaussPointSampler* sampler=NULL;
    this->getContext()->get(sampler,core::objectmodel::BaseContext::Local);
    if(sampler)
    {
        unsigned int nbp=sampler->getNbSamples();
        this->resize(nbp);

        Data<VecCoord>* x_wAData = this->write(VecCoordId::position());
        VecCoord& x_wA = *x_wAData->beginWriteOnly();
        for(unsigned int i=0;i<nbp;i++) DataTypes::set(x_wA[i], sampler->getSample(i)[0], sampler->getSample(i)[1], sampler->getSample(i)[2]);

        VecCoord *x0_edit = x0.beginEdit();
        x0.setValue(x.getValue());
        if (restScale.getValue() != (Real)1) { Real s = (Real)restScale.getValue(); for (unsigned int i=0; i<x0_edit->size(); i++) (*x0_edit)[i] *= s;        }
        x0.endEdit();

        if(this->f_printLog.getValue())  std::cout<<this->getName()<<" : "<< nbp <<" gauss points imported"<<std::endl;
        reinit();
    }
}
*/

// ==========================================================================
// Instanciation

SOFA_DECL_CLASS ( DefGradientMechanicalObject )


int DefGradientMechanicalObjectClass = core::RegisterObject ( "mechanical state vectors" )
#ifndef SOFA_FLOAT
        .add< MechanicalObject<F331dTypes> >()
        .add< MechanicalObject<F321dTypes> >()
        .add< MechanicalObject<F311dTypes> >()
        .add< MechanicalObject<F332dTypes> >()
        .add< MechanicalObject<F221dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<F331fTypes> >()
        .add< MechanicalObject<F321fTypes> >()
        .add< MechanicalObject<F311fTypes> >()
        .add< MechanicalObject<F332fTypes> >()
        .add< MechanicalObject<F221fTypes> >()
#endif
		;

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API MechanicalObject<F331dTypes>;
template class SOFA_Flexible_API MechanicalObject<F321dTypes>;
template class SOFA_Flexible_API MechanicalObject<F311dTypes>;
template class SOFA_Flexible_API MechanicalObject<F332dTypes>;
template class SOFA_Flexible_API MechanicalObject<F221dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API MechanicalObject<F331fTypes>;
template class SOFA_Flexible_API MechanicalObject<F321fTypes>;
template class SOFA_Flexible_API MechanicalObject<F311fTypes>;
template class SOFA_Flexible_API MechanicalObject<F332fTypes>;
template class SOFA_Flexible_API MechanicalObject<F221fTypes>;
#endif

} // namespace container
} // namespace component
} // namespace sofa
