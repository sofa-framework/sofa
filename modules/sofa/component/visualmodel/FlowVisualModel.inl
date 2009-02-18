/*
 * FlowVisualModel.h
 *
 *  Created on: 18 févr. 2009
 *      Author: froy
 */

#ifndef FLOWVISUALMODEL_INL_
#define FLOWVISUALMODEL_INL_

#include "FlowVisualModel.h"

namespace sofa
{

namespace component
{

namespace visualmodel
{

template <class DataTypes>
FlowVisualModel<DataTypes>::FlowVisualModel()
{

}

template <class DataTypes>
FlowVisualModel<DataTypes>::~FlowVisualModel()
{

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    fstate = context->core::objectmodel::BaseContext::get<FluidState>();
    if (fstate)
        std::cout << "cool." << std::endl;
}

template <class DataTypes>
void FlowVisualModel<DataTypes>::initVisual()
{

}

template <class DataTypes>
void FlowVisualModel<DataTypes>::draw()
{

}



}

}

}

#endif /* FLOWVISUALMODEL_INL_ */
