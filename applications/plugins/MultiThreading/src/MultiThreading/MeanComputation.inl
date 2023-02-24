/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <MultiThreading/MeanComputation.h>

#include <sofa/component/statecontainer/MechanicalObject.h>

#include <sofa/simulation/AnimateBeginEvent.h>

namespace sofa::component::engine
{

using sofa::component::statecontainer::MechanicalObject;

template <class DataTypes>
MeanComputation<DataTypes>::MeanComputation()
    //: d_inputs(initData(&d_inputs, "input", "List of all input values for mean computation"))
    : d_result(initData(&d_result, "result", "Result: mean computed from the input values"))
{
}

template <class DataTypes>
void MeanComputation<DataTypes>::init()
{
    f_listening.setValue(true);

    helper::ReadAccessor< Data<VecCoord> > output = d_result;

    std::vector<MechanicalObject<DataTypes>*> mechObjs;
    this->getContext()->template get<MechanicalObject<DataTypes> >(
        &mechObjs,
        core::objectmodel::BaseContext::Local);

    // temp map to check Mech obj data input size
    std::map<size_t, std::string> MechObjSizeMap;

    for (auto mObj : mechObjs)
    {
        core::objectmodel::TagSet tags = mObj->getTags();
        if (tags.find(core::objectmodel::Tag("MeanOutput")) != tags.end())
        {
            mObj->x.setValue(*d_result.beginEdit());
            mObj->x.setParent(&d_result, std::string("to"));

        }
        else
        {
            helper::ReadAccessor<Data<VecCoord> >  inputpos = mObj->readPositions();
            //helper::WriteAccessor<Data<VecCoord> >  output =  mObj->writePositions();

            const VecCoord& positions = inputpos.ref();

            if (output.size() > 0 && &output[0] == &positions[0])
            {
                continue;
            }

            const size_t size = mObj->x.beginEdit()->size();
            auto iter = MechObjSizeMap.find(size);

            if (MechObjSizeMap.size() > 0 && iter== MechObjSizeMap.end())
            {
                dmsg_warning("Different input data size detected. MechanicalObject name " +
                    iter->second + " and " + MechObjSizeMap.begin()->second +
                    ". The lower size will be used.\n");
                //serr << "Different input data size. The lower size will be used.\n";
            }

            MechObjSizeMap[size] = mObj->getName();

            // create and add a new input data for mechanical object positions (mObj->x)
            Data<VecCoord>* input = new Data<VecCoord>();
            input->setValue(*mObj->x.beginEdit());
            input->setParent(&mObj->x, std::string("to"));
            input->setReadOnly(true);
            input->setDirtyValue();

            _inputs.push_back(input);
        }
    }

    // get the lowest size
    _resultSize = MechObjSizeMap.begin()->first;
    d_result.beginEdit()->resize(_resultSize);

    compute();
}

template <class DataTypes>
void MeanComputation<DataTypes>::reinit()
{
    compute();
}


template <class DataTypes>
void MeanComputation<DataTypes>::handleEvent(core::objectmodel::Event* event)
{
    if (dynamic_cast<simulation::AnimateBeginEvent*>(event) != NULL)
    {
        compute();
    }
}


template <class DataTypes>
void MeanComputation<DataTypes>::compute()
{
    VecCoord& result = *d_result.beginEdit();

    const size_t nbInputs = _inputs.size();
    if (nbInputs == 0)
        return;

    const double invNbInputs = 1.0 / nbInputs;

    // accumulate all the input elems in result
    for (size_t j = 0; j<nbInputs; ++j)
    {
        const VecCoord& pos = *_inputs[j]->beginEdit();

        for (size_t i = 0; i<_resultSize; ++i)
        {
            result[i] += pos[i];
        }
    }

    for (size_t i = 0; i<_resultSize; ++i)
    {
        result[i] *= invNbInputs;
    }

    d_result.endEdit();
//                d_result.setDirtyValue();
}

} // namespace sofa::component::engine
