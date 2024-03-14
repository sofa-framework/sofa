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
#include <sofa/component/engine/transform/config.h>



#include <sofa/core/DataEngine.h>
#include <sofa/type/vector.h>
#include <sofa/core/objectmodel/vectorData.h>


namespace sofa::component::engine::transform
{

/**
 * This class returns a list of values from value-indices pairs
 */

class ROIValueMapper : public sofa::core::DataEngine
{
public:
    typedef core::DataEngine Inherited;

    SOFA_CLASS(ROIValueMapper,Inherited);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Real, SReal);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Index, sofa::Index);

    //Input
    Data<unsigned int> nbROIs; ///< size of indices/value vector
    core::objectmodel::vectorData<type::vector<sofa::Index> > f_indices;
    core::objectmodel::vectorData<SReal> f_value;

    //Output
    Data<sofa::type::vector<SReal> > f_outputValues; ///< New vector of values

    //Parameter
    Data<SReal> p_defaultValue; ///< Default value for indices out of ROIs

    void init() override
    {
        f_indices.resize(nbROIs.getValue());
        f_value.resize(nbROIs.getValue());

        setDirtyValue();
    }

    void reinit() override
    {
        f_indices.resize(nbROIs.getValue());
        f_value.resize(nbROIs.getValue());
        update();
    }


    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override
    {
        f_indices.parseSizeData(arg, nbROIs);
        f_value.parseSizeData(arg, nbROIs);
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str ) override
    {
        f_indices.parseFieldsSizeData(str, nbROIs);
        f_value.parseFieldsSizeData(str, nbROIs);
        Inherit1::parseFields(str);
    }

protected:

    ROIValueMapper(): Inherited()
        , nbROIs ( initData ( &nbROIs,(unsigned int)0,"nbROIs","size of indices/value vector" ) )
        , f_indices(this, "indices", "ROIs", sofa::core::objectmodel::DataEngineDataType::DataEngineInput)
        , f_value(this, "value", "Values", sofa::core::objectmodel::DataEngineDataType::DataEngineInput)
        , f_outputValues(initData(&f_outputValues, "outputValues", "New vector of values"))
        , p_defaultValue(initData(&p_defaultValue, (SReal) 0.0, "defaultValue", "Default value for indices out of ROIs"))
    {
        addInput(&nbROIs);
        addOutput(&f_outputValues);
    }

    ~ROIValueMapper() override {}

    void doUpdate() override
    {
        const unsigned int nb = nbROIs.getValue();
        f_indices.resize(nb);
        f_value.resize(nb);
        if(!nb) return;

        const SReal& defaultValue = p_defaultValue.getValue();
        helper::WriteOnlyAccessor< Data< type::vector<SReal> > > outputValues = f_outputValues;
        outputValues.clear();

        for(size_t j=0; j<nb;j++)
        {
            helper::ReadAccessor< Data< type::vector<sofa::Index> > > indices = f_indices[j];
            const SReal& value = f_value[j]->getValue();

            for (const sofa::Index ind : indices)
            {
                if (ind >= outputValues.size())
                {
                    outputValues.wref().resize(ind+1, defaultValue);
                }
                outputValues[ind] = value;
            }
        }
    }

};


} //namespace sofa::component::engine::transform
