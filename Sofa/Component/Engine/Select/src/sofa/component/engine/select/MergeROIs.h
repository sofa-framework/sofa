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
#include <sofa/component/engine/select/config.h>



#include <sofa/core/DataEngine.h>
#include <sofa/type/vector.h>
#include <sofa/core/objectmodel/vectorData.h>
#include <sofa/type/SVector.h>


namespace sofa::component::engine::select
{

/**
 * This class merges a list of ROIs (vector<Indices>) into a single Data (vector<svector<Indices>>)
 */

class SOFA_COMPONENT_ENGINE_SELECT_API MergeROIs : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(MergeROIs, DataEngine);

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Inherited, Inherit1);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Index, sofa::Index);

    //Input
    Data<unsigned int> d_nbROIs; ///< size of indices/value vector
    core::objectmodel::vectorData<type::vector<sofa::Index> > f_indices;

    //Output
    Data<type::vector<type::SVector<sofa::Index> > > d_outputIndices; ///< Vector of ROIs

    void init() override;
    void reinit() override;

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( core::objectmodel::BaseObjectDescription* arg ) override;

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str ) override;

protected:

    MergeROIs(): Inherit1()
        , d_nbROIs ( initData ( &d_nbROIs,(unsigned int)0,"nbROIs","size of indices/value vector" ) )
        , f_indices(this, "indices", "ROIs", sofa::core::objectmodel::DataEngineDataType::DataEngineInput)
        , d_outputIndices(initData(&d_outputIndices, "roiIndices", "Vector of ROIs"))
    {
    }

    ~MergeROIs() override {}

    void doUpdate() override;

};


} //namespace sofa::component::engine::select
