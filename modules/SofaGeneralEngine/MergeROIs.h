/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef MergeROIs_H_
#define MergeROIs_H_
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/DataEngine.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/vectorData.h>
#include <sofa/helper/SVector.h>


namespace sofa
{
namespace component
{
namespace engine
{

/**
 * This class merges a list of ROIs (vector<Indices>) into a single Data (vector<svector<Indices>>)
 */

class MergeROIs : public sofa::core::DataEngine
{
public:
    typedef core::DataEngine Inherited;

    SOFA_CLASS(MergeROIs,Inherited);
    typedef unsigned int Index;

    //Input
    Data<unsigned int> nbROIs;
    helper::vectorData<helper::vector<Index> > f_indices;

    //Output
    Data<helper::vector<helper::SVector<Index> > > f_outputIndices;

    virtual std::string getTemplateName() const    override {        return templateName(this);    }
    static std::string templateName(const MergeROIs* = NULL)    {        return std::string();    }

    virtual void init() override
    {
        addInput(&nbROIs);
        f_indices.resize(nbROIs.getValue());
        addOutput(&f_outputIndices);
        setDirtyValue();
    }

    virtual void reinit() override
    {
        f_indices.resize(nbROIs.getValue());
        update();
    }


    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( core::objectmodel::BaseObjectDescription* arg ) override
    {
        f_indices.parseSizeData(arg, nbROIs);
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str ) override
    {
        f_indices.parseFieldsSizeData(str, nbROIs);
        Inherit1::parseFields(str);
    }

protected:

    MergeROIs(): Inherited()
        , nbROIs ( initData ( &nbROIs,(unsigned int)0,"nbROIs","size of indices/value vector" ) )
        , f_indices(this, "indices", "ROIs", helper::DataEngineInput)
        , f_outputIndices(initData(&f_outputIndices, "roiIndices", "Vector of ROIs"))
    {
    }

    virtual ~MergeROIs() {}

    virtual void update() override
    {
        size_t nb = nbROIs.getValue();
        f_indices.resize(nb);
        if(!nb) return;

        helper::WriteOnlyAccessor< Data< helper::vector<helper::SVector<Index> > > > outputIndices = f_outputIndices;
        outputIndices.resize(nb);

        for(size_t j=0; j<nb;j++)
        {
            helper::ReadAccessor< Data< helper::vector<Index> > > indices = f_indices[j];
            outputIndices[j].resize(indices.size());
            for(size_t i=0 ; i<indices.size() ; i++) outputIndices[j][i]=indices[i];
        }

        cleanDirty();
    }

};


} // namespace engine
} // namespace component
} // namespace sofa

#endif /* MergeROIs_H_ */
