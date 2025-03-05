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
#include <sofa/type/SVector.h>
#include <sofa/core/objectmodel/vectorData.h>

namespace sofa::component::engine::select
{

/**
 * Select a subset of points or cells labeled from different sources, that are connected given a list of connection pairs
 */

template <class _T>
class SelectConnectedLabelsROI : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SelectConnectedLabelsROI,_T),DataEngine);

    typedef _T T;

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Inherited, Inherit1);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Index, sofa::Index);

    //Input
    Data<unsigned int> d_nbLabels; ///< number of label lists
    typedef type::vector<type::SVector<T> > VecVLabels;
    core::objectmodel::vectorData<VecVLabels> d_labels;
    Data<type::vector<T> > d_connectLabels; ///< Pairs of label to be connected across different label lists

    //Output
    Data<type::vector<sofa::Index> > d_indices; ///< selected point/cell indices

    SelectConnectedLabelsROI(): Inherit1()
      , d_nbLabels ( initData ( &d_nbLabels,(unsigned int)0,"nbLabels","number of label lists" ) )
      , d_labels(this, "labels", "lists of labels associated to each point/cell", core::objectmodel::DataEngineDataType::DataEngineInput)
      , d_connectLabels ( initData ( &d_connectLabels,"connectLabels","Pairs of label to be connected across different label lists" ) )
      , d_indices ( initData ( &d_indices,"indices","selected point/cell indices" ) )
    {
        d_labels.resize(d_nbLabels.getValue());

        addInput(&d_nbLabels);
        addInput(&d_connectLabels);
        addOutput(&d_indices);
    }

    ~SelectConnectedLabelsROI() override {}

    void init() override
    {
        d_labels.resize(d_nbLabels.getValue());

        setDirtyValue();
    }

    void reinit() override
    {
        d_labels.resize(d_nbLabels.getValue());
        update();
    }


    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override
    {
        d_labels.parseSizeData(arg, d_nbLabels);
        Inherit1::parse(arg);
    }

    /// Assign the field values stored in the given map of name -> value pairs
    void parseFields ( const std::map<std::string,std::string*>& str ) override
    {
        d_labels.parseFieldsSizeData(str, d_nbLabels);
        Inherit1::parseFields(str);
    }

protected:


    void doUpdate() override
    {
        helper::WriteOnlyAccessor< Data< type::vector<sofa::Index> > > indices = d_indices;
        indices.clear();

        const unsigned int nb = d_nbLabels.getValue();
        if(nb<2) return;

        // convert connectLabels to set for efficient look-up
        helper::ReadAccessor<Data< type::vector<T> > > connectL(this->d_connectLabels);
        typedef std::pair<T,T>  TPair;
        std::set<TPair> connectS;
        for(unsigned int i=0;i<connectL.size()/2;i++)
        {
            connectS.insert(TPair(connectL[2*i],connectL[2*i+1]));
            connectS.insert(TPair(connectL[2*i+1],connectL[2*i]));
        }

        // read input label lists
        std::vector<const VecVLabels* > labels;
        for(unsigned int l=0; l<nb ; ++l)
        {
            helper::ReadAccessor< Data<VecVLabels> > rlab(d_labels[l]);
            labels.push_back(&rlab.ref());
        }

        const size_t nbp = (*labels[0]).size();
        for(size_t i=0; i<nbp;i++)
        {
            bool connected = false;
            for(unsigned int l1=0; l1<nb && !connected;l1++)
                for(size_t i1=0; i1<(*labels[l1])[i].size() && !connected;i1++)
                    for(unsigned int l2=0; l2<nb && !connected;l2++)
                        if(l1!=l2)
                            for(size_t i2=0; i2<(*labels[l2])[i].size() && !connected;i2++)
                                if(connectS.contains(TPair((*labels[l1])[i][i1],(*labels[l2])[i][i2])))
                                    connected=true;
            if(connected)
                indices.push_back((sofa::Index)i);
        }
    }

};

#ifndef SelectConnectedLabelsROI_CPP_
extern template class SOFA_COMPONENT_ENGINE_SELECT_API SelectConnectedLabelsROI<unsigned int>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API SelectConnectedLabelsROI<unsigned char>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API SelectConnectedLabelsROI<unsigned short>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API SelectConnectedLabelsROI<int>;
#endif ///

} //namespace sofa::component::engine::select
