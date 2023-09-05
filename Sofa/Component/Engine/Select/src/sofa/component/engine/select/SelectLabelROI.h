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


namespace sofa::component::engine::select
{

/**
 * Select a subset of labeled points or cells stored in (vector<svector<label>>) given certain labels
 */

template <class _T>
class SelectLabelROI : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SelectLabelROI,_T), DataEngine);

    typedef _T T;
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Inherited, Inherit1);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Index, sofa::Index);

    //Input
    Data<type::vector<type::SVector<T> > > d_labels; ///< lists of labels associated to each point/cell
    Data<type::vector<T> > d_selectLabels; ///< list of selected labels

    //Output
    Data<type::vector<sofa::Index> > d_indices; ///< selected point/cell indices

    void init() override
    {
        setDirtyValue();
    }

    void reinit() override
    {
        update();
    }

protected:

    SelectLabelROI(): Inherit1()
      , d_labels ( initData ( &d_labels,"labels","lists of labels associated to each point/cell" ) )
      , d_selectLabels ( initData ( &d_selectLabels,"selectLabels","list of selected labels" ) )
      , d_indices ( initData ( &d_indices,"indices","selected point/cell indices" ) )
    {
        addInput(&d_labels);
        addInput(&d_selectLabels);
        addOutput(&d_indices);
    }

    ~SelectLabelROI() override {}

    void doUpdate() override
    {
        helper::ReadAccessor< Data< type::vector<T>  > > selectLabels = d_selectLabels;
        // convert to set for efficient look-up
        std::set<T> selectLabelsSet;
        selectLabelsSet.insert(selectLabels.begin(), selectLabels.end());

        helper::ReadAccessor< Data< type::vector<type::SVector<T> >  > > labels = d_labels;
        const size_t nb = labels.size();
        helper::WriteOnlyAccessor< Data< type::vector<sofa::Index> > > indices = d_indices;
        indices.clear();
        for(sofa::Index i=0; i<nb;i++)
        {
            for(size_t j=0; j<labels[i].size();j++)
                if(selectLabelsSet.find(labels[i][j])!=selectLabelsSet.end())
                {
                    indices.push_back(i);
                    break;
                }
        }
    }

};


} //namespace sofa::component::engine::select
