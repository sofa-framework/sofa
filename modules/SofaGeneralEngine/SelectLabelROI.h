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
#ifndef SelectLabelROI_H_
#define SelectLabelROI_H_
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/DataEngine.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/SVector.h>


namespace sofa
{
namespace component
{
namespace engine
{

/**
 * Select a subset of labeled points or cells stored in (vector<svector<label>>) given certain labels
 */

template <class _T>
class SelectLabelROI : public sofa::core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(SelectLabelROI,_T),Inherited);

    typedef _T T;
    typedef unsigned int Index;

    //Input
    Data<helper::vector<helper::SVector<T> > > d_labels;
    Data<helper::vector<T> > d_selectLabels;

    //Output
    Data<helper::vector<Index> > d_indices;

    virtual std::string getTemplateName() const    override {        return templateName(this);    }
    static std::string templateName(const SelectLabelROI* = NULL)    {       return sofa::defaulttype::DataTypeName<T>::name();    }

    virtual void init() override
    {
        addInput(&d_labels);
        addInput(&d_selectLabels);
        addOutput(&d_indices);
        setDirtyValue();
    }

    virtual void reinit() override
    {
        update();
    }

protected:

    SelectLabelROI(): Inherited()
      , d_labels ( initData ( &d_labels,"labels","lists of labels associated to each point/cell" ) )
      , d_selectLabels ( initData ( &d_selectLabels,"selectLabels","list of selected labels" ) )
      , d_indices ( initData ( &d_indices,"indices","selected point/cell indices" ) )
    {
    }

    virtual ~SelectLabelROI() {}

    virtual void update() override
    {
        helper::ReadAccessor< Data< helper::vector<T>  > > selectLabels = d_selectLabels;
        // convert to set for efficient look-up
        std::set<T> selectLabelsSet;
        selectLabelsSet.insert(selectLabels.begin(), selectLabels.end());

        helper::ReadAccessor< Data< helper::vector<helper::SVector<T> >  > > labels = d_labels;
        size_t nb = labels.size();
        helper::WriteOnlyAccessor< Data< helper::vector<Index> > > indices = d_indices;
        indices.clear();
        for(Index i=0; i<nb;i++)
        {
            for(size_t j=0; j<labels[i].size();j++)
                if(selectLabelsSet.find(labels[i][j])!=selectLabelsSet.end())
                {
                    indices.push_back(i);
                    break;
                }
        }

        cleanDirty();
    }

};


} // namespace engine
} // namespace component
} // namespace sofa

#endif /* SelectLabelROI_H_ */
