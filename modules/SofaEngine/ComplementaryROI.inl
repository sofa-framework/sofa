/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "ComplementaryROI.h"

#include <set>
#include <sstream>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
ComplementaryROI<DataTypes>::ComplementaryROI()
    : d_position(initData(&d_position, "position", "input positions"))
    , d_nbSet( initData(&d_nbSet, (unsigned int)0, "nbSet", "number of sets to complement"))
    , vd_setIndices(this, "setIndices",  "particles indices in the set")
    , d_indices( initData(&d_indices, "indices", "indices of the point in the ROI") )
    , d_pointsInROI(initData(&d_pointsInROI, "pointsInROI", "points in the ROI"))
{
    vd_setIndices.resize(d_nbSet.getValue());
}

template <class DataTypes>
ComplementaryROI<DataTypes>::~ComplementaryROI()
{}

/// Parse the given description to assign values to this object's fields and potentially other parameters
template <class DataTypes>
void ComplementaryROI<DataTypes>::parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    vd_setIndices.parseSizeData(arg, d_nbSet);
    Inherit1::parse(arg);
}

/// Assign the field values stored in the given map of name -> value pairs
template <class DataTypes>
void ComplementaryROI<DataTypes>::parseFields ( const std::map<std::string,std::string*>& str )
{
    vd_setIndices.parseFieldsSizeData(str, d_nbSet);
    Inherit1::parseFields(str);
}

template <class DataTypes>
void ComplementaryROI<DataTypes>::init()
{
    addInput(&d_position);
    addInput(&d_nbSet);

    vd_setIndices.resize(d_nbSet.getValue());

    addOutput(&d_indices);
    addOutput(&d_pointsInROI);

    setDirtyValue();
}

template <class DataTypes>
void ComplementaryROI<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void ComplementaryROI<DataTypes>::update()
{
    cleanDirty();

    helper::ReadAccessor<Data<VecCoord> > position(d_position);
    helper::ReadAccessor<Data<unsigned int> > nbSet(d_nbSet);

    helper::WriteAccessor< Data<SetIndex> > indices(d_indices);
    helper::WriteAccessor< Data<VecCoord> > pointsInROI(d_pointsInROI);

    vd_setIndices.resize(nbSet);

    indices.clear();
    pointsInROI.clear();

    std::set<index_type> myIndices;
    for (index_type i=0 ; i<position.size() ; ++i)
        myIndices.insert(i);

    // build the set of indices in the ROI
    for (unsigned int i=0;i<vd_setIndices.size();++i) {
        helper::ReadAccessor< Data<SetIndex> > setIndices(vd_setIndices[i]);
        for (unsigned int j=0;j<setIndices.size();++j) {
            std::set<index_type>::iterator it = myIndices.find(setIndices[j]);
            if (it==myIndices.end())
                serr << "index " << setIndices[j] << " does not exist" << sendl;
            else
                myIndices.erase(it);
        }
    }

    // copy the set of indices to the output
    indices.wref().insert(indices.begin(), myIndices.begin(), myIndices.end());

    // copy positions
    for (unsigned int i=0;i<indices.size();++i)
        pointsInROI.push_back(position[indices[i]]);

    sout << "Created ROI containing " << indices.size() << " points not in " << nbSet << " sets" << sendl;

}

template <class DataTypes>
std::string ComplementaryROI<DataTypes>::getTemplateName() const
{
    return templateName(this);
}

template <class DataTypes>
std::string ComplementaryROI<DataTypes>::templateName(const ComplementaryROI<DataTypes>*)
{
    return DataTypes::Name();
}

} // namespace engine

} // namespace component

} // namespace sofa
