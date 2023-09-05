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

#include <sofa/component/engine/select/ComplementaryROI.h>

#include <set>
#include <sstream>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::engine::select
{

using std::string;
using std::set;
using std::map;
using sofa::core::objectmodel::BaseObjectDescription;

using helper::ReadAccessor;
using helper::WriteAccessor;

template <class DataTypes>
ComplementaryROI<DataTypes>::ComplementaryROI()
    : d_position(initData(&d_position, "position", "input positions"))
    , d_nbSet( initData(&d_nbSet, (unsigned int)0, "nbSet", "number of sets to complement"))
    , vd_setIndices(this, "setIndices",  "particles indices in the set")
    , d_indices( initData(&d_indices, "indices", "indices of the point in the ROI") )
    , d_pointsInROI(initData(&d_pointsInROI, "pointsInROI", "points in the ROI"))
{
    vd_setIndices.resize(d_nbSet.getValue());

    addInput(&d_position);
    addInput(&d_nbSet);

    addOutput(&d_indices);
    addOutput(&d_pointsInROI);
}

template <class DataTypes>
ComplementaryROI<DataTypes>::~ComplementaryROI()
{}

/// Parse the given description to assign values to this object's fields and potentially other parameters
template <class DataTypes>
void ComplementaryROI<DataTypes>::parse ( BaseObjectDescription* arg )
{
    vd_setIndices.parseSizeData(arg, d_nbSet);
    Inherit1::parse(arg);
}

/// Assign the field values stored in the given map of name -> value pairs
template <class DataTypes>
void ComplementaryROI<DataTypes>::parseFields ( const map<string,string*>& str )
{
    vd_setIndices.parseFieldsSizeData(str, d_nbSet);
    Inherit1::parseFields(str);
}

template <class DataTypes>
void ComplementaryROI<DataTypes>::init()
{
    vd_setIndices.resize(d_nbSet.getValue());
    setDirtyValue();
}

template <class DataTypes>
void ComplementaryROI<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void ComplementaryROI<DataTypes>::doUpdate()
{
    ReadAccessor<Data<VecCoord> > position(d_position);
    const ReadAccessor<Data<unsigned int> > nbSet(d_nbSet);

    WriteAccessor<Data<SetIndex> > indices(d_indices);
    WriteAccessor<Data<VecCoord> > pointsInROI(d_pointsInROI);

    vd_setIndices.resize(nbSet);

    indices.clear();
    pointsInROI.clear();

    set<sofa::Index> myIndices;
    for (sofa::Index i=0 ; i<position.size() ; ++i)
        myIndices.insert(i);

    // build the set of indices in the ROI
    for (unsigned int i=0;i<vd_setIndices.size();++i) {
        ReadAccessor< Data<SetIndex> > setIndices(vd_setIndices[i]);
        for (unsigned int j=0;j<setIndices.size();++j) {
            set<sofa::Index>::iterator it = myIndices.find(setIndices[j]);
            if (it == myIndices.end())
                msg_error() << "index " << setIndices[j] << " does not exist";
            else
                myIndices.erase(it);
        }
    }

    // copy the set of indices to the output
    indices.wref().insert(indices.begin(), myIndices.begin(), myIndices.end());

    // copy positions
    for (unsigned int i=0;i<indices.size();++i)
        pointsInROI.push_back(position[indices[i]]);

    msg_info() << "Created ROI containing " << indices.size() << " points not in " << nbSet.ref() << " sets" ;
}

} //namespace sofa::component::engine::select
