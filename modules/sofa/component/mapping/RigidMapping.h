/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_RIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_RIGIDMAPPING_H

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class RigidMappingInternalData
{
public:
};

template <class BasicMapping>
class RigidMapping : public BasicMapping, public virtual core::objectmodel::BaseObject
{
public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename In::Deriv InDeriv;
    typedef typename In::SparseDeriv InSparseDeriv;
    typedef typename Coord::value_type Real;
    enum { N=Coord::static_size };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef typename sofa::defaulttype::Vector3::value_type Real_Sofa;

    Data< VecCoord > points;
    VecCoord rotatedPoints;
    RigidMappingInternalData<typename In::DataTypes, typename Out::DataTypes> data;
    Data<unsigned int> index;
    Data< std::string > filename;

    RigidMapping ( In* from, Out* to )
        : Inherit ( from, to ),
          points ( initData ( &points,"initialPoints", "Local Coordinates of the points" ) ),
          index ( initData ( &index, ( unsigned ) 0,"index","input DOF index" ) ),
          filename ( initData ( &filename,"filename","Filename" ) ),
          repartition ( initData ( &repartition,"repartition","number of dest dofs per entry dof" ) )
    {
    }

    virtual ~RigidMapping()
    {}

    int addPoint ( const Coord& c );
    int addPoint ( const Coord& c, int indexFrom );

    void init();

    //void disable(); //useless now that points are saved in a Data

    void parse ( core::objectmodel::BaseObjectDescription* arg )
    {
        if ( !filename.getValue().empty() ) this->load ( filename.getValue().c_str() );
        this->Inherit::parse ( arg );
    }

    virtual void apply ( typename Out::VecCoord& out, const typename In::VecCoord& in );

    virtual void applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    virtual void applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT ( typename In::VecConst& out, const typename Out::VecConst& in );

    void draw();

    void clear ( int reserve=0 );

    void setRepartition ( unsigned int value );
    void setRepartition ( sofa::helper::vector<unsigned int> values );

protected:
    class Loader;
    void load ( const char* filename );
    Data<sofa::helper::vector<unsigned int> >  repartition;
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
