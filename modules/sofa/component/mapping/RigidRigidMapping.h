/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_H

#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <vector>

namespace sofa
{


namespace component
{

namespace mapping
{
using namespace sofa::defaulttype;

template <class BasicMapping>
class RigidRigidMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RigidRigidMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::DataTypes OutDataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename Coord::value_type Real;
    enum { N=OutDataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef Vec<N,Real> Vector ;

protected:
    Data < VecCoord > points;
    VecCoord pointsR0;
    Mat rotation;
    class Loader;
    void load(const char* filename);
    Data<sofa::helper::vector<unsigned int> >  repartition;

public:
    Data<unsigned> index;
    sofa::core::objectmodel::DataFileName fileRigidRigidMapping;
    //axis length for display
    Data<double> axisLength;
    Data< bool > indexFromEnd;
    Data< bool > globalToLocalCoords;

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;


    RigidRigidMapping(In* from, Out* to)
        : Inherit(from, to),
          points(initData(&points, "initialPoints", "Initial position of the points")),
          repartition(initData(&repartition,"repartition","number of dest dofs per entry dof")),
          index(initData(&index,(unsigned)0,"index","input DOF index")),
          fileRigidRigidMapping(initData(&fileRigidRigidMapping,"fileRigidRigidMapping","Filename")),
          axisLength(initData( &axisLength, 0.7, "axisLength", "axis length for display")),
          indexFromEnd( initData ( &indexFromEnd,false,"indexFromEnd","input DOF index starts from the end of input DOFs vector") ),
          globalToLocalCoords ( initData ( &globalToLocalCoords,"globalToLocalCoords","are the output DOFs initially expressed in global coordinates" ) )
    {
        this->addAlias(&fileRigidRigidMapping,"filename");
        maskFrom = NULL;
        if (core::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *>(from))
            maskFrom = &stateFrom->forceMask;
        maskTo = NULL;
        if (core::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::behavior::BaseMechanicalState *>(to))
            maskTo = &stateTo->forceMask;
    }

    virtual ~RigidRigidMapping()
    {
    }

    void init();

    //	void disable(); //useless now that points are saved in a Data

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );

    void computeAccFromMapping(  typename Out::VecDeriv& acc_out, const typename In::VecDeriv& v_in, const typename In::VecDeriv& acc_in);

    void draw();

    void clear();

    sofa::helper::vector<unsigned int> getRepartition() {return repartition.getValue(); }

    void setRepartition(unsigned int value);
    void setRepartition(sofa::helper::vector<unsigned int> values);

protected:

    bool getShow(const core::objectmodel::BaseObject* m) const { return m->getContext()->getShowMappings(); }

    bool getShow(const core::behavior::BaseMechanicalMapping* m) const { return m->getContext()->getShowMechanicalMappings(); }
};



using core::Mapping;
using core::behavior::MechanicalMapping;
using core::behavior::MappedModel;
using core::behavior::State;
using core::behavior::MechanicalState;

using sofa::defaulttype::Vec2dTypes;
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec2fTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec2fTypes;
using sofa::defaulttype::ExtVec3fTypes;
using sofa::defaulttype::Rigid2dTypes;
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid2fTypes;
using sofa::defaulttype::Rigid3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Mapping< State<Rigid3dTypes>, MechanicalState<Rigid3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3dTypes> > >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< MechanicalMapping<MechanicalState<Rigid3fTypes>, MechanicalState<Rigid3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Mapping< State<Rigid3fTypes>, MechanicalState<Rigid3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3fTypes> > >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Mapping< State<Rigid3dTypes>, MechanicalState<Rigid3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Mapping< State<Rigid3fTypes>, MechanicalState<Rigid3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3dTypes> > >;
#endif
#endif
#endif







} // namespace mapping

} // namespace component

} // namespace sofa

#endif
