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
#ifndef SOFA_COMPONENT_MAPPING_DEFORMABLEONRIGIDFRAME_H
#define SOFA_COMPONENT_MAPPING_DEFORMABLEONRIGIDFRAME_H

#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MappedModel.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <vector>
#include <sofa/core/behavior/BaseMass.h>

namespace sofa
{

namespace component
{

namespace mapping
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class DeformableOnRigidFrameMappingInternalData
{
public:
};


template <class BasicMapping>
class DeformableOnRigidFrameMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DeformableOnRigidFrameMapping,BasicMapping), BasicMapping);

    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::DataTypes OutDataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename In::Deriv InDeriv;
    typedef typename defaulttype::SparseConstraint<Deriv> OutSparseConstraint;
    typedef typename OutSparseConstraint::const_data_iterator OutConstraintIterator;
    typedef typename Coord::value_type Real;
    enum { N=OutDataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef defaulttype::Vec<N,Real> Vector ;

    typedef sofa::core::behavior::MechanicalState<defaulttype::Rigid3dTypes> InRoot;
    typedef typename InRoot::Coord InRootCoord;
    typedef typename InRoot::VecCoord InRootVecCoord;
    typedef typename InRoot::Deriv InRootDeriv;
    typedef typename InRoot::VecDeriv InRootVecDeriv;

    typedef typename core::behavior::BaseMechanicalState::VecId VecId;

    InRoot* rootModel;
    Data<std::string> m_rootModelName;

    //Data< VecCoord > points;
    VecCoord rotatedPoints;
    DeformableOnRigidFrameMappingInternalData<typename In::DataTypes, typename Out::DataTypes> data;
    Data<unsigned int> index;
    sofa::core::objectmodel::DataFileName fileDeformableOnRigidFrameMapping;
    Data< bool > useX0;
    Data< bool > indexFromEnd;
    Data<sofa::helper::vector<unsigned int> >  repartition;
    Data< bool > globalToLocalCoords;

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;


    DeformableOnRigidFrameMapping ( In* from, Out* to );

    virtual ~DeformableOnRigidFrameMapping()
    {}

    int addPoint ( const Coord& c );
    int addPoint ( const Coord& c, int indexFrom );

    void init();

    //override mapping methods to handle a second "In" component
    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in, const typename InRoot::VecCoord * inroot  );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in, const typename InRoot::VecDeriv* inroot );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in, typename InRoot::VecDeriv* outroot );

    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in, typename InRoot::MatrixDeriv* outroot );


    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
    {
        //serr<<"WARNING apply without rigid frame is called "<<sendl;
        apply(out, in, NULL);
    }
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
    {
        applyJ(out,in, NULL);
    }

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
    {
        applyJT(out,in, NULL);
    }

    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
    {
        applyJT(out,in, NULL);
    }

    /**
    	 * @name
    	 */
    //@{
    /**
     * @brief
     */
    void propagateX();

    /**
     * @brief
     */
    void propagateXfree();


    /**
     * @brief
     */
    void propagateV();

    /**
     * @brief
     */
    void propagateDx();

    /**
     * @brief
     */
    void accumulateForce();

    /**
     * @brief
     */
    void accumulateDf();

    /**
     * @brief
     */
    void accumulateConstraint();

    /**
      * @brief
      MAP the mass: this function recompute the rigid mass (gravity center position and inertia) of the object
          based on its deformed shape
      */
    void recomputeRigidMass() {}

    //@}

    void draw();

    void clear ( int reserve=0 );

    void setRepartition ( unsigned int value );
    void setRepartition ( sofa::helper::vector<unsigned int> values );

protected:
    class Loader;
    void load ( const char* filename );  /// SUPRESS ? ///
    //const VecCoord& getPoints();         /// SUPRESS ? ///
    InRoot::Coord rootX;
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

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_DEFORMABLEONRIGIDFRAME_CPP)  //// ATTENTION PB COMPIL WIN3Z
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< MechanicalMapping<MechanicalState<Rigid2dTypes>, MechanicalState<Vec2dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
// extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< MechanicalMapping<MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< MechanicalMapping<MechanicalState<Rigid2fTypes>, MechanicalState<Vec2fTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;
// extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > >;
//extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif

/*
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< MechanicalMapping<MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< MechanicalMapping<MechanicalState<Rigid2dTypes>, MechanicalState<Vec2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< MechanicalMapping<MechanicalState<Rigid2fTypes>, MechanicalState<Vec2dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API DeformableOnRigidFrameMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif
*/
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
