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

#ifndef SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_H
#define SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_H

#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vector>
#include <sofa/component/container/ArticulatedHierarchyContainer.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace sofa::component::container;
//using namespace sofa::simulation::tree;

template <class BasicMapping>
class ArticulatedSystemMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ArticulatedSystemMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::Real Real;
    typedef typename OutCoord::value_type OutReal;

    typedef sofa::core::behavior::MechanicalState<typename Out::DataTypes> InRoot;
    typedef typename InRoot::VecCoord InRootVecCoord;
    typedef typename InRoot::VecDeriv InRootVecDeriv;

    typedef typename core::behavior::BaseMechanicalState::VecId VecId;

    InRoot* rootModel;

    /*
    ArticulatedSystemMapping(In* from, Out* to)
    : Inherit(from, to), rootModel(NULL), ahc(NULL)
    , m_rootModelName(initData(&m_rootModelName, std::string(""), "rootModel", "Root position if a rigid root model is specified."))
    {
    }
    */

    ArticulatedSystemMapping(In* from, Out* to);

    virtual ~ArticulatedSystemMapping()
    {
    }

    void init();
    void reset();


    //void applyOld( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
    {
        apply(out, in, NULL);
    }

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in, const typename InRoot::VecCoord* inroot  );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in, const typename InRoot::VecDeriv* inroot );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in, typename InRoot::VecDeriv* outroot );

    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in, typename InRoot::MatrixDeriv* outroot );


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

    //@}

    void draw();

    /**
    *	Stores al the articulation centers
    */
    vector<ArticulatedHierarchyContainer::ArticulationCenter*> articulationCenters;

    ArticulatedHierarchyContainer* ahc;

    Data<std::string> m_rootModelName;

private:
    Vec<1,Quat> Buf_Rotation;
    std::vector< Vec<3,OutReal> > ArticulationAxis;
    std::vector< Vec<3,OutReal> > ArticulationPos;
    InVecCoord CoordinateBuf;
    InVecDeriv dxVec1Buf;
    OutVecDeriv dxRigidBuf;
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3dTypes> > >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3fTypes> > >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3fTypes> > >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
