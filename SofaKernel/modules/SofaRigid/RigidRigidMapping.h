/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_H
#include "config.h"

#include <sofa/core/Mapping.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/visual/VisualParams.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
class RigidRigidMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(RigidRigidMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef Out OutDataTypes;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename Out::Coord::value_type Real;
    enum { N=OutDataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef defaulttype::Vec<N,Real> Vector ;
    typedef typename Inherit::ForceMask ForceMask;

protected:
    Data < OutVecCoord > points; ///< Initial position of the points
    OutVecCoord pointsR0;
    Mat rotation;
    class Loader;
    void load(const char* filename);
    /// number of child frames per parent frame.
    /// If empty, all the children are attached to the parent with index
    /// given in the "index" attribute. If one value, each parent frame drives
    /// the given number of children frames. Otherwise, the values are the number
    /// of child frames driven by each parent frame.
    Data< sofa::helper::vector<unsigned int> >  repartition;

public:
    Data<unsigned> index; ///< input frame index
    sofa::core::objectmodel::DataFileName fileRigidRigidMapping; ///< Filename
    //axis length for display
    Data<double> axisLength; ///< axis length for display
    Data< bool > indexFromEnd; ///< input DOF index starts from the end of input DOFs vector
    Data< bool > globalToLocalCoords; ///< are the output DOFs initially expressed in global coordinates

protected:
    RigidRigidMapping()
        : Inherit(),
          points(initData(&points, "initialPoints", "Initial position of the points")),
          repartition(initData(&repartition,"repartition","number of child frames per parent frame. \n"
                               "If empty, all the children are attached to the parent with index \n"
                               "given in the \"index\" attribute. If one value, each parent frame drives \n"
                               "the given number of children frames. Otherwise, the values are the number \n"
                               "of child frames driven by each parent frame. ")),
          index(initData(&index,(unsigned)0,"index","input frame index")),
          fileRigidRigidMapping(initData(&fileRigidRigidMapping,"fileRigidRigidMapping","Filename")),
          axisLength(initData( &axisLength, 0.7, "axisLength", "axis length for display")),
          indexFromEnd( initData ( &indexFromEnd,false,"indexFromEnd","input DOF index starts from the end of input DOFs vector") ),
          globalToLocalCoords ( initData ( &globalToLocalCoords,"globalToLocalCoords","are the output DOFs initially expressed in global coordinates" ) )
    {
        this->addAlias(&fileRigidRigidMapping,"filename");
    }

    virtual ~RigidRigidMapping()
    {
    }
public:
    virtual void init() override;

    virtual void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in) override;

    virtual void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in) override;

    virtual void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in) override;

    virtual void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in) override;

    virtual void computeAccFromMapping(const core::MechanicalParams *mparams, Data<OutVecDeriv>& acc_out, const Data<InVecDeriv>& v_in, const Data<InVecDeriv>& acc_in) override;

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce ) override;

    virtual const sofa::defaulttype::BaseMatrix* getJ() override
    {
        return NULL;
    }

    void draw(const core::visual::VisualParams* vparams) override;

    void clear();

    sofa::helper::vector<unsigned int> getRepartition() {return repartition.getValue(); }

    void setRepartition(unsigned int value);
    void setRepartition(sofa::helper::vector<unsigned int> values);

protected:

    bool getShow(const core::objectmodel::BaseObject* /*m*/, const core::visual::VisualParams* vparams) const { return vparams->displayFlags().getShowMappings(); }

    bool getShow(const core::BaseMapping* /*m*/, const core::visual::VisualParams* vparams) const { return vparams->displayFlags().getShowMechanicalMappings(); }

    virtual void updateForceMask() override { /*already done in applyJT*/ }
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_RIGID_API RigidRigidMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_RIGID_API RigidRigidMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_RIGID_API RigidRigidMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Rigid3fTypes >;
extern template class SOFA_RIGID_API RigidRigidMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Rigid3dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
