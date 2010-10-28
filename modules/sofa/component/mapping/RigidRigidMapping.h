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

#include <sofa/component/component.h>

#include <sofa/core/Mapping.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>

#include <vector>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

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
    typedef Vec<N,Real> Vector ;

protected:
    Data < OutVecCoord > points;
    OutVecCoord pointsR0;
    Mat rotation;
    class Loader;
    void load(const char* filename);
    Data< sofa::helper::vector<unsigned int> >  repartition;

public:
    Data<unsigned> index;
    sofa::core::objectmodel::DataFileName fileRigidRigidMapping;
    //axis length for display
    Data<double> axisLength;
    Data< bool > indexFromEnd;
    Data< bool > globalToLocalCoords;

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;

    RigidRigidMapping(core::State< In >* from, core::State< Out >* to)
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

    void apply(Data<OutVecCoord>& out, const Data<InVecCoord>& in, const core::MechanicalParams *mparams);

    void applyJ(Data<OutVecDeriv>& out, const Data<InVecDeriv>& in, const core::MechanicalParams *mparams);

    void applyJT(Data<InVecDeriv>& out, const Data<OutVecDeriv>& in, const core::MechanicalParams *mparams);

    void applyJT(Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in, const core::ConstraintParams *cparams);

    void computeAccFromMapping(Data<OutVecDeriv>& acc_out, const Data<InVecDeriv>& v_in, const Data<InVecDeriv>& acc_in, const core::MechanicalParams *mparams);

    void draw();

    void clear();

    sofa::helper::vector<unsigned int> getRepartition() {return repartition.getValue(); }

    void setRepartition(unsigned int value);
    void setRepartition(sofa::helper::vector<unsigned int> values);

protected:

    bool getShow(const core::objectmodel::BaseObject* m) const { return m->getContext()->getShowMappings(); }

    bool getShow(const core::BaseMapping* m) const { return m->getContext()->getShowMechanicalMappings(); }
};

using sofa::defaulttype::Rigid2dTypes;
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid2fTypes;
using sofa::defaulttype::Rigid3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Rigid3dTypes, Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Rigid3fTypes, Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Rigid3dTypes, Rigid3fTypes >;
extern template class SOFA_COMPONENT_MAPPING_API RigidRigidMapping< Rigid3fTypes, Rigid3dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
