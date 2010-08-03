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
#ifndef SOFA_COMPONENT_MAPPING_RIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_RIGIDMAPPING_H

#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/component/component.h>
#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MappedModel.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <vector>
#include <memory>

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

template<class BasicMapping>
class RigidMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RigidMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::DataTypes OutDataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename In::Deriv InDeriv;
    typedef typename InDeriv::Rot DRot;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
//    typedef typename defaulttype::SparseConstraint<Deriv> OutSparseConstraint;
//    typedef typename OutSparseConstraint::const_data_iterator OutConstraintIterator;
    typedef typename Coord::value_type Real;
    enum
    {
        N = OutDataTypes::spatial_dimensions
    };
    enum
    {
        NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size
    };
    enum
    {
        NOut = sofa::defaulttype::DataTypeInfo<Deriv>::Size
    };
    typedef defaulttype::Mat<N, N, Real> Mat;
    typedef defaulttype::Vec<N, Real> Vector;
    typedef defaulttype::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<MBloc> MatrixType;

    Data<VecCoord> points;
    VecCoord rotatedPoints;
    RigidMappingInternalData<typename In::DataTypes, typename Out::DataTypes> data;
    Data<unsigned int> index;
    sofa::core::objectmodel::DataFileName fileRigidMapping;
    Data<bool> useX0;
    Data<bool> indexFromEnd;
    /**
     * Repartitions:
     *  - no value specified : simple rigid mapping
     *  - one value specified : uniform repartition mapping on the input dofs
     *  - n values are specified : heterogen repartition mapping on the input dofs
     */
    Data<sofa::helper::vector<unsigned int> > repartition;
    Data<bool> globalToLocalCoords;

    ///// new: functions for continuous friction contact
    Data<bool> contactDuplicate;
    Data<std::string> nameOfInputMap;

    helper::ParticleMask* maskFrom;
    helper::ParticleMask* maskTo;


    RigidMapping ( In* from, Out* to )
        : Inherit ( from, to ),
          points ( initData ( &points,"initialPoints", "Local Coordinates of the points" ) ),
          index ( initData ( &index, ( unsigned ) 0,"index","input DOF index" ) ),
          fileRigidMapping ( initData ( &fileRigidMapping,"fileRigidMapping","Filename" ) ),
          useX0( initData ( &useX0,false,"useX0","Use x0 instead of local copy of initial positions (to support topo changes)") ),
          indexFromEnd( initData ( &indexFromEnd,false,"indexFromEnd","input DOF index starts from the end of input DOFs vector") ),
          repartition ( initData ( &repartition,"repartition","number of dest dofs per entry dof" ) ),
          globalToLocalCoords ( initData ( &globalToLocalCoords,"globalToLocalCoords","are the output DOFs initially expressed in global coordinates" ) ),
          contactDuplicate(initData(&contactDuplicate,false,"contactDuplicate","if true, this mapping is a copy of an input mapping and is used to gather contact points (ContinuousFrictionContact Response)")),
          nameOfInputMap(initData(&nameOfInputMap,"nameOfInputMap", "if contactDuplicate==true, it provides the name of the input mapping")),
          matrixJ(),
          updateJ(false)
    {
        this->addAlias(&fileRigidMapping, "filename");
        maskFrom = NULL;
        if (core::behavior::BaseMechanicalState* stateFrom = dynamic_cast<core::behavior::BaseMechanicalState*>(from))
        {
            maskFrom = &stateFrom->forceMask;
        }
        maskTo = NULL;
        if (core::behavior::BaseMechanicalState* stateTo = dynamic_cast<core::behavior::BaseMechanicalState*>(to))
        {
            maskTo = &stateTo->forceMask;
        }
    }

    virtual ~RigidMapping() {}

    int addPoint(const Coord& c);
    int addPoint(const Coord& c, int indexFrom);

    // interface for continuous friction contact
    void beginAddContactPoint();
    int addContactPointFromInputMapping(const sofa::defaulttype::Vector3& pos, std::vector< std::pair<int, double> > & baryCoords);


    void init();
    void bwdInit();

    //void disable(); //useless now that points are saved in a Data

    virtual void apply(VecCoord& out, const InVecCoord& in);

    virtual void applyJ(VecDeriv& out, const InVecDeriv& in);

    virtual void applyJT(InVecDeriv& out, const VecDeriv& in);

    void applyJT(InMatrixDeriv& out, const OutMatrixDeriv& in);

    const sofa::defaulttype::BaseMatrix* getJ();

    void draw();

    void clear(int reserve = 0);

    void setRepartition(unsigned int value);
    void setRepartition(sofa::helper::vector<unsigned int> values);

protected:
    class Loader;

    void load(const char* filename);
    const VecCoord& getPoints();
    void setJMatrixBlock(unsigned outIdx, unsigned inIdx);

    RigidMapping<BasicMapping> *_inputMapping; // for continuous_friction_contact:

    std::auto_ptr<MatrixType> matrixJ;
    bool updateJ;
};

template <int N, class Real>
struct RigidMappingMatrixHelper;

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

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_RIGIDMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid2dTypes>, MechanicalState<Vec2dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
// extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid2fTypes>, MechanicalState<Vec2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;
// extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid2dTypes>, MechanicalState<Vec2fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< MechanicalMapping<MechanicalState<Rigid2fTypes>, MechanicalState<Vec2dTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
