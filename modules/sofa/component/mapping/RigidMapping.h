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

#include <sofa/core/Mapping.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/component/component.h>

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

template <class TIn, class TOut>
class RigidMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(RigidMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef Out OutDataTypes;
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
    RigidMappingInternalData<In, Out> data;
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

    RigidMapping(core::State< In >* from, core::State< Out >* to);
    virtual ~RigidMapping() {}

    int addPoint(const Coord& c);
    int addPoint(const Coord& c, int indexFrom);

    // interface for continuous friction contact
    void beginAddContactPoint();
    int addContactPointFromInputMapping(const sofa::defaulttype::Vector3& pos, std::vector< std::pair<int, double> > & baryCoords);


    void init();
    void bwdInit();

    //void disable(); //useless now that points are saved in a Data

    void apply(Data<VecCoord>& out, const Data<InVecCoord>& in, const core::MechanicalParams *mparams);

    void applyJ(Data<VecDeriv>& out, const Data<InVecDeriv>& in, const core::MechanicalParams *mparams);

    void applyJT(Data<InVecDeriv>& out, const Data<VecDeriv>& in, const core::MechanicalParams *mparams);

    void applyJT(Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in, const core::ConstraintParams *cparams);

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

    RigidMapping<TIn, TOut> *_inputMapping; // for continuous_friction_contact:

    std::auto_ptr<MatrixType> matrixJ;
    bool updateJ;
};

template <int N, class Real>
struct RigidMappingMatrixHelper;

using sofa::defaulttype::Vec2dTypes;
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec2fTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3fTypes;
using sofa::defaulttype::Rigid2dTypes;
using sofa::defaulttype::Rigid3dTypes;
using sofa::defaulttype::Rigid2fTypes;
using sofa::defaulttype::Rigid3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_RIGIDMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Rigid3dTypes, Vec3dTypes >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Rigid2dTypes, Vec2dTypes >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Rigid3dTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Rigid3fTypes, Vec3fTypes >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Rigid2fTypes, Vec2fTypes >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Rigid3fTypes, ExtVec3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Rigid3dTypes, Vec3fTypes >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Rigid3fTypes, Vec3dTypes >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Rigid2dTypes, Vec2fTypes >;
extern template class SOFA_COMPONENT_MAPPING_API RigidMapping< Rigid2fTypes, Vec2dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
