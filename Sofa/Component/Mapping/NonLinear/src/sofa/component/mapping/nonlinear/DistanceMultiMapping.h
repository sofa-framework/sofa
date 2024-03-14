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
#pragma once

#include <sofa/component/mapping/nonlinear/config.h>
#include <sofa/core/MultiMapping.h>
#include <sofa/component/mapping/nonlinear/NonLinearMappingData.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>

namespace sofa::component::mapping::nonlinear
{


/** Maps point positions from serveral mstates to distances (in distance unit).
 * @tparam TIn parent point positions
 * @tparam TOut corresponds to a scalar value: distance between point pairs, minus a rest distance.
 * The pairs are given in a topology with edges in the same node.
 * The points index are given as pair(mstate_index,dof_index) in the Data indexPairs.
 * If the rest lengths are not defined, they are set using the initial values.
 * If computeDistance is set to true, the rest lengths are set to 0.
 * @author Matthieu Nesme
 */
template <class TIn, class TOut>
class DistanceMultiMapping : public core::MultiMapping<TIn, TOut>, public NonLinearMappingData<true>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DistanceMultiMapping,TIn,TOut), SOFA_TEMPLATE2(core::MultiMapping,TIn,TOut));

    typedef core::MultiMapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;
    typedef typename In::Real InReal;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;
    typedef linearalgebra::EigenSparseMatrix<TIn,TOut>   SparseMatrixEigen;
    typedef linearalgebra::EigenSparseMatrix<TIn,TIn>    SparseKMatrixEigen;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };
    typedef typename type::vector<const InVecCoord*> vecConstInVecCoord;
    typedef sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef type::Vec<In::spatial_dimensions,Real> Direction;


    Data<bool> f_computeDistance;                    ///< if 'computeDistance = true', then rest length of each element equal 0, otherwise rest length is the initial lenght of each of them
    Data<type::vector<Real>> f_restLengths;          ///< Rest lengths of the connections
    Data<Real> d_showObjectScale;                    ///< Scale for object display
    Data<sofa::type::RGBAColor> d_color;             ///< Color for object display. (default=[1.0,1.0,0.0,1.0])
    Data<type::vector<type::Vec2i>> d_indexPairs;    ///< list of couples (parent index + index in the parent)

    /// Link to be set to the topology container in the component graph.
    SingleLink<DistanceMultiMapping<TIn, TOut>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;


    // Append particle of given index within the given model to the subset.
    void addPoint(const core::BaseState* fromModel, int index);
    // Append particle of given index within the given model to the subset.
    void addPoint(int fromModel, int index);



    void init() override;

    void apply(const core::MechanicalParams *mparams, const type::vector<OutDataVecCoord*>& dataVecOutPos, const type::vector<const InDataVecCoord*>& dataVecInPos) override
    {
        SOFA_UNUSED(mparams);

        //Not optimized at all...
        type::vector<OutVecCoord*> vecOutPos;
        for(unsigned int i=0; i<dataVecOutPos.size(); i++)
            vecOutPos.push_back(dataVecOutPos[i]->beginEdit());

        type::vector<const InVecCoord*> vecInPos;
        for(unsigned int i=0; i<dataVecInPos.size(); i++)
            vecInPos.push_back(&dataVecInPos[i]->getValue());

        this->apply(vecOutPos, vecInPos);

        //Really Not optimized at all...
        for(unsigned int i=0; i<dataVecOutPos.size(); i++)
            dataVecOutPos[i]->endEdit();

    }

    void applyJ(const core::MechanicalParams *mparams, const type::vector<OutDataVecDeriv*>& dataVecOutVel, const type::vector<const InDataVecDeriv*>& dataVecInVel) override
    {
        SOFA_UNUSED(mparams);

        //Not optimized at all...
        type::vector<OutVecDeriv*> vecOutVel;
        for(unsigned int i=0; i<dataVecOutVel.size(); i++)
            vecOutVel.push_back(dataVecOutVel[i]->beginEdit());

        type::vector<const InVecDeriv*> vecInVel;
        for(unsigned int i=0; i<dataVecInVel.size(); i++)
            vecInVel.push_back(&dataVecInVel[i]->getValue());

        this->applyJ(vecOutVel, vecInVel);

        //Really Not optimized at all...
        for(unsigned int i=0; i<dataVecOutVel.size(); i++)
            dataVecOutVel[i]->endEdit();

    }

    void applyJT(const core::MechanicalParams *mparams, const type::vector<InDataVecDeriv*>& dataVecOutForce, const type::vector<const OutDataVecDeriv*>& dataVecInForce) override
    {
        SOFA_UNUSED(mparams);

        //Not optimized at all...
        type::vector<InVecDeriv*> vecOutForce;
        for(unsigned int i=0; i<dataVecOutForce.size(); i++)
            vecOutForce.push_back(dataVecOutForce[i]->beginEdit());

        type::vector<const OutVecDeriv*> vecInForce;
        for(unsigned int i=0; i<dataVecInForce.size(); i++)
            vecInForce.push_back(&dataVecInForce[i]->getValue());

        this->applyJT(vecOutForce, vecInForce);

        //Really Not optimized at all...
        for(unsigned int i=0; i<dataVecOutForce.size(); i++)
            dataVecOutForce[i]->endEdit();

    }

    using Inherit::apply;
    using Inherit::applyJ;
    using Inherit::applyJT;

    virtual void apply(const type::vector<OutVecCoord*>& outPos, const vecConstInVecCoord& inPos);
    virtual void applyJ(const type::vector<OutVecDeriv*>& outDeriv, const type::vector<const  InVecDeriv*>& inDeriv);
    virtual void applyJT(const type::vector< InVecDeriv*>& outDeriv, const type::vector<const OutVecDeriv*>& inDeriv);
    void applyJT( const core::ConstraintParams* /* cparams */, const type::vector< InDataMatrixDeriv* >& /* dataMatOutConst */, const type::vector< const OutDataMatrixDeriv* >& /* dataMatInConst */ ) override {}
    void applyDJT(const core::MechanicalParams*, core::MultiVecDerivId inForce, core::ConstMultiVecDerivId outForce) override;

    virtual const type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override;

    void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForce ) override;
    const linearalgebra::BaseMatrix* getK() override;
    void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices) override;

    void draw(const core::visual::VisualParams* vparams) override;

protected:
    DistanceMultiMapping();
    ~DistanceMultiMapping() override;

    type::vector<linearalgebra::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector
    type::vector<Direction> directions;                         ///< Unit vectors in the directions of the lines
    type::vector< Real > invlengths;                          ///< inverse of current distances. Null represents the infinity (null distance)

    SparseKMatrixEigen K;

    /// r=b-a only for position (eventual rotation, affine transform... remains null)
    void computeCoordPositionDifference( Direction& r, const InCoord& a, const InCoord& b );


private:

    // allocate jacobians
    virtual void alloc()
    {
        const unsigned n = this->getFrom().size();
        if (n!=baseMatrices.size())
        {
            release(n); // will only do something if n<oldsize
            const size_t oldsize = baseMatrices.size();
            baseMatrices.resize(n);
            for (unsigned i = oldsize; i<n; ++i) // will only do something if n>oldsize
                baseMatrices[i] = new SparseMatrixEigen;
        }
    }

    // delete jacobians
    void release(size_t from = 0)
    {
        for (unsigned i = from, n = baseMatrices.size(); i<n; ++i)
        {
            delete baseMatrices[i];
            baseMatrices[i] = 0;
        }
    }


};

#if !defined(SOFA_COMPONENT_MAPPING_DistanceMultiMapping_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceMultiMapping< defaulttype::Vec3Types, defaulttype::Vec1Types >;
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceMultiMapping< defaulttype::Rigid3Types, defaulttype::Vec1Types >;
#endif

}
