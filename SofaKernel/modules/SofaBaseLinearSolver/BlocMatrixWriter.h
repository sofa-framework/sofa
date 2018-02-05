/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_BLOCMATRIXWRITER_H
#define SOFA_COMPONENT_LINEARSOLVER_BLOCMATRIXWRITER_H
#include "config.h"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// This class is a helper to efficiently implement addKToMatrix in forcefields (and is could later be used for mapping, etc.)
/// See TriangularFEMForceFieldOptim for an example.
template<typename TBloc>
class BlocMatrixWriter
{
public:
    typedef TBloc Bloc;
    typedef matrix_bloc_traits<Bloc> traits;
    typedef typename traits::Real Real;
    enum { NL = traits::NL };
    enum { NC = traits::NC };

    typedef Bloc MatBloc;

    class BaseMatrixWriter
    {
        defaulttype::BaseMatrix* m;
        unsigned int offsetL, offsetC;
    public:
        BaseMatrixWriter(defaulttype::BaseMatrix* m, unsigned int offsetL, unsigned int offsetC) : m(m), offsetL(offsetL), offsetC(offsetC) {}
        void add(unsigned int bi, unsigned int bj, const MatBloc& b)
        {
            unsigned int i0 = offsetL + bi*NL;
            unsigned int j0 = offsetC + bj*NC;
            for (unsigned int i=0; i<NL; ++i)
                for (unsigned int j=0; j<NC; ++j)
                    m->add(i0+i,j0+j,b[i][j]);
        }
    };

    class BlocBaseMatrixWriter
    {
        defaulttype::BaseMatrix* m;
        unsigned int boffsetL, boffsetC;
    public:
        BlocBaseMatrixWriter(defaulttype::BaseMatrix* m, unsigned int boffsetL, unsigned int boffsetC) : m(m), boffsetL(boffsetL), boffsetC(boffsetC) {}
        void add(unsigned int bi, unsigned int bj, const MatBloc& b)
        {
            unsigned int i0 = boffsetL + bi;
            unsigned int j0 = boffsetC + bj;
            m->blocAdd(i0,j0,b.ptr());
        }
    };

    template<class MReal>
    class BlocCRSMatrixWriter
    {
        sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<NL,NC,MReal> >* m;
        unsigned int boffsetL, boffsetC;
    public:
        BlocCRSMatrixWriter(sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<NL,NC,MReal> >* m, unsigned int boffsetL, unsigned int boffsetC) : m(m), boffsetL(boffsetL), boffsetC(boffsetC) {}
        void add(unsigned int bi, unsigned int bj, const MatBloc& b)
        {
            unsigned int i0 = boffsetL + bi;
            unsigned int j0 = boffsetC + bj;
            //defaulttype::Mat<NL,NC,MReal> bconv = b;
            *m->wbloc(i0,j0,true) += b;
        }
    };

    template<class MReal>
    class CRSMatrixWriter
    {
        sofa::component::linearsolver::CompressedRowSparseMatrix<MReal>* m;
        unsigned int offsetL, offsetC;
    public:
        CRSMatrixWriter(sofa::component::linearsolver::CompressedRowSparseMatrix<MReal>* m, unsigned int offsetL, unsigned int offsetC) : m(m), offsetL(offsetL), offsetC(offsetC) {}
        void add(unsigned int bi, unsigned int bj, const MatBloc& b)
        {
            unsigned int i0 = offsetL + bi*NL;
            unsigned int j0 = offsetC + bj*NC;
            for (unsigned int i=0; i<NL; ++i)
                for (unsigned int j=0; j<NC; ++j)
                    *m->wbloc(i0+i,j0+j,true) += (MReal)b[i][j];
        }
    };


    template<class Dispatcher>
    void apply(Dispatcher& dispatch, sofa::defaulttype::BaseMatrix *m, unsigned int offsetL, unsigned int offsetC)
    {
        if ((offsetL % NL) == 0 && (offsetC % NC) == 0 && m->getBlockRows() == NL && m->getBlockCols() == NC)
        {
            unsigned int boffsetL = offsetL / NL;
            unsigned int boffsetC = offsetC / NC;
            if (sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<NL,NC,double> > * mat = dynamic_cast<sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<NL,NC,double> > * >(m))
            {
                dispatch(BlocCRSMatrixWriter<double>(mat, boffsetL, boffsetC));
            }
            else if (sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<NL,NC,float> > * mat = dynamic_cast<sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<NL,NC,float> > * >(m))
            {
                dispatch(BlocCRSMatrixWriter<float>(mat, boffsetL, boffsetC));
            }
            else
            {
                dispatch(BlocBaseMatrixWriter(m, boffsetL, boffsetC));
            }
        }
        else
        {
            if (sofa::component::linearsolver::CompressedRowSparseMatrix<double> * mat = dynamic_cast<sofa::component::linearsolver::CompressedRowSparseMatrix<double> * >(m))
            {
                dispatch(CRSMatrixWriter<double>(mat, offsetL, offsetC));
            }
            else if (sofa::component::linearsolver::CompressedRowSparseMatrix<float> * mat = dynamic_cast<sofa::component::linearsolver::CompressedRowSparseMatrix<float> * >(m))
            {
                dispatch(CRSMatrixWriter<float>(mat, offsetL, offsetC));
            }
            else
            {
                dispatch(BaseMatrixWriter(m, offsetL, offsetC));
            }
        }
    }


    template<class FF>
    struct DispatcherForceField_addKToMatrix
    {
        FF* main;
        const sofa::core::MechanicalParams* mparams;
        DispatcherForceField_addKToMatrix(FF* main, const sofa::core::MechanicalParams* mparams) : main(main), mparams(mparams) {}
        template <class MatrixWriter>
        void operator()(const MatrixWriter& m)
        {
            main->addKToMatrixT(mparams, m);
        }
    };

    template<class FF>
    void addKToMatrix(FF* main, const sofa::core::MechanicalParams* mparams, sofa::core::behavior::MultiMatrixAccessor::MatrixRef r)
    {
        if (!r) return;
        DispatcherForceField_addKToMatrix<FF> dispatch(main, mparams);
        apply(dispatch, r.matrix, r.offset, r.offset);
    }
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_LINEARSOLVER_BLOCMATRIXWRITER_H
