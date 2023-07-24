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
#include <sofa/core/behavior/DefaultMultiMatrixAccessor.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>

using sofa::core::behavior::BaseMechanicalState;

/// This line registers the DefaultMultiMatrixAccessor to the messaging system
/// allowing to use msg_info() instead of msg_info("DefaultMultiMatrixAccessor")
MSG_REGISTER_CLASS(sofa::core::behavior::DefaultMultiMatrixAccessor, "DefaultMultiMatrixAccessor")

namespace sofa::core::behavior
{


DefaultMultiMatrixAccessor::DefaultMultiMatrixAccessor()
    : globalMatrix(nullptr)
    , globalDim(0)
{
}


DefaultMultiMatrixAccessor::~DefaultMultiMatrixAccessor()
{
    this->clear();
}

void DefaultMultiMatrixAccessor::clear()
{
    globalDim = 0;
    for (auto it = realStateOffsets.begin(), itend = realStateOffsets.end(); it != itend; ++it)
        it->second = -1;

    for (std::map< const sofa::core::behavior::BaseMechanicalState*, linearalgebra::BaseMatrix* >::iterator it = mappedMatrices.begin(), itend = mappedMatrices.end(); it != itend; ++it)
        if (it->second != nullptr) delete it->second;
    mappedMatrices.clear();
    diagonalStiffnessBloc.clear();

    for (std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.begin(), itend = interactionStiffnessBloc.end(); it != itend; ++it)
        if (it->second.matrix != nullptr && it->second.matrix != globalMatrix) delete it->second.matrix;

    interactionStiffnessBloc.clear();
    mappingList.clear();

}


void DefaultMultiMatrixAccessor::setGlobalMatrix(linearalgebra::BaseMatrix* matrix)
{
    this->globalMatrix = matrix;
}

void DefaultMultiMatrixAccessor::addMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate)
{
    const auto dim = mstate->getMatrixSize();
    realStateOffsets[mstate] = globalDim;
    globalDim += dim;

    if(m_doPrintInfo)/////////////////////////////////////////////////////////
    {
        msg_info() << "Adding '" << mstate->getPathName()
                << "' in the global matrix ["<<dim<<"."<<dim
                <<"] at offset (" << realStateOffsets[mstate] <<","<< realStateOffsets[mstate]<<")";
    }
}


void DefaultMultiMatrixAccessor::addMechanicalMapping(sofa::core::BaseMapping* mapping)
{
    const sofa::linearalgebra::BaseMatrix* jmatrix = nullptr;
    if (mapping->isMechanical() && mapping->areMatricesMapped())
        jmatrix = mapping->getJ();

    if (jmatrix)
    {

        const BaseMechanicalState* mappedState  = const_cast<const BaseMechanicalState*>(mapping->getMechTo()[0]);
        linearalgebra::BaseMatrix* mappedstiffness;
        mappedstiffness = mapping->createMappedMatrix(mappedState,mappedState,&DefaultMultiMatrixAccessor::createMatrix);
        mappedMatrices[mappedState]=mappedstiffness;

        mappingList.push_back(mapping);

        if(m_doPrintInfo)/////////////////////////////////////////////////////////
        {
            msg_info() << "Adding validated MechanicalMapping '" << mapping->getPathName()
                       << "' with J["<< jmatrix->rowSize()<<"."<<jmatrix->colSize()<<"]" ;
        }
    }
}


void DefaultMultiMatrixAccessor::addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* /*mstate*/)
{
    // do not add the mapped mechanical state here because
    // a mapped mechanical state is added if and only if it has its own stiffness matrix and its mapping must satistify several conditions
    // so if and only if there are a forcefield or other component call getMatrix(mappedstate)
}

void DefaultMultiMatrixAccessor::setupMatrices()
{
    auto it = realStateOffsets.begin(), itend = realStateOffsets.end();
    while (it != itend)
    {
        if (globalMatrix)
        {
            MatrixRef& r = diagonalStiffnessBloc[it->first];
            r.matrix = globalMatrix;
            r.offset = it->second;
        }
        ++it;
    }

    if(m_doPrintInfo)
    {
        msg_info() << "Setting up the Global Matrix [" << globalDim << "." << globalDim << "] for " << realStateOffsets.size() << " real mechanical state(s)." ;
    }
}

DefaultMultiMatrixAccessor::Index DefaultMultiMatrixAccessor::getGlobalDimension() const
{
    return globalDim;
}

int DefaultMultiMatrixAccessor::getGlobalOffset(const sofa::core::behavior::BaseMechanicalState* mstate) const
{
    const auto it = realStateOffsets.find(mstate);
    if (it != realStateOffsets.end())
        return it->second;
    return -1;
}

DefaultMultiMatrixAccessor::MatrixRef DefaultMultiMatrixAccessor::getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const
{
    MatrixRef r;
    const auto itRealState = realStateOffsets.find(mstate);

    if (itRealState != realStateOffsets.end()) //case where mechanical state is a non mapped state
    {
        if (globalMatrix)
        {
            r.matrix = globalMatrix;
            r.offset = itRealState->second;
        }
    }
    else //case where mechanical state is a mapped state
    {
        const std::map< const sofa::core::behavior::BaseMechanicalState*, linearalgebra::BaseMatrix*>::iterator itmapped = mappedMatrices.find(mstate);
        if (itmapped != mappedMatrices.end()) // this mapped state and its matrix has been already added and created
        {
            r.matrix = itmapped->second;
            r.offset = 0;
        }
        else // this mapped state and its matrix hasnt been created we creat it and its matrix by "createMatrix"
        {

            linearalgebra::BaseMatrix* m = createMatrixImpl(mstate,mstate, m_doPrintInfo);
            r.matrix = m;
            r.offset = 0;
            //when creating an matrix, it dont have to be added before
            assert(diagonalStiffnessBloc.find(mstate) == diagonalStiffnessBloc.end());
            mappedMatrices[mstate]=r.matrix;
        }
    }

    diagonalStiffnessBloc[mstate] = r;

    if(m_doPrintInfo)
    {
        if (r.matrix != nullptr)
        {
            msg_info() << "Giving Stiffness Matrix [" << r.matrix->rowSize() << "." << r.matrix->colSize() << "] for state '" << mstate->getPathName()
                    << "' at offset (" << r.offset  <<","<< r.offset <<")" ;
        }
        else
            msg_warning() << "nullptr matrix found for state " << mstate->getName() ;
    }
    return r;
}

DefaultMultiMatrixAccessor::InteractionMatrixRef DefaultMultiMatrixAccessor::getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const
{
    InteractionMatrixRef r2;
    if (mstate1 == mstate2)// case where state1 == state2, interaction matrix is on the diagonal stiffness bloc
    {
        const MatrixRef r = diagonalStiffnessBloc.find(mstate1)->second;
        r2.matrix = r.matrix;
        r2.offRow = r.offset;
        r2.offCol = r.offset;

        if(m_doPrintInfo)///////////////////////////////////////////
        {
            if (r2.matrix != nullptr)
            {
                msg_info() << "Giving Interaction Stiffness Matrix ["
                        << r2.matrix->rowSize() << "." << r2.matrix->colSize()
                        <<"] at offset ("<<r2.offRow <<","<< r2.offCol<<") for self-interaction : "
                        <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]" ;
            }else
            {
                msg_warning() << "Giving nullptr matrix for self-interaction "<<
                        mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]" ;
            }
        }
    }
    else// case where state1 # state2
    {
        const std::pair<const BaseMechanicalState*,const BaseMechanicalState*> pairMS = std::make_pair(mstate1,mstate2);

        const std::map< std::pair<const BaseMechanicalState*,const BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.find(pairMS);
        if (it != interactionStiffnessBloc.end())// the interaction is already added
        {
            if(it->second.matrix != nullptr)
            {
                r2 = it->second;
            }

            if(m_doPrintInfo)///////////////////////////////////////////
            {
                if(r2.matrix != nullptr)
                {
                    msg_info() << "Giving Interaction Stiffness Matrix ["
                            << r2.matrix->rowSize() << "." << r2.matrix->colSize()
                            <<"] at offset ("<<r2.offRow <<","<< r2.offCol<<")  for interaction : "
                            <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]" ;
                }
                else
                {
                    msg_warning() << "Giving nullptr matrix  for interaction "
                            <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]" ;
                }
            }
        }
        else// the interaction is not added, we need to creat it and its matrix
        {
            const auto it1 = realStateOffsets.find(mstate1);
            const auto it2 = realStateOffsets.find(mstate2);

            if(it1 != realStateOffsets.end() && it2 != realStateOffsets.end())// case where all of two ms are real DOF (non-mapped)
            {
                if (globalMatrix)
                {
                    r2.matrix = globalMatrix;
                    r2.offRow = it1->second;
                    r2.offCol = it2->second;

                    if(m_doPrintInfo)/////////////////////////////////////////////////////////
                    {
                        if (r2.matrix != nullptr)
                        {
                            msg_info() <<  "Giving Interaction Stiffness Matrix ["
                                    << r2.matrix->rowSize() << "." << r2.matrix->colSize()
                                    <<"] at offset ("<<r2.offRow <<","<< r2.offCol<<")  for interaction : "
                                    <<mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]" ;
                        }else{
                            msg_warning() << "Giving nullptr matrix  for interaction "
                                    << " for interaction : "
                                    << mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]" ;
                        }
                    }
                }
            }
            else //case where at least one ms is a mapped
            {
                linearalgebra::BaseMatrix* m = createMatrixImpl(mstate1,mstate2,m_doPrintInfo);
                r2.matrix = m;
                r2.offRow = 0;
                r2.offCol = 0;
                //when creating an matrix, it dont have to be added before
                assert(interactionStiffnessBloc.find(pairMS) == interactionStiffnessBloc.end());

                if(m_doPrintInfo)/////////////////////////////////////////////////////////
                {
                    if (r2.matrix != nullptr)
                    {
                        msg_info() <<   "Giving Interaction Stiffness Matrix ["
                                << r2.matrix->rowSize() << "." << r2.matrix->colSize()
                                << "] at offset ("<<r2.offRow <<","<< r2.offCol<<")  for interaction : "
                                << mstate1->getName()<< "[" <<mstate1->getMatrixSize()<< "] --- " <<mstate2->getName()<<"[" <<mstate2->getMatrixSize()<<"]" ;
                    }
                    else
                    {
                        msg_info() << "Giving nullptr matrix  for interaction "
                                << " for interaction : "
                                << mstate1->getName()<<"["<<mstate1->getMatrixSize()<<"] --- "<<mstate2->getName()<<"["<<mstate2->getMatrixSize()<<"]" ;
                    }
                }
            }

            interactionStiffnessBloc[pairMS]=r2;
        }// end of the interaction is not added, we need to creat it and its matrix

    }//end of case where state1 # state2


    if(m_doPrintInfo && r2.matrix == nullptr)
    {
        msg_warning() << "nullptr matrix found for interaction " << mstate1->getName()<<" --- "<<mstate2->getName() ;
    }

    return r2;
}

void DefaultMultiMatrixAccessor::computeGlobalMatrix()
{
    const int lastMappingId = (int)mappingList.size() - 1;
    for(int id=lastMappingId; id>=0; --id)
    {
        sofa::core::BaseMapping* m_mapping = mappingList[id];
        const BaseMechanicalState* instate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechFrom()[0]);
        const BaseMechanicalState* outstate  = const_cast<const BaseMechanicalState*>(m_mapping->getMechTo()[0]);

        const linearalgebra::BaseMatrix* matrixJ = m_mapping->getJ();
        const auto nbR_J = matrixJ->rowSize();
        const auto nbC_J = matrixJ->colSize();

        if(m_doPrintInfo)/////////////////////////////////////////////////////////
        {
            msg_info() << "[CONTRIBUTION] of " << id << "-th mapping named : " << m_mapping->getName()
                    << " with fromModel : "<< instate->getName()
                    << " and toModel : "   << outstate->getName() <<" -- with matrix J["<<nbR_J <<"."<<nbC_J <<"]" ;
        }


        //for toModel -----------------------------------------------------------
        if(mappedMatrices.find(outstate) == mappedMatrices.end())
        {
            if(m_doPrintInfo)
            {
                msg_info() << "	[Propa.Stiff] WARNING toModel : "<< outstate->getName()<< " dont have stiffness matrix" ;
            }
        }

        if(diagonalStiffnessBloc.find(outstate) != diagonalStiffnessBloc.end())
        {
            //=================================
            //           K11 += Jt * K22 * J
            //=================================
            const MatrixRef K1 = this->getMatrix(instate);
            const MatrixRef K2 = this->getMatrix(outstate);

            const auto offset1  = K1.offset;
            const auto offset2  = K2.offset;
            const Index sizeK1 = Index(K1.matrix->rowSize() - offset1);
            const Index sizeK2 = Index(K2.matrix->rowSize() - offset2);

            if(m_doPrintInfo)/////////////////////////////////////////////////////////
            {
                msg_info() << "	[Propa.Stiff] propagating stiffness of : "<< outstate->getName()<< " to stifness "<<instate->getName()<< msgendl;
                msg_info() <<"	                    **multiplication of "
                        <<" K1["<<sizeK1<<"."<<sizeK1<< "]("<< offset1<<","<<offset1 <<  ")   =  "
                        <<" Jt["<<nbC_J<<"."<<nbR_J<< "] * "
                        <<" K2["<<sizeK2<<"."<<sizeK2<<"]("<< offset2<<","<<offset2 <<  ") * "
                        <<" J["<<nbR_J<<"."<<nbC_J<< "]"
                        << msgendl;
            }

            // Matrix multiplication  K11 += Jt * K22 * J
            for(Index i1 =0; i1 < sizeK1 ; ++i1)
            {
                for(Index j1 =0 ; j1 < sizeK1 ; ++j1)
                {
                    double Jt_K2_J_i1j1 = 0;

                    for(Index i2 =0 ; i2 < sizeK2 ; ++i2)
                    {
                        for(Index j2 =0 ; j2 < sizeK2 ; ++j2)
                        {
                            const double K2_i2j2 = (double) K2.matrix->element(offset2 + i2, offset2 + j2);
                            for(Index k2=0 ; k2 < sizeK2 ; ++k2)
                            {
                                const double Jt_i1k2 = (double) matrixJ->element( i1 , k2 ) ;
                                const double  J_k2j1 = (double) matrixJ->element( k2 , j1 ) ;

                                Jt_K2_J_i1j1 += Jt_i1k2 * K2_i2j2  * J_k2j1;
                            }
                        }
                    }
                    K1.matrix->add(offset1 + i1 , offset1 + j1 , Jt_K2_J_i1j1);
                }
            }
            // Matrix multiplication  K11 += Jt * K22 * J
        }

        std::vector<std::pair<const BaseMechanicalState*, const BaseMechanicalState*> > interactionList;
        for (std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef >::iterator it = interactionStiffnessBloc.begin(), itend = interactionStiffnessBloc.end(); it != itend; ++it)
        {
            if(it->first.first == outstate || it->first.second == outstate )
            {
                interactionList.push_back(it->first);
            }
        }


        const size_t nbInteraction = interactionList.size();
        for(size_t i=0; i< nbInteraction; i++)
        {

            if(m_doPrintInfo)/////////////////////////////////////////////////////////
            {
                msg_info() << "	[Propa.Interac.Stiff] detected interaction between toModel : "<<interactionList[i].first->getName()
                        << " and : " <<interactionList[i].second->getName() ;
            }
            //                   |       |
            //                 MS1     MS2
            //                  /      /
            //                map    inter
            //                   \   /
            //                   MS3/
            //
            //           K_11 += Jt * K_33 * J
            //           I_12 += Jt * I_32
            //           I_21 +=      I_23 * J
            //

            if(interactionList[i].first == outstate)
            {
                const InteractionMatrixRef I_32 = this->getMatrix( outstate , interactionList[i].second);
                const InteractionMatrixRef I_12 = this->getMatrix( instate  , interactionList[i].second);
                //===========================
                //          I_12 += Jt * I_32
                //===========================
                const Index offR_I_12  = I_12.offRow;                       //      row offset of I12 matrix
                const Index offC_I_12  = I_12.offCol;                       //    colum offset of I12 matrix
                const Index nbR_I_12   = I_12.matrix->rowSize() - offR_I_12;//number of rows   of I12 matrix
                const Index nbC_I_12   = I_12.matrix->colSize() - offC_I_12;//number of colums of I12 matrix

                const Index offR_I_32  = I_32.offRow;                     //      row offset of I32 matrix
                const Index offC_I_32  = I_32.offCol;                     //    colum offset of I32 matrix
                const Index nbR_I_32 = I_32.matrix->rowSize() - offR_I_32;//number of rows   of I32 matrix
                const Index nbC_I_32 = I_32.matrix->colSize() - offC_I_32;//number of colums of I32 matrix


                if(m_doPrintInfo)/////////////////////////////////////////////////////////
                {
                    msg_info() << "	[Propa.Interac.Stiff] propagating interaction "
                            <<outstate->getName() << "--" <<interactionList[i].second->getName()
                            <<"  to : "
                            <<instate->getName() << "--" <<interactionList[i].second->getName() ;

                    msg_info() <<"	                    **multiplication of "
                            <<" I12["<<nbR_I_12<<"."<<nbC_I_12<< "]("<< offR_I_12<<","<<offC_I_12 <<  ")  =  "
                            <<" Jt["<<nbC_J<<"."<<nbR_J<< "]  *  "
                            <<" I32["<<nbR_I_32<<"."<<nbC_I_32<<"]("<< offR_I_32<<","<<offC_I_32 <<  ")" ;
                }

                // Matrix multiplication   I_12 += Jt * I_32
                for(Index _i = 0; _i < nbR_I_12 ; _i++)
                {
                    for(Index _j = 0; _j < nbC_I_12 ; _j++)
                    {
                        double Jt_I32_ij = 0;
                        for(Index _k = 0; _k < nbR_I_32 ; _k++)
                        {
                            const double Jt_ik    = (double) matrixJ->element( _k, _i ) ;
                            const double  I_32_kj = (double) I_32.matrix->element( offR_I_32 + _k, offC_I_32+_j) ;

                            Jt_I32_ij += Jt_ik  *  I_32_kj;
                        }
                        I_12.matrix->add(offR_I_12 + _i , offC_I_12 +  _j , Jt_I32_ij);
                    }
                }// Matrix multiplication   I_12 += Jt * I_32
            }

            if(interactionList[i].second == outstate)
            {
                const InteractionMatrixRef I_21 = this->getMatrix(interactionList[i].first,instate);
                const InteractionMatrixRef I_23 = this->getMatrix(interactionList[i].first,outstate);
                //=========================================
                //          I_21 +=      I_23 * J
                //=========================================
                const Index offR_I_21  = I_21.offRow;
                const Index offC_I_21  = I_21.offCol;
                const Index nbR_I_21 = I_21.matrix->rowSize() - offR_I_21;
                const Index nbC_I_21 = I_21.matrix->colSize() - offC_I_21;

                const Index offR_I_23  = I_23.offRow;
                const Index offC_I_23  = I_23.offCol;
                const Index nbR_I_23   = I_23.matrix->rowSize() - offR_I_23;
                const Index nbC_I_23   = I_23.matrix->colSize() - offC_I_23;

                if(m_doPrintInfo)/////////////////////////////////////////////////////////
                {
                    msg_info() << "	[Propa.Interac.Stiff] propagating interaction "
                            <<interactionList[i].first->getName()<< "--" <<outstate->getName()
                            <<" to : "<<interactionList[i].first->getName()<< "--" <<instate->getName() ;

                    msg_info() <<"	                    **multiplication of "
                            <<" I_21["<<nbR_I_21<<"."<<nbC_I_21<<"]("<< offR_I_21<<","<<offC_I_21 <<  ")  =  "
                            <<" I23["<<nbR_I_23<<"."<<nbC_I_23<< "]("<< offR_I_23<<","<<offC_I_23 <<  ")  *  "
                            <<" J["<<nbR_J<<"."<<nbC_J<<"]" ;
                }


                // Matrix multiplication  I_21 +=  I_23 * J
                for(Index _i = 0; _i < nbR_I_21 ; _i++)
                {
                    for(Index _j = 0; _j < nbC_I_21 ; _j++)
                    {
                        double I23_J_ij = 0;
                        for(Index _k = 0; _k < nbC_I_23 ; _k++)
                        {
                            const double I_23_ik = (double) I_23.matrix->element( offR_I_23 + _i, offC_I_23+_k) ;
                            const double J_kj    = (double) matrixJ->element( _k, _j ) ;

                            I23_J_ij += I_23_ik  * J_kj ;
                        }
                        I_21.matrix->add(offR_I_21 + _i , offC_I_21 + _j , I23_J_ij);
                    }
                }// Matrix multiplication  I_21 +=  I_23 * J

            }

            //after propagating the interaction, we remove the older interaction
            interactionStiffnessBloc.erase( interactionStiffnessBloc.find(interactionList[i]) );
            if(m_doPrintInfo)
            {
                msg_info() << "	--[Propa.Interac.Stiff] remove interaction of : "<<interactionList[i].first->getName()
                        << " and : " << interactionList[i].second->getName() ;
            }

        }//end of interaction loop

    }//end of mapping loop
}

linearalgebra::BaseMatrix* DefaultMultiMatrixAccessor::createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2)
{
    return createMatrixImpl(mstate1, mstate2, false) ;
}

linearalgebra::BaseMatrix* DefaultMultiMatrixAccessor::createMatrixImpl(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2, bool doPrintInfo)
{
    linearalgebra::CompressedRowSparseMatrix<SReal>* m = new linearalgebra::CompressedRowSparseMatrix<SReal>;
    if(mstate1 == mstate2)
    {
        m->resize( mstate1->getMatrixSize(),mstate1->getMatrixSize());

        if(doPrintInfo)/////////////////////////////////////////////////////////
        {
            msg_info("DefaultMultiMatrixAccessor") << "			++ Creating matrix["<< m->rowSize() <<"."<< m->colSize() <<"]   for mapped state " << mstate1->getName() << "[" << mstate1->getMatrixSize()<<"]" ;
        }
    }
    else
    {
        m->resize( mstate1->getMatrixSize(),mstate2->getMatrixSize() );

        if(doPrintInfo)/////////////////////////////////////////////////////////
        {
            msg_info("DefaultMultiMatrixAccessor") << "			++ Creating interraction matrix["<< m->rowSize() <<"x"<< m->colSize()
                      << "] for interaction " << mstate1->getName() << "[" << mstate1->getMatrixSize()
                      << "] --- "             << mstate2->getName() << "[" << mstate2->getMatrixSize()<<"]" ;
        }
    }

    return m;
}

} // namespace sofa::core::behavior
