/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CUBATUREBaseMaterialFORCEFIELD_H
#define SOFA_CUBATUREBaseMaterialFORCEFIELD_H

#include "../initFlexible.h"
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/MechanicalState.h>

#include "../material/BaseMaterial.h"
#include "../quadrature/BaseGaussPointSampler.h"

#include <SofaEigen2Solver/EigenSparseMatrix.h>

using std::cerr;
using std::endl;

namespace sofa
{
namespace component
{
namespace forcefield
{

using helper::vector;

/** Abstract interface to allow for resizing
*/
class SOFA_Flexible_API CubatureBaseMaterialForceField : public virtual core::objectmodel::BaseObject
{
public:
    virtual void resize()=0;
    virtual SReal getPotentialEnergy( const unsigned int index ) const=0;
};


/** Abstract forcefield using MaterialBlocks or sparse eigen matrix
*/

template <class MaterialBlockType>
class CubatureBaseMaterialForceFieldT : public core::behavior::ForceField<typename MaterialBlockType::T>, public CubatureBaseMaterialForceField
{
public:
    typedef core::behavior::ForceField<typename MaterialBlockType::T> Inherit;
    SOFA_ABSTRACT_CLASS2(SOFA_TEMPLATE(CubatureBaseMaterialForceFieldT,MaterialBlockType),SOFA_TEMPLATE(core::behavior::ForceField,typename MaterialBlockType::T),CubatureBaseMaterialForceField);

    /** @name  Input types    */
    //@{
    typedef typename MaterialBlockType::T DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
    typedef core::behavior::MechanicalState<DataTypes> mstateType;
    typedef Eigen::VectorXd Vector;

    typedef Eigen::MatrixXd matriceDense ;
    typedef core::objectmodel::MultiLink<core::BaseMapping, core::BaseMapping, BaseLink::FLAG_DOUBLELINK|BaseLink::FLAG_STRONGLINK> Links;
    typedef Eigen::Triplet<double> triplet;
    typedef Eigen::SparseMatrix<SReal, Eigen::ColMajor> cmat;

    //@}

    /** @name  material types    */
    //@{
    typedef MaterialBlockType BlockType;  ///< Material block object
    typedef vector<BlockType >  SparseMatrix;

    typedef typename BlockType::MatBlock  MatBlock;  ///< Material block matrix
    typedef linearsolver::EigenSparseMatrix<DataTypes,DataTypes>    SparseMatrixEigen;
    //@}

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////// LES FONCTIONS SUIVANTES SERONT UTILES POUR L'ALGO NN-HTP ////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    void initW(Vector &v , size_t subSet, SReal val, bool isAlea)
    {
        for(size_t i = 0 ; i < v.size() ; ++i) // boucle sur les lignes de U
        {
            v(i)=0.;
        }

        if(isAlea)
        {
            size_t n=0;
            while(n<subSet)
            {
                size_t index = (size_t)(rand() % v.size());
                v(index) = val;
                ++n;
            }
        }
        else
        {
            size_t gap = v.size()/subSet;
            for(size_t i=0 ; i<v.size() ; ++i) // boucle sur les lignes de U
            {
                if(i%gap==0) v(i) = val;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    std::vector<SReal> ReducedSupport(Vector v)
    {
        std::vector<SReal> supp;
        for(size_t i = 0 ; i < v.size() ; ++i)
        {
            if(v[i] != 0)
            {
                supp.push_back(i);
            }
        }
        return supp ;
    }

    ////////////////////////////////////////////////////////////////////////
    ///////////// Fonction qui permet de calculer H_s(vecteur) /////////////
    // A présent, implémentons la fonction H_s(vecteur &v, entier s)
    // qui garde les s plus grandes coordonnées du vecteur v
    // et met les autres à zéro.
    void Hs(Vector &v , size_t s)
    {
        Vector vbis = v ;
        v=0.*v;
        size_t indexMax, indexMin, n=0;
        SReal minVal=vbis.minCoeff(&indexMin);
        while(n<s)
        {
            SReal maxVal=vbis.maxCoeff(&indexMax) ;
            vbis(indexMax) = minVal; // comme cela, cette valeur ne sera plus prise en compte comme valeur minimal

            if(maxVal>=0)
            {
                v(indexMax) = maxVal;
                ++n;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////
    /////// Fonction qui permet de calculer le support d'un vecteur ////////
    // fonction qui renvoit le support d'un vecteur,
    // sous la forme d'un vecteur dont l'entrée vaut 0
    // si l'entrée du vecteur dont on calcul le support vaut 0
    // et 1 si l'entrée du vecteur dont on calcul le support est non-nulle.
    Eigen::VectorXi support(Vector v)
    {
        Eigen::VectorXi supp(v.size()) ;
        for(size_t i = 0 ; i < v.size() ; ++i)
        {
            if(v[i] != 0)
            {
                supp(i) = 1 ;
            }
            else
            {
                supp(i) = 0;
            }
        }
        return supp;
    }

    ////////////////////////////////////////////////////////////////////////
    ///// Fonction qui permet de calculer la reunion de deux supports //////
    Eigen::VectorXi supportUnion(Vector u , Vector v)
    {
        Eigen::VectorXi Usupp = support(u) , Vsupp = support(v) , UVsupp(u.size()) ;

        for(size_t i = 0 ; i < u.size() ; ++i)
        {
            if(u[i] != 0 || v[i] != 0)
            {
                UVsupp(i) = 1 ;
            }
            else
            {
                UVsupp(i) = 0;
            }
        }
        return UVsupp;
    }

    ////////////////////////////////////////////////////////////////////////
    ///////// Fonction qui permet de calculer le grad_set( f(w) ) //////////
    // pour cela on récupère le grad_f_w de l'étape 6
    // et on met à zero toutes les entrées du vecteur grad_f
    // qui correspondent aux entrées nulles du vecteur set.
    void grad_set_f_w(Vector& grad_f_w, Eigen::VectorXi set)
    {
        for(size_t i = 0 ; i < grad_f_w.size() ; ++i)
        {
            if(set[i] == 0)
            {
                grad_f_w[i] = 0 ;
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    ///////// Fonction qui permet de calculer l'ensemble I\supp(w^i) /////////
    // La fonction suivante prend en argument le supp(w^i) et renvoie l'ensemble I\supp(w^i)
    // c'est à dire {1,...,m}\supp(w^i)
    Eigen::VectorXi Kset(Eigen::VectorXi supp_w, size_t m)
    {
        Eigen::VectorXi I(m) ;
        for(size_t i=0; i<I.size(); ++i) I(i)=1;
        if(I.size() == supp_w.size()) return I-supp_w ;
        else cerr << "dans la fonction Kset, problèmes de dimensions..." << endl;
    }

    //////////////////////////////////////////////////////////////////////////
    /////////////////// Mettre énergie potentielle dans fichier //////////////
    void energieToFile(Eigen::MatrixXd U, Data<std::string>& file)
    {
        if(file.getValue().size() > 0)
        {
            std::ofstream s(file.getValue().c_str() );

            s << U.rows() << '\n' << U.cols() << '\n' ;
            for(size_t i = 0 ; i < U.rows() ; ++i) // boucle sur les lignes de U
            {
                for(size_t j = 0 ; j < U.cols() ; ++j)
                {
                    s << U(i , j) << '\n' ;
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////// FIN FONCTIONS UTILES //////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    virtual void resize()
    {
        if(!(this->mstate)) return;

        // init material
        material.resize( this->mstate->getSize() );

        if(this->f_printLog.getValue()) std::cout<<SOFA_CLASS_METHOD<<" "<<material.size()<<std::endl;

        // retrieve volume integrals
        engine::BaseGaussPointSampler* sampler=NULL;
        this->getContext()->get(sampler,core::objectmodel::BaseContext::SearchUp);
        if( !sampler ) { serr<<"Gauss point sampler not found -> use unit volumes"<< sendl; for(unsigned int i=0; i<material.size(); i++) material[i].volume=NULL; }
        else for(unsigned int i=0; i<material.size(); i++) material[i].volume=&sampler->f_volume.getValue()[i];

        reinit();
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    /** @name forceField functions */
    //@{
    virtual void init()
    {
        nbStep = 0;
        nbe = nbElements.getValue();
        nbPI = 8;
        totalPI = nbe*nbPI;

        stressSize = 6;
        r = nbTrainingPositions.getValue();
        b = Vector(totalPI*stressSize*r);
        A = cmat(r*totalPI*stressSize,totalPI);
        s = (size_t) (totalPI*integrationPointsPercent.getValue());
        w = Vector(totalPI) ;

        theForce = Eigen::MatrixXd(totalPI*stressSize,r);

        initialValue = 1/integrationPointsPercent.getValue();

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////// Etape 2 : initialisation de w: cf init() //////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if(!optimizedCubature.getValue()) initW(w , s, initialValue, false);
        else initW(w , s, 1., true);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////// calcul du support de w //////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        reducedSupportW = ReducedSupport(w);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////// Initialisation de la matrice de sélection /////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        selectionMatrix = cmat(9*reducedSupportW.size(),9*totalPI);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////// Ouverture du fichier pour enregistrer //////////////////////////////////
        /////////////////////////////// l'énergie potentielle, les positions et les vitesses //////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if(potentialEnergyFile.getValue().size() > 0)
        {
            potentialEnergyStream.open(potentialEnergyFile.getValue().c_str());
        }

        if(!(this->mstate))
        {
            this->mstate = dynamic_cast<mstateType*>(this->getContext()->getMechanicalState());
            if(!(this->mstate)) { serr<<"state not found"<< sendl; return; }
        }

        resize();

        Inherit::init();
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    virtual void reinit()
    {

        addForce(NULL, *this->mstate->write(core::VecDerivId::force()), *this->mstate->read(core::ConstVecCoordId::position()), *this->mstate->read(core::ConstVecDerivId::velocity()));

        // reinit matrices
        if(this->assemble.getValue() && BlockType::constantK)
        {
            if( this->isCompliance.getValue() ) updateC();
            else updateK();
            updateB();
        }

        Inherit::reinit();
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    //Pierre-Luc : Implementation in HookeForceField
    virtual void addForce(DataVecDeriv& /*_f*/ , const DataVecCoord& /*_x*/ , const DataVecDeriv& /*_v*/, const vector<SReal> /*_vol*/)
    {
        std::cout << "Do nothing" << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    //////// FONCTION APPELLÉE DANS CubatureHookeForceField ///////
    virtual void addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v)
    {
        if(this->mstate->getSize()!=(int)material.size()) resize();

        VecDeriv&  f = *_f.beginEdit();
        const VecCoord&  x = _x.getValue();
        const VecDeriv&  v = _v.getValue();

        /*
        SReal sum=0;
        size_t sup = 0;
        std::cerr << "w=" << std::endl;
        for(size_t i=0; i<w.size(); ++i)
        {
            sum+=w(i);
            if(w(i) != 0) ++sup;
            //std::cerr << "LIGNE " << i << ": " << w(i) << std::endl;
        }

        std::cerr << "somme des entrées de w= " << sum << " " << "nb entrées non nulles: " << sup  << " " << s << std::endl;

        std::vector<SReal> supp = ReducedSupport(w);
        std::cerr << "support de w =" << std::endl;
        for(size_t i=0; i<supp.size(); ++i)
        {
            std::cerr << i << " " << supp[i] << " " << w(supp[i]) << std::endl;
        }*/
        /*
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////// Cubature switched //////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::string comment="";
        if(switchMode.getValue())
        {
            if(nbStep%2)
            {
                comment="    AVEC cubature: ";
                for(unsigned int i=0; i<reducedSupportW.size(); i++)
                {
                    size_t r = reducedSupportW[i];
                    material[r].addForce(f[r],x[r],v[r]);
                    for(size_t c=0; c<f[r].size(); ++c)
                    {
                        f[r][c] *= w(r);
                    }
                }
            }
            else //nbStep%2
            {
                comment="    SANS cubature: ";
                for(unsigned int i=0; i<material.size(); i++)
                {
                    material[i].addForce(f[i],x[i],v[i]);
                }
            } //nbStep%2
        } // switchMode

        else // !switchMode*/
        if(!switchMode.getValue())
        {
            if(cubature.getValue())
            {
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////////////////////////////// Cubature naive ///////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                if(!optimizedCubature.getValue())
                {
                    for(unsigned int i=0; i<reducedSupportW.size(); i++)
                    {
                        size_t r = reducedSupportW[i];
                        material[r].addForce(f[r],x[r],v[r]);

                        for(size_t c=0; c<f[r].size(); ++c)
                        {
                            f[r][c] *= w(r);
                        }
                    }
                }

                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ////////////////////////////////////////////// Cubature optimisée /////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                else
                { // optimizedCubature
                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    //////////////////////////////////////// PRÉLIMINAIRES À L'ALGO NN-HTP /////////////////////////////////////
                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    // dans l'interface utilisateur de ma scene, je permet à l'utilisateur
                    // de rentrer la valeur du vecteur q=(q_1,...,q_r)
                    // le premier vecteur qu'il rentrera sera q^1
                    // le second, q^2, ...
                    // et ainsi de suite...jusqu'à q(T)

                    // a chaque fois qu'on génère une training position
                    // on enregistre le tenseur des contraintes associé dans la matrice theForce

                    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    //////////////////// ETAPE 1 : Stockage du tenseur des contraintes dans la matrice theForce ///////////////////
                    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    if(nbStep > 1 && nbStep < r+2)
                    {
                        for(unsigned int i=0; i<material.size(); i++)
                        {
                            material[i].addForce(f[i],x[i],v[i]);
                        }

                        //cerr << f << endl;

                        for(size_t pi = 0 ; pi < totalPI ; ++pi)
                        {
                            for(size_t s = 0 ; s < stressSize ; ++s)
                            {
                                theForce(stressSize*pi+s,nbStep-2) = f[pi][s];
                            }
                        }
                    }

                    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    ///////////////////////////// ETAPE 2 : On fabrique la matrice A et le vecteur b //////////////////////////////
                    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    else if(nbStep == r+2)
                    {
                        // calcul de la norme de la force pour chaque mode
                        Vector forceNormForEachMode = theForce.colwise().norm();

                        for(size_t mode=0 ; mode<r ; ++mode)
                        {
                            for(size_t i=0 ; i<totalPI*stressSize ; ++i)
                            {
                                b(mode*totalPI*stressSize+i) = theForce(i,mode);
                                b(mode*totalPI*stressSize+i)/=forceNormForEachMode(mode);
                            }
                        }

                        for(size_t pi=0 ; pi<totalPI ; ++pi) // iteration  sur les colonnes de A
                        {
                            //iteration sur les lignes
                            for(size_t mode = 0 ; mode < r ; ++mode) // iteration  sur les blocks de taille totalPI de chaque colonne
                            {
                                for(size_t q = 0 ; q < stressSize ; ++q) // iteration sur les 6 coordonnées du stress
                                {
                                    SReal c=theForce(stressSize*pi+q,mode)/forceNormForEachMode(mode);
                                    tripletListA.push_back( triplet(mode*totalPI*stressSize+stressSize*pi+q,pi,c ) );
                                }
                            }
                        }
                        A.setFromTriplets(tripletListA.begin(), tripletListA.end());
                        cerr << "etape 2: " << "stressSize= " << stressSize << " totalPI= " << totalPI << " r= " << r << " nbStep= " << nbStep << endl;
                        cerr << "A lines= " << A.rows() << " A cols= " << A.cols() << endl;

                        /*size_t mode = 0  ;

                        for(size_t i=0; i<A.rows(); ++i)
                        {
                            if(i>=mode*totalPI*stressSize && i<(mode+1)*totalPI*stressSize) cerr << i << " " << A.coeff(i,totalPI-1) << endl;
                        }*/

                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        ///////////////////////////////////// FIN PRÉLIMINAIRES À L'ALGO NN-HTP ////////////////////////////////////
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        ///////////////////////////////////////////// DÉBUT ALGO NN-HTP ////////////////////////////////////////////
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

                        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        ///////////////////////////////// Etape 1 : On choisit un seuil pour l'erreur /////////////////////////////////
                        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

                        SReal tol = 0.1 ;

                        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        /////////////////////////////////// Etape 2 : initialisation de w: cf init() //////////////////////////////////
                        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        /////////////////////////// ENTRÉE DANS LE while(squaredNorm(grad_f_w) > tol) //////////////////////////////
                        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        // On considère l'erreur au carré: error(w)² = ||A*w-b||²/T
                        // et nous, on s'interessera à la fonction f(w) = ||A*w-b||²
                        // dont on veut le minimum. Pour cela, on calcul son gradient (w étant un vecteur)
                        // et on cherche le vecteur w qui annule ce gradient.
                        // Grad[f(w)] = 2*A.transpose()*(Aw-b). C'est un vecteur
                        // Calculons ce gradient...
                        cmat AT(r*totalPI*stressSize,totalPI);
                        Eigen::VectorXi supp_w(w.size()), tempSupp_w(w.size()), K(w.size()), S(w.size());
                        Vector tempGrad_f_w(totalPI), minusTempGrad_f_w(totalPI), Agrad_S_f_w(totalPI*stressSize), vec_erreur(stressSize*totalPI);
                        SReal mu, erreur;

                        AT = A.transpose();

                        /*size_t mode = 0;

                        for(size_t i=0; i<A.rows(); ++i)
                        {
                            if(i>=mode*totalPI*stressSize && i<(mode+1)*totalPI*stressSize) cerr << i << " " << AT.coeff(totalPI-1,i) << endl;
                        }*/

                        Vector grad_f_w = 2*AT*(A*w-b) ;
                        cerr << "norme de grad_f_w= " << grad_f_w.norm() << endl;
                        bool consecutiveWareTheSame = false;
                        size_t iter = 0;

                        vec_erreur = A*w-b;

                        cerr << grad_f_w.norm() << endl;
                        //cerr << "erreur= " << vec_erreur.norm()/sqrt(r) << endl;

                        while(grad_f_w.norm() > tol) // || !consecutiveWareTheSame)
                        {
                            ///vec_erreur = A*w-b;
                            cerr << "grad_f_w.norm()= " << grad_f_w.norm() << endl;
                            ///cerr << "erreur= " << vec_erreur.norm()/sqrt(r) << endl;

                            // calcul de supp(w)
                            supp_w = support(w) ;
                            tempSupp_w = supp_w ;

                            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            //////////////////////////////////////////////// ETAPE 2 : calcul de S_i //////////////////////////////////////
                            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            // calcul  de K = I\supp(w)
                            K = Kset(supp_w, totalPI) ;

                            // calcul de grad_K_f_w
                            grad_set_f_w(tempGrad_f_w , K ) ;

                            //Calcul de H_s{ -grad_K_f_w }
                            minusTempGrad_f_w = -tempGrad_f_w;
                            Hs(minusTempGrad_f_w, s);

                            // calcul du support S = supp(w) U supp[ H_s{ -grad_K_f_w } ]
                            S = supportUnion(w , minusTempGrad_f_w) ;

                            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            ///////////////////////////////////////////// ETAPE 3 : calcul de µ ///////////////////////////////////////////
                            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            // calcul de grad_S_f_w
                            ///grad_set_f_w(grad_f_w , S ) ;

                            // calcul de A*grad_S_f_w
                            Agrad_S_f_w = A*grad_f_w;
                            //cerr << "A*grad_f_w= " << Agrad_S_f_w.norm() << endl;

                            // calcul de µ = calcul de || grad_S_f_w ||² / calcul de || A*grad_S_f_w ||²
                            mu = grad_f_w.norm() / Agrad_S_f_w.norm() ;
                            mu = pow(mu,2);
                            //cerr << "mu= "<< mu << endl;

                            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            ////////////////////// ETAPE 4 : actuallisation de w par la procédure H_s(w-µ*grad_S_f(w)) ////////////////////
                            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

                            w-=mu*grad_f_w;
                            Hs(w, s);
/*
                            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            //////////// ETAPE 5 : si le support courant est le même que le support précédent, alors on arrête ////////////
                            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
                            size_t i=0;
                            bool isIdentical = true;
                            while(i<w.size() && isIdentical)
                            {
                                if(support(w)(i)!=tempSupp_w(i))
                                {
                                    isIdentical = false ;
                                }
                                ++i;
                            }

                            if(isIdentical)
                            {
                                consecutiveWareTheSame = true ;
                                cerr << "les supports consécutifs de w sont identiques" << endl;
                            }
                            else cerr << "les supports consécutifs de w sont différents" << "  norme de w= " << w.norm() << endl;
*/
                            grad_f_w = 2*AT*(A*w-b) ;
                        } // while(grad_f_w.squaredNorm() > tol || !consecutiveWareTheSame)
                    } // else if(nbStep == r+2)

                    reducedSupportW = ReducedSupport(w);

                    for(unsigned int i=0; i<reducedSupportW.size(); i++)
                    {
                        size_t r = reducedSupportW[i];
                        material[r].addForce(f[r],x[r],v[r]);
                        for(size_t c=0; c<f[r].size(); ++c)
                        {
                            f[r][c] *= w(r);
                        }
                        //std::cerr << "après: " << r << " " << f[r] << std::endl;
                    }
                } // optimizedCubature
/*
                /////////////////////////////// Calcul de la matrice de sélection ////////////////////////////
                /// la cubature permet de ne pas prendre en compte la majorité des points d'integration.
                /// Ensuite, il faut passer du noeud (F,P) au noeud (x,f) par le biais d'un mapping
                /// qui fait le lien entre les deux noeuds par le produit
                /// F = Jx où J est une matrice de 9*totalPI lignes.
                /// Après la cubature, on souhaite obtenir un J réduit ont le nombre de lignes 9*totalPIApresCubature
                /// où totalPIApresCubature << totalPI.
                /// Ce J réduit, que je nomme J', s'obtient en faisant J' = SJ
                /// où S est la "matrice de séléction".
                /// Si J' = J, alors S serait une matrice "ientité par blocs de 9" de 9*totalPI ligne et 9*totalPI colonnes
                /// Sinon, S est une matrice possèdant un bloc de 9 à partir de la 9*i ième colonne et de la i-k ième ligne
                /// si "i" est le numéro d'un des PI sélectionné et sion en a retiré "k" avant lui.
                /// S est donc de taille 9*totalPIApresCubature lignes sur 9*totalPI colonnes.

                typedef Eigen::Triplet<double> TselectionMatrix;
                std::vector<TselectionMatrix> tripletListTselectionMatrix;
                tripletListTselectionMatrix.reserve(9*reducedSupportW.size());

                for(std::size_t e = 0 ; e < reducedSupportW.size() ; ++e)
                {
                    size_t selectedPI = reducedSupportW[e];
                    for(std::size_t i = 0 ; i < 9 ; ++i)
                    {
                        tripletListTselectionMatrix.push_back(TselectionMatrix(e+i,selectedPI+i,1));
                    }
                }
                selectionMatrix.setFromTriplets(tripletListTselectionMatrix.begin(), tripletListTselectionMatrix.end());*/
            } // cubature

            else // !cubature
            {
                SReal Ec = 0;
                for(unsigned int i=0; i<material.size(); i++)
                {
                    material[i].addForce(f[i],x[i],v[i]);
                    //Ec+=0.5*pow(v[i][c],2)
                }

            } // !cubature
        } // !switchMode

        _f.endEdit();

        if(!BlockType::constantK && this->assemble.getValue())
        {
            updateK();
            updateB();
        }

        //if(this->f_printLog.getValue())
        if(false)
        {
            Real W=0;
            for(unsigned int i=0; i<material.size(); i++)
            {
                W+=material[i].getPotentialEnergy(x[i]);
            }

            if(true)
            {
                ////////// recording energy versus time in a file /////////
                if(nbStep>1)
                {
                    if(nbStep==2)cerr << "start recording" << endl;
                    potentialEnergyStream << nbStep << " " << W << endl ;
                }
                if(nbStep==100)
                {
                    cerr << "stop recording" << endl;
                    potentialEnergyStream.close();
                }
            }
        }
        ++nbStep;
    }

    virtual void addDForce( const core::MechanicalParams* mparams, DataVecDeriv&  _df, const DataVecDeriv& _dx )
    {
        VecDeriv&  df = *_df.beginEdit();
        const VecDeriv&  dx = _dx.getValue();

        if(this->assemble.getValue())
        {
            B.addMult(df,dx,mparams->bFactor());
            K.addMult(df,dx,mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()));
        }
        else
        {
            for(unsigned int i=0; i<material.size(); i++)
            {
                material[i].addDForce(df[i],dx[i],mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()),mparams->bFactor());
            }
        }

        _df.endEdit();
    }


    const defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams * /*mparams*/)
    {
        if( !this->assemble.getValue() || !BlockType::constantK)
        {
            // MattN: quick and dirty fix to update the compliance matrix for a non-linear material
            // C is generally computed as K^{-1}, K is computed in addForce that is not call for compliances...
            // A deeper modification in forcefield API is required to fix this for all forcedields
            // maybe a cleaner fix is possible only for flexible
            {
                const DataVecCoord& xx = *this->mstate->read(core::ConstVecCoordId::position());
                const DataVecCoord& vv = *this->mstate->read(core::ConstVecDerivId::velocity());
                const VecCoord&  x = xx.getValue();
                const VecDeriv&  v = vv.getValue();
                VecDeriv f_bidon; f_bidon.resize( x.size() );
                for(unsigned int i=0; i<material.size(); i++)
                    material[i].addForce(f_bidon[i],x[i],v[i]); // too much stuff is computed there but at least C is updated
            }

            updateC();
        }
        return &C;
    }

    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset )
    {
        if(!this->assemble.getValue() || !BlockType::constantK)
        {
            updateK();
        }

        K.addToBaseMatrix( matrix, kFact, offset );
    }

    virtual void addBToMatrix(sofa::defaulttype::BaseMatrix *matrix, SReal bFact, unsigned int &offset)
    {
        if(!this->assemble.getValue() || !BlockType::constantK) updateB();

        B.addToBaseMatrix( matrix, bFact, offset );
    }

    void draw(const core::visual::VisualParams* /*vparams*/)
    {
    }
    //@}


    virtual SReal getPotentialEnergy( const core::MechanicalParams* /*mparams*/, const DataVecCoord& x ) const
    {
        SReal e = 0;
        const VecCoord& _x = x.getValue();

        for( unsigned int i=0 ; i<material.size() ; i++ )
        {
            e += material[i].getPotentialEnergy( _x[i] );
        }
        return e;
    }

    virtual SReal getPotentialEnergy( const unsigned int index ) const
    {
        if(!this->mstate) return 0;
        helper::ReadAccessor<Data< VecCoord > >  x(*this->mstate->read(core::ConstVecCoordId::position()));
        if(index>=material.size()) return 0;
        if(index>=x.size()) return 0;
        return material[index].getPotentialEnergy( x[index] );
    }


    Data<bool> assemble;

protected:

    CubatureBaseMaterialForceFieldT(core::behavior::MechanicalState<DataTypes> *mm = NULL)
        : Inherit(mm)
        , assemble ( initData ( &assemble,false, "assemble","Assemble the needed material matrices (compliance C,stiffness K,damping B)" ) )
        , cubature(initData(&cubature, (bool) false, "cubature", "Do we use cubature (naive or optimized) ?"))
        , optimizedCubature(initData(&optimizedCubature, (bool) false, "optimizedCubature", "Do we use the optimized cubature?"))
        , integrationPointsPercent(initData(&integrationPointsPercent, (SReal) 0.1, "integrationPointsPercent", "Number of integration points we want to select"))
        , nbElements(initData(&nbElements, (size_t) 0, "nbElements", "number of finite elements"))
        , nbTrainingPositions(initData(&nbTrainingPositions, (size_t) 0, "nbTrainingPositions", "number of training position used for the conjugate gradient algo in the optimized cubature"))
        , switchMode(initData(&switchMode, (bool) false, "switchMode", "is the force computed alternatively with and without cubature"))
        , potentialEnergyFile(initData(&potentialEnergyFile, "potentialEnergyFile", "file that contains the potential energie versus the time, with a certain starting position"))
    {

    }

    virtual ~CubatureBaseMaterialForceFieldT()    {     }

    Data<bool> cubature;
    Data<bool> optimizedCubature;
    Data<bool> switchMode;
    Data<SReal> integrationPointsPercent;
    Data<size_t> nbElements;
    Data<size_t> nbEigenModes;
    Data<size_t> nbTrainingPositions;

    // files for the tests
    Data<std::string> potentialEnergyFile;

    // Stream for the files above
    std::ofstream potentialEnergyStream;
    SparseMatrix material;

    SparseMatrixEigen C;

    Eigen::MatrixXd theForce;
    size_t nbe;
    size_t nbPI;
    size_t totalPI;
    size_t s;
    Vector b, w;
    SReal initialValue;
    std::vector<SReal> reducedSupportW;

    cmat A, selectionMatrix;
    std::vector<triplet> tripletListA;
    Eigen::MatrixXd spreadStress, stressAfterCubature, stressBeforeCubature ;

    size_t nbDDL;
    size_t r;
    size_t nbStep;
    size_t stressSize;
    size_t nbRows;


    void updateC()
    {
        unsigned int size = this->mstate->getSize();

        C.resizeBlocks(size,size);

        if(cubature.getValue())
        {
            for(unsigned int i=0; i<reducedSupportW.size(); i++)
            {
                size_t r = reducedSupportW[i];
                C.beginBlockRow(r);
                C.createBlock(r,material[r].getK()*w(r));
                C.endBlockRow();
            }
        }
        else
        {
            for(unsigned int i=0; i<material.size(); i++)
            {
                C.beginBlockRow(i);
                C.createBlock(i,material[i].getK());
                C.endBlockRow();
            }
        }
        C.compress();
    }

    SparseMatrixEigen K;

    void updateK()
    {
        unsigned int size = this->mstate->getSize();

        K.resizeBlocks(size,size);

        if(cubature.getValue())
        {
            for(unsigned int i=0; i<reducedSupportW.size(); i++)
            {
                size_t r = reducedSupportW[i];
                K.beginBlockRow(r);
                K.createBlock(r,material[r].getK()*w(r));
                K.endBlockRow();
            }
        }
        else
        {
            for(unsigned int i=0; i<material.size(); i++)
            {
                K.beginBlockRow(i);
                K.createBlock(i,material[i].getK());
                K.endBlockRow();
            }
        }
        K.compress();
    }


    SparseMatrixEigen B;

    void updateB()
    {
        unsigned int size = this->mstate->getSize();

        B.resizeBlocks(size,size);

        if(cubature.getValue())
        {
            for(unsigned int i=0; i<reducedSupportW.size(); i++)
            {
                size_t r = reducedSupportW[i];
                B.beginBlockRow(r);
                B.createBlock(r,material[r].getK()*w(r));
                B.endBlockRow();
            }
        }
        else
        {
            for(unsigned int i=0; i<material.size(); i++)
            {
                B.beginBlockRow(i);
                B.createBlock(i,material[i].getK());
                B.endBlockRow();
            }
        }
        B.compress();
    }

};


}
}
}

#endif
