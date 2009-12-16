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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/LCPcalc.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <cstring>

namespace sofa
{

namespace helper
{

using namespace std;


LCP::LCP(unsigned int mxC) : maxConst(mxC), tol(0.00001), numItMax(1000), useInitialF(true), mu(0.0), dim(0)
{
    W = new double*[maxConst];
    for (int i = 0; i < maxConst; i++)
    {
        W[i] = new double[maxConst];
        memset(W[i], 0, maxConst * sizeof(double));
    }
    dfree = new double[maxConst];
    f = new double[2 * maxConst + 1];

    memset(dfree, 0, maxConst * sizeof(double));
    memset(f, 0, (2 * maxConst + 1) * sizeof(double));
}
/*
LCP& LCP::operator=(LCP& lcp)
{
	if(this == &lcp) return *this; //self assignment

	if(maxConst != lcp.maxConst)
	{
		maxConst = lcp.maxConst;

		delete [] dfree;
		for (unsigned int i = 0; i < maxConst; i++)
		{
			delete [] W[i];
		}
		delete [] W;

		W = new double*[maxConst];
		for (unsigned int i = 0; i < maxConst; i++)
		{
			W[i] = new double[maxConst];
		}
		dfree = new double[maxConst];
		f = new double[2 * maxConst + 1];
	}

	dim = lcp.dim;
	mu = lcp.mu;
	tol = lcp.tol;
	numItMax = lcp.numItMax;
	useInitialF = lcp.useInitialF;
	nbConst = lcp.nbConst;

	for (unsigned int i = 0; i < maxConst; i++)
		memcpy(W[i], lcp.W[i], maxConst * sizeof(double));
	memcpy(dfree, lcp.dfree, maxConst * sizeof(double));
	memcpy(f, lcp.f, maxConst * sizeof(double));

	return *this;
}
*/

LCP::~LCP()
{
    delete [] dfree;
    for (int i = 0; i < maxConst; i++)
    {
        delete [] W[i];
    }
    delete [] W;
}


void LCP::reset(void)
{

    for (int i = 0; i < maxConst; i++)
    {
        memset(W[i], 0, maxConst * sizeof(double));
    }

    memset(dfree, 0, maxConst * sizeof(double));
}



//#include "mex.h"
/* Resoud un LCP écrit sous la forme U = q + M.F
 * dim : dimension du pb
 * res[0..dim-1] = U
 * res[dim..2*dim-1] = F
 */
int resoudreLCP(int dim, double * q, double ** M, double * res)
{

    // déclaration des variables
    int compteur;	// compteur de boucle
    int compteur2;	// compteur de boucle
    double ** mat;	// matrice de travail
    int * base;		// base des variables non nulles
    int ligPiv;		// ligne du pivot
    int colPiv;		// colonne du pivot
    double pivot;	// pivot
    double min;		// recherche du minimum pour le pivot
    double coeff;	// valeur du coefficient de la combinaison linéaire
    int boucles;	// compteur du nombre de passages dans la boucle
    int result=1;

    // allocation de la mémoire nécessaire
    mat = (double **)malloc(dim*sizeof(double *));
    for(compteur=0; compteur<dim; compteur++)
    {
        mat[compteur]=(double *)malloc((2*dim+1)*sizeof(double));
    }

    base = (int *)malloc(dim*sizeof(int));

    // initialisation de la matrice de travail
    for(compteur=0; compteur<dim; compteur++)
    {
        // colonnes correspondantes ?w
        for(compteur2=0; compteur2<dim; compteur2++)
        {
            if(compteur2==compteur)
            {
                mat[compteur][compteur2] = 1;
            }
            else
            {
                mat[compteur][compteur2] = 0;
            }
        }
        // colonnes correspondantes ?z
        for(; compteur2<2*dim; compteur2++)
        {
            mat[compteur][compteur2] = -(M[compteur][compteur2-dim]);
        }
        // colonne correspondante ?q
        mat[compteur][compteur2] = q[compteur];
    }

    /*printf("mat = [");
    for(compteur=0;compteur<dim;compteur++) {
    	for(compteur2=0;compteur2<2*dim+1;compteur2++) {
    		printf("\t%.2f",mat[compteur][compteur2]);
    	}
    	printf("\n");
    }
    printf("      ]\n\n");*/

    // initialisation de la base
    for(compteur=0; compteur<dim; compteur++)
    {
        base[compteur]=compteur;
    }

    // initialisation du nombre de boucles
    boucles=0;

    // recherche de la ligne du pivot
    ligPiv=-1;
    min = -EPSILON_LCP;
    for(compteur=0; compteur<dim; compteur++)
    {
        if (mat[compteur][2*dim]<min)
        {
            ligPiv=compteur;
            min=mat[compteur][2*dim];
        }
    }

    // tant que tous les q[i] ne sont pas > 0 et qu'on ne boucle pas trop
    while ((ligPiv>=0) && (boucles<MAX_BOU))
    {
        // augmentation du nombre de passages dans cette boucle
        boucles++;
        // recherche de la colonne du pivot
        if (base[ligPiv]<dim)
        {
            // c'est un wi dans la base
            colPiv=dim+base[ligPiv];
        }
        else
        {
            // c'est un zi dans la base
            colPiv=base[ligPiv]-dim;
        }

        // stockage de la valeur du pivot
        pivot=mat[ligPiv][colPiv];
        // et son affichage
        // printf("pivot=mat[%d][%d]=%f\n\n",ligPiv,colPiv,pivot);

        // si le pivot est nul, le LCP echoue
        if (fabs(pivot)<EPSILON_LCP)
        {
            afficheLCP(q,M,dim);
            printf("*** Pas de solution *** \n");
            boucles=MAX_BOU;
            result=0;
            return result;
        }
        else
        {
            // division de la ligne du pivot par le pivot
            for(compteur=0; compteur<2*dim+1; compteur++)
            {
                mat[ligPiv][compteur]/=pivot;
            }

            // combinaisons linéaires mettant la colonne du pivot a 0
            for(compteur=0; compteur<dim; compteur++)
            {
                if (compteur!=ligPiv)
                {
                    coeff=mat[compteur][colPiv];
                    for(compteur2=0; compteur2<2*dim+1; compteur2++)
                    {
                        mat[compteur][compteur2]-=coeff*mat[ligPiv][compteur2];
                    }
                }
            }

            // on rentre dans la base la nouvelle variable
            base[ligPiv]=colPiv;

            // recherche de la nouvelle ligne du pivot
            ligPiv=-1;
            min = -EPSILON_LCP;
            for(compteur=0; compteur<dim; compteur++)
            {
                if (mat[compteur][2*dim]<min)
                {
                    ligPiv=compteur;
                    min=mat[compteur][2*dim];
                }
            }

        }
    }

    // affichage du nb de boucles
    //printf("\n %d boucle(s) ",boucles);

    // stockage du resultat
    for(compteur=0; compteur<2*dim; compteur++)
    {
        res[compteur]=0;
    }
    // si on est arrivé à résoudre le pb, seules les variables en base sont non nulles
    if (boucles<MAX_BOU)
    {
        for(compteur=0; compteur<dim; compteur++)
        {
            res[base[compteur]]=mat[compteur][2*dim];
        }
    }

    // libération de la mémoire allouée
    for(compteur=0; compteur<dim; compteur++)
    {
        free(mat[compteur]);
    }
    free(mat);

    free(base);

    return result;
}




void afficheSyst(double *q,double **M, int *base, double **mat, int dim)
{
    int compteur, compteur2;

    // affichage de la matrice du LCP
    printf("M = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        for(compteur2=0; compteur2<dim; compteur2++)
        {
            printf("\t%.4f",M[compteur][compteur2]);
        }
        printf("\n");
    }
    printf("      ]\n\n");

    // affichage de q
    printf("q = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        printf("\t%.4f\n",q[compteur]);
    }
    printf("      ]\n\n");

    // afficahge base courante
    printf("B = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        printf("\t%d",base[compteur]);
    }
    printf("\t]\n\n");

    // affichage matrice courante
    printf("mat = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        for(compteur2=0; compteur2<2*dim+1; compteur2++)
        {
            printf("\t%.4f",mat[compteur][compteur2]);
        }
        printf("\n");
    }
    printf("      ]\n\n");
}

/* Siconos-Numerics version 1.2.0, Copyright INRIA 2005-2006.
 * Siconos is a program dedicated to modeling, simulation and control
 * of non smooth dynamical systems.
 * Siconos is a free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * Siconos is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Siconos; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Contact: Vincent ACARY vincent.acary@inrialpes.fr
*/

/*!\file lcp_lexicolemke.c
 *
 * This subroutine allows the resolution of LCP (Linear Complementary Problem).\n
 * Try \f$(z,w)\f$ such that:\n
 * \f$
 *  \left\lbrace
 *   \begin{array}{l}
 *    w - M z = q\\
 *    0 \le z \perp w \ge 0\\
 *   \end{array}
 *  \right.
 * \f$
 *
 * where M is an (\f$nn \times nn\f$)-matrix, q , w and z nn-vectors.
 */


/*!\fn  void lcp_lexicolemke( int *nn , double *vec , double *q , double *zlem , double *wlem , int *info , int *iparamLCP , double *dparamLCP )

  lcp_lexicolemke is a direct solver for LCP based on pivoting method principle for degenrate problem.\n
  Choice of pivot variable is performed via lexicographic ordering
  Ref: "The Linear Complementary Problem" Cottle, Pang, Stone (1992)\n


  \param nn      On enter, an integer which represents the dimension of the system.
  \param vec     On enter, a (\f$nn\times nn\f$)-vector of doubles which contains the components of the matrix with a fortran storage.
  \param q       On enter, a nn-vector of doubles which contains the components of the right hand side vector.
  \param zlem    On return, a nn-vector of doubles which contains the solution of the problem.
  \param wlem    On return, a nn-vector of doubles which contains the solution of the problem.
  \param info    On return, an integer which returns the termination value:\n
                 0 : convergence\n
                 1 : iter = itermax\n
                 2 : negative diagonal term\n

  \param iparamLCP  On enter/return, a vetor of integers:\n
                 - iparamLCP[0] = itermax On enter, the maximum number of pivots allowed.
                 - iparamLCP[1] = ispeak  On enter, the output log identifiant:\n
                        0 : no output\n
                        >0: active screen output\n
                 - iparamLCP[2] = it_end  On return, the number of pivots performed by the algorithm.

  \param dparamLCP  On enter/return, a vetor of doubles (not used).\n



  \author Mathieu Renouf

 */
//void lcp_lexicolemke( int *nn , double *vec , double *q , double *zlem , double *wlem , int *info , int *iparamLCP , double *dparamLCP ){
//void lcp_lexicolemke( int dim , double *vec , double *q , double *zlem )

int lcp_lexicolemke(int dim, double * q, double ** M, double * res)
{

    int i,drive,block,Ifound;
    int ic,jc;
    int dim2,ITER;
    int nobasis;
    int itermax,ispeak;

    double qs,z0,zb,dblock;
    double pivot, tovip;
    double tmp;
    int *basis;
    static double** A;
    //static int dimTest=0;

    dim2 = 2*(dim+1);

    /*input*/

    itermax = dim2;
    ispeak  = 0;

    /*output*/


    basis = (int *)malloc( dim*sizeof(int) );

    /* Allocation */
    A = (double **)malloc( dim*sizeof(double*) );
    for( ic = 0 ; ic < dim; ++ic )
        A[ic] = (double *)malloc( dim2*sizeof(double) );

    for( ic = 0 ; ic < dim; ++ic )
        for( jc = 0 ; jc < dim2; ++jc )
            A[ic][jc] = 0.0;

    /* construction of A matrix such as
     * A = [ q | Id | -d | -M ] with d = (1,...1)
     */

    for( ic = 0 ; ic < dim; ++ic )
        for( jc = 0 ; jc < dim; ++jc )
            A[ic][jc+dim+2] = -M[ic][jc];

    for( ic = 0 ; ic < dim; ++ic ) A[ic][0] = q[ic];

    for( ic = 0 ; ic < dim; ++ic ) A[ic][ic+1 ] =  1.0;
    for( ic = 0 ; ic < dim; ++ic ) A[ic][dim+1] = -1.0;

    /* End of construction of A */

    /* STEP 0
     * qs = min{ q[i], i=1,...,NC }
     */

    qs = q[0];

    for( ic = 1 ; ic < dim ; ++ic )
    {
        if( q[ic] < qs ) qs = q[ic];
    }

    Ifound = 0;

    ITER=0;
    if( qs >= 0 )
    {

        /* TRIVIAL CASE
         * z = 0 and w = q is solution of LCP(q,M)
         */

        for( ic = 0 ; ic < dim; ++ic )
        {
            res[ic] = 0.0;
            //wlem[ic] = q[ic];
            z0 = 0.0;
        }

        Ifound=1;

    }
    else
    {

        for( ic = 0 ; ic < dim  ; ++ic ) basis[ic]=ic+1;

        drive = dim+1;
        block = 0;
        z0 = A[block][0];


        /* Start research of argmin lexico */
        /* With this first step the covering vector enter in the basis */

        for( ic = 1 ; ic < dim ; ++ic )
        {
            zb = A[ic][0];
            if( zb < z0 )
            {
                z0    = zb;
                block = ic;
            }
            else if( zb == z0 )
            {
                for( jc = 0 ; jc < dim ; ++jc )
                {
                    dblock = A[block][1+jc] - A[ic][1+jc];
                    if( dblock < 0 )
                    {
                        break;
                    }
                    else if( dblock > 0 )
                    {
                        block = ic;
                        break;
                    }
                }
            }
        }

        /* Stop research of argmin lexico */

        pivot = A[block][drive];
        tovip = 1.0/pivot;

        /* Pivot < block , drive > */

        A[block][drive] = 1;
        for( ic = 0       ; ic < drive ; ++ic ) A[block][ic] = A[block][ic]*tovip;
        for( ic = drive+1 ; ic < dim2  ; ++ic ) A[block][ic] = A[block][ic]*tovip;

        /* */

        for( ic = 0 ; ic < block ; ++ic )
        {
            tmp = A[ic][drive];
            for( jc = 0 ; jc < dim2 ; ++jc ) A[ic][jc] -=  tmp*A[block][jc];
        }
        for( ic = block+1 ; ic < dim ; ++ic )
        {
            tmp = A[ic][drive];
            for( jc = 0 ; jc < dim2 ; ++jc ) A[ic][jc] -=  tmp*A[block][jc];
        }

        nobasis = basis[block];
        basis[block] = drive;

        while( ITER < itermax && !Ifound )
        {

            ++ITER;

            if( nobasis < dim + 1 )      drive = nobasis + (dim+1);
            else if( nobasis > dim + 1 ) drive = nobasis - (dim+1);

            /* Start research of argmin lexico for minimum ratio test */

            pivot = 1e20;
            block = -1;

            for( ic = 0 ; ic < dim ; ++ic )
            {
                zb = A[ic][drive];
                if( zb > 0.0 )
                {
                    z0 = A[ic][0]/zb;
                    if( z0 > pivot ) continue;
                    if( z0 < pivot )
                    {
                        pivot = z0;
                        block = ic;
                    }
                    else
                    {
                        for( jc = 1 ; jc < dim+1 ; ++jc )
                        {
                            dblock = A[block][jc]/pivot - A[ic][jc]/zb;
                            if( dblock < 0 ) break;
                            else if( dblock > 0 )
                            {
                                block = ic;
                                break;
                            }
                        }
                    }
                }
            }
            if( block == -1 ) break;

            if( basis[block] == dim+1 ) Ifound = 1;

            /* Pivot < block , drive > */

            pivot = A[block][drive];
            tovip = 1.0/pivot;
            A[block][drive] = 1;

            for( ic = 0       ; ic < drive ; ++ic ) A[block][ic] = A[block][ic]*tovip;
            for( ic = drive+1 ; ic < dim2  ; ++ic ) A[block][ic] = A[block][ic]*tovip;

            /* */

            for( ic = 0 ; ic < block ; ++ic )
            {
                tmp = A[ic][drive];
                for( jc = 0 ; jc < dim2 ; ++jc ) A[ic][jc] -=  tmp*A[block][jc];
            }
            for( ic = block+1 ; ic < dim ; ++ic )
            {
                tmp = A[ic][drive];
                for( jc = 0 ; jc < dim2 ; ++jc ) A[ic][jc] -=  tmp*A[block][jc];
            }

            nobasis = basis[block];
            basis[block] = drive;

        }

        for( ic = 0 ; ic < dim; ++ic )
        {
            drive = basis[ic];
            if( drive < dim + 1 )
            {
                res[drive-1] = 0.0;
                //wlem[drive-1] = A[ic][0];
            }
            else if( drive > dim + 1 )
            {
                res[drive-dim-2] = A[ic][0];
                //wlem[drive-dim-2] = 0.0;
            }
        }

    }



//  if(Ifound) *info = 0;
// else *info = 1;

    free(basis);


    for( i = 0 ; i < dim ; ++i ) free(A[i]);
    free(A);



    // for compatibility with previous LCP solver
    for( i = 0 ; i < dim ; ++i )
        res[i+dim] = res[i];


    if (Ifound) return 1;

    printf("\n Problem with this LCP :\n");
    afficheLCP(q,M,dim);
    return 0;

}

/******************************** WITHOUT ALLOCATION of A ************************************/
int lcp_lexicolemke(int dim, double * q, double ** M, double **A, double * res)
{

    int i,drive,block,Ifound;
    int ic,jc;
    int dim2,ITER;
    int nobasis;
    int itermax,ispeak;

    double qs,z0,zb,dblock;
    double pivot, tovip;
    double tmp;
    int *basis;

    dim2 = 2*(dim+1);

    /*input*/

    itermax = dim2;
    ispeak  = 0;

    /*output*/

    basis = (int *)malloc( dim*sizeof(int) );

    for( ic = 0 ; ic < dim; ++ic )
        for( jc = 0 ; jc < dim2; ++jc )
            A[ic][jc] = 0.0;

    /* construction of A matrix such as
     * A = [ q | Id | -d | -M ] with d = (1,...1)
     */

    for( ic = 0 ; ic < dim; ++ic )
        for( jc = 0 ; jc < dim; ++jc )
            A[ic][jc+dim+2] = -M[ic][jc];

    for( ic = 0 ; ic < dim; ++ic ) A[ic][0] = q[ic];

    for( ic = 0 ; ic < dim; ++ic ) A[ic][ic+1 ] =  1.0;
    for( ic = 0 ; ic < dim; ++ic ) A[ic][dim+1] = -1.0;

    /* End of construction of A */

    /* STEP 0
     * qs = min{ q[i], i=1,...,NC }
     */

    qs = q[0];

    for( ic = 1 ; ic < dim ; ++ic )
    {
        if( q[ic] < qs ) qs = q[ic];
    }

    Ifound = 0;

    ITER=0;
    if( qs >= 0 )
    {

        /* TRIVIAL CASE
         * z = 0 and w = q is solution of LCP(q,M)
         */

        for( ic = 0 ; ic < dim; ++ic )
        {
            res[ic] = 0.0;
            //wlem[ic] = q[ic];
            z0 = 0.0;
        }

        Ifound=1;

    }
    else
    {

        for( ic = 0 ; ic < dim  ; ++ic ) basis[ic]=ic+1;

        drive = dim+1;
        block = 0;
        z0 = A[block][0];


        /* Start research of argmin lexico */
        /* With this first step the covering vector enter in the basis */

        for( ic = 1 ; ic < dim ; ++ic )
        {
            zb = A[ic][0];
            if( zb < z0 )
            {
                z0    = zb;
                block = ic;
            }
            else if( zb == z0 )
            {
                for( jc = 0 ; jc < dim ; ++jc )
                {
                    dblock = A[block][1+jc] - A[ic][1+jc];
                    if( dblock < 0 )
                    {
                        break;
                    }
                    else if( dblock > 0 )
                    {
                        block = ic;
                        break;
                    }
                }
            }
        }

        /* Stop research of argmin lexico */

        pivot = A[block][drive];
        tovip = 1.0/pivot;

        /* Pivot < block , drive > */

        A[block][drive] = 1;
        for( ic = 0       ; ic < drive ; ++ic ) A[block][ic] = A[block][ic]*tovip;
        for( ic = drive+1 ; ic < dim2  ; ++ic ) A[block][ic] = A[block][ic]*tovip;

        /* */

        for( ic = 0 ; ic < block ; ++ic )
        {
            tmp = A[ic][drive];
            for( jc = 0 ; jc < dim2 ; ++jc ) A[ic][jc] -=  tmp*A[block][jc];
        }
        for( ic = block+1 ; ic < dim ; ++ic )
        {
            tmp = A[ic][drive];
            for( jc = 0 ; jc < dim2 ; ++jc ) A[ic][jc] -=  tmp*A[block][jc];
        }

        nobasis = basis[block];
        basis[block] = drive;

        while( ITER < itermax && !Ifound )
        {

            ++ITER;

            if( nobasis < dim + 1 )      drive = nobasis + (dim+1);
            else if( nobasis > dim + 1 ) drive = nobasis - (dim+1);

            /* Start research of argmin lexico for minimum ratio test */

            pivot = 1e20;
            block = -1;

            for( ic = 0 ; ic < dim ; ++ic )
            {
                zb = A[ic][drive];
                if( zb > 0.0 )
                {
                    z0 = A[ic][0]/zb;
                    if( z0 > pivot ) continue;
                    if( z0 < pivot )
                    {
                        pivot = z0;
                        block = ic;
                    }
                    else
                    {
                        for( jc = 1 ; jc < dim+1 ; ++jc )
                        {
                            dblock = A[block][jc]/pivot - A[ic][jc]/zb;
                            if( dblock < 0 ) break;
                            else if( dblock > 0 )
                            {
                                block = ic;
                                break;
                            }
                        }
                    }
                }
            }
            if( block == -1 ) break;

            if( basis[block] == dim+1 ) Ifound = 1;

            /* Pivot < block , drive > */

            pivot = A[block][drive];
            tovip = 1.0/pivot;
            A[block][drive] = 1;

            for( ic = 0       ; ic < drive ; ++ic ) A[block][ic] = A[block][ic]*tovip;
            for( ic = drive+1 ; ic < dim2  ; ++ic ) A[block][ic] = A[block][ic]*tovip;

            /* */

            for( ic = 0 ; ic < block ; ++ic )
            {
                tmp = A[ic][drive];
                for( jc = 0 ; jc < dim2 ; ++jc ) A[ic][jc] -=  tmp*A[block][jc];
            }
            for( ic = block+1 ; ic < dim ; ++ic )
            {
                tmp = A[ic][drive];
                for( jc = 0 ; jc < dim2 ; ++jc ) A[ic][jc] -=  tmp*A[block][jc];
            }

            nobasis = basis[block];
            basis[block] = drive;

        }

        for( ic = 0 ; ic < dim; ++ic )
        {
            drive = basis[ic];
            if( drive < dim + 1 )
            {
                res[drive-1] = 0.0;
                //wlem[drive-1] = A[ic][0];
            }
            else if( drive > dim + 1 )
            {
                res[drive-dim-2] = A[ic][0];
                //wlem[drive-dim-2] = 0.0;
            }
        }

    }



//  if(Ifound) *info = 0;
// else *info = 1;

    free(basis);

    /*
    for( i = 0 ; i < dim ; ++i ) free(A[i]);
    free(A);
    */


    // for compatibility with previous LCP solver
    for( i = 0 ; i < dim ; ++i )
        res[i+dim] = res[i];


    if (Ifound) return 1;

    printf("\n Problem with this LCP :\n");
//  afficheLCP(q,M,dim);
    return 0;

}
/********************************************************************************************/
void afficheLCP(double *q, double **M, int dim)
{
    int compteur, compteur2;
    // affichage de la matrice du LCP
    printf("M = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        for(compteur2=0; compteur2<dim; compteur2++)
        {
            printf("\t%.4f",M[compteur][compteur2]);
        }
        printf("\n");
    }
    printf("      ];\n\n");

    // affichage de q
    printf("q = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        printf("\t%.4f\n",q[compteur]);
    }
    printf("      ]\n\n");
}

/********************************************************************************************/
void afficheLCP(double *q, double **M, double *f, int dim)
{
    int compteur, compteur2;
    // affichage de la matrice du LCP
    printf("\n M = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        for(compteur2=0; compteur2<dim; compteur2++)
        {
            printf("\t%.9f",M[compteur][compteur2]);
        }
        printf("\n");
    }
    printf("      ];\n\n");

    // affichage de q
    printf("q = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        printf("\t%.9f\n",q[compteur]);
    }
    printf("      ];\n\n");

    // affichage de f
    printf("f = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        printf("\t%.9f\n",f[compteur]);
    }
    printf("      ];\n\n");

}


/********************************************************************************************/
// special class to obtain the inverse of a symetric matrix 3x3
void LocalBlock33::compute(double &w11, double &w12, double &w13, double &w22, double &w23, double &w33)
{
    w[0]=w11; w[1]=w12; w[2] = w13; w[3]=w22; w[4]=w23; w[5]=w33;
    det = w11*w22*w33-w11*w23*w23-w12*w12*w33+2*w12*w13*w23-w13*w13*w22;
    wInv[0] = (w22*w33-w23*w23)/det;
    wInv[1] = -(w12*w33-w13*w23)/det;
    wInv[2] = (w12*w23-w13*w22)/det;
    wInv[3] = (w11*w33-w13*w13)/det;
    wInv[4] = -(w11*w23-w12*w13)/det;
    wInv[5] = (w11*w22-w12*w12)/det;
    computed=true;
}

void LocalBlock33::stickState(double &dn, double &dt, double &ds, double &fn, double &ft, double &fs)
{
    fn = -wInv[0]*dn - wInv[1]*dt - wInv[2]*ds;
    ft = -wInv[1]*dn - wInv[3]*dt - wInv[4]*ds;
    fs = -wInv[2]*dn - wInv[4]*dt - wInv[5]*ds;
}




void LocalBlock33::slipState(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs)
{
    double d[3];
    double normFt;

    for (int iteration=0; iteration<10000; iteration++)
    {
        // we set the previous value of the force
        f_1[0]=fn; f_1[1]=ft; f_1[2]=fs;

        // evaluation of the current normal position
        d[0] = w[0]*fn + w[1]*ft + w[2]*fs + dn;
        // evaluation of the new contact force
        fn -= d[0]/w[0];

        // evaluation of the current tangent positions
        d[1] = w[1]*fn + w[3]*ft + w[4]*fs + dt;
        d[2] = w[2]*fn + w[4]*ft + w[5]*fs + ds;

        // envaluation of the new fricton forces
        ft -= 2*d[1]/(w[3]+w[5]);
        fs -= 2*d[2]/(w[3]+w[5]);
        normFt=sqrt(ft*ft+fs*fs);
        ft *=mu*fn/normFt;
        fs *=mu*fn/normFt;


        if (normError(fn,ft,fs,f_1[0],f_1[1],f_1[2]) < 0.000001)
        {
            dn=d[0]; dt=d[1]; ds=d[2];
            //mexPrintf("\n convergence of slipState after %d iteration(s)",iteration);
            return;
        }

    }
//	mexPrintf("\n No convergence in slipState function: error =%f",normError(fn,ft,fs,f_1[0],f_1[1],f_1[2]));
//	printf("\n No convergence in slipState function");

}

// computation of a new state using a simple gauss-seidel loop // pseudo-potential (new: dn, dt, ds already take into account current value of fn, ft and fs)
void LocalBlock33::New_GS_State(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs)
{

    double d[3];
    double normFt;
    f_1[0]=fn; f_1[1]=ft; f_1[2]=fs;

    // evaluation of the current normal position
    d[0] = dn;
    // evaluation of the new contact force
    fn -= d[0]/w[0];

    if (fn < 0)
    {
        fn=0; ft=0; fs=0;
        // if the force was previously not null -> update the state
        if (f_1[0]>0)
        {
            double df[3];
            df[0] = fn-f_1[0];  df[1] = ft-f_1[1];  df[2] = fs-f_1[2];

            dn += w[0]*df[0] + w[1]*df[1] + w[2]*df[2];
            dt += w[1]*df[0] + w[3]*df[1] + w[4]*df[2];
            ds += w[2]*df[0] + w[4]*df[1] + w[5]*df[2];
        }
        return;
    }


    // evaluation of the current tangent positions
    d[1] = w[1]*(fn-f_1[0]) + dt;
    d[2] = w[2]*(fn-f_1[0]) + ds;

    // envaluation of the new fricton forces
    ft -= 2*d[1]/(w[3]+w[5]);
    fs -= 2*d[2]/(w[3]+w[5]);

    normFt=sqrt(ft*ft+fs*fs);

    if (normFt > mu*fn)
    {
        ft *=mu*fn/normFt;
        fs *=mu*fn/normFt;
    }

    double df[3];
    df[0] = fn-f_1[0];  df[1] = ft-f_1[1];  df[2] = fs-f_1[2];

    dn += w[0]*df[0] + w[1]*df[1] + w[2]*df[2];
    dt += w[1]*df[0] + w[3]*df[1] + w[4]*df[2];
    ds += w[2]*df[0] + w[4]*df[1] + w[5]*df[2];



}

void LocalBlock33::GS_State(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs)
{
    double d[3];
    double normFt;
    f_1[0]=fn; f_1[1]=ft; f_1[2]=fs;

    // evaluation of the current normal position
    d[0] = w[0]*fn + w[1]*ft + w[2]*fs + dn;
    // evaluation of the new contact force
    fn -= d[0]/w[0];

    if (fn < 0)
    {
        fn=0; ft=0; fs=0;
        return;
    }


    // evaluation of the current tangent positions
    d[1] = w[1]*fn + w[3]*ft + w[4]*fs + dt;
    d[2] = w[2]*fn + w[4]*ft + w[5]*fs + ds;

    // envaluation of the new fricton forces
    ft -= 2*d[1]/(w[3]+w[5]);
    fs -= 2*d[2]/(w[3]+w[5]);

    normFt=sqrt(ft*ft+fs*fs);

    if (normFt > mu*fn)
    {
        ft *=mu*fn/normFt;
        fs *=mu*fn/normFt;
    }

    dn += w[0]*fn + w[1]*ft + w[2]*fs;
    dt += w[1]*fn + w[3]*ft + w[4]*fs;
    ds += w[2]*fn + w[4]*ft + w[5]*fs;

}


void LocalBlock33::BiPotential(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs)
{
    double d[3];
    double normFt;
    f_1[0]=fn; f_1[1]=ft; f_1[2]=fs;
///////////
// evaluation of a new contact force based on bi-potential approach
///////////

    // evaluation of the current position///
    d[0] = w[0]*fn + w[1]*ft + w[2]*fs + dn;
    d[1] = w[1]*fn + w[3]*ft + w[4]*fs + dt;
    d[2] = w[2]*fn + w[4]*ft + w[5]*fs + ds;

    // evaluate a unique compliance for both normal and tangential direction //
    double rho = (w[0] + w[3] + w[5]) / 3;

    // evaluation of the bi-potential
    double v[3];
    v[0] = d[0] + mu * sqrt(d[1]*d[1] + d[2]*d[2]);
    v[1] = d[1];
    v[2] = d[2];

    // evaluation of the new contact force
    fn -= v[0]/rho;
    ft -= v[1]/rho;
    fs -= v[2]/rho;


    // projection of the contact force on the Coulomb's friction cone

    if (fn < 0)
    {
        fn=0; ft=0; fs=0;
        return;
    }

    normFt=sqrt(ft*ft+fs*fs);
    if (normFt > mu*fn)
    {
        double proj = (normFt - mu * fn) / (1 + mu*mu);

        fn += mu * proj ;
        ft -= proj * ft/normFt;
        fs -= proj * fs/normFt;
    }

    dn += w[0]*fn + w[1]*ft + w[2]*fs;
    dt += w[1]*fn + w[3]*ft + w[4]*fs;
    ds += w[2]*fn + w[4]*ft + w[5]*fs;

}

////////////////////////////////
// // test sur LocalBlock33 // //
////////////////////////////////
// LocalBlock33 *Z;
// Z = new LocalBlock33(W[0][0],W[0][1],W[0][2],W[1][1],W[1][2],W[2][2]);
// Z->stickState(dfree[0],dfree[1],dfree[2],f[0],f[1],f[2]);

// if (nlhs>0)
// {
//double *prF;
//plhs[0]=mxCreateDoubleMatrix(3,1,mxREAL);
//prF = mxGetPr(plhs[0]);
//for(i=0; i<3; i++)
//	prF[i] = f[i];
// }


/********************************************************************************************/

//////////////
// sorted list of the contact (depending on interpenetration)
//////////////

typedef struct { double value; int index;} listElem;
struct listSortAscending
{
    bool operator()(const listElem& e1, const listElem& e2)
    {
        return e1.value < e2.value;
    }
};

int nlcp_gaussseidel(int dim, double *dfree, double**W, double *f, double mu, double tol, int numItMax, bool useInitialF, bool verbose)
{
    double test = dim/3;
    double zero = 0.0;
    int numContacts =  (int) floor(test);
    test = dim/3 - numContacts;

    if (test>0.01)
    {
        printf("\n WARNING dim should be dividable by 3 in nlcp_gaussseidel");
        return 0;
    }
    // iterators
    int it,c1,i;

    // memory allocation of vector d
    double *d;
    d = (double*)malloc(dim*sizeof(double));
    // put the vector force to zero
    if (!useInitialF)
        memset(f, 0, dim*sizeof(double));

    // previous value of the force and the displacment
    double f_1[3];
    double d_1[3];

    // allocation of the inverted system 3x3
    LocalBlock33 **W33;
    W33 = (LocalBlock33 **) malloc (dim*sizeof(LocalBlock33));
    for (c1=0; c1<numContacts; c1++)
        W33[c1] = new LocalBlock33();
    /*
    std::vector<listElem> sortedList;
    listElem buf;
    sortedList.clear();
    for (c1=0; c1<numContacts; c1++)
    {
    	buf.value = dfree[3*c1];
    	buf.index = c1;
    	sortedList.push_back(buf);
    }
    */

    //////////////
    // Beginning of iterative computations
    //////////////
    double error = 0;
    double dn, dt, ds, fn, ft, fs;

    for (it=0; it<numItMax; it++)
    {
        error =0;
        for (c1=0; c1<numContacts; c1++)
        {
            // index of contact
            int index1 = c1;

            // put the previous value of the contact force in a buffer and put the current value to 0
            f_1[0] = f[3*index1]; f_1[1] = f[3*index1+1]; f_1[2] = f[3*index1+2];
            set3Dof(f,index1,zero,zero,zero); //		f[3*index] = 0.0; f[3*index+1] = 0.0; f[3*index+2] = 0.0;

            // computation of actual d due to contribution of other contacts
            dn=dfree[3*index1]; dt=dfree[3*index1+1]; ds=dfree[3*index1+2];
            for (i=0; i<dim; i++)
            {
                dn += W[3*index1  ][i]*f[i];
                dt += W[3*index1+1][i]*f[i];
                ds += W[3*index1+2][i]*f[i];
            }
            d_1[0] = dn + W[3*index1  ][3*index1  ]*f_1[0]+W[3*index1  ][3*index1+1]*f_1[1]+W[3*index1  ][3*index1+2]*f_1[2];
            d_1[1] = dt + W[3*index1+1][3*index1  ]*f_1[0]+W[3*index1+1][3*index1+1]*f_1[1]+W[3*index1+1][3*index1+2]*f_1[2];
            d_1[2] = ds + W[3*index1+2][3*index1  ]*f_1[0]+W[3*index1+2][3*index1+1]*f_1[1]+W[3*index1+2][3*index1+2]*f_1[2];

            if(W33[index1]->computed==false)
            {
                W33[index1]->compute(W[3*index1][3*index1],W[3*index1][3*index1+1],W[3*index1][3*index1+2],
                        W[3*index1+1][3*index1+1], W[3*index1+1][3*index1+2],W[3*index1+2][3*index1+2]);
            }


            fn=f_1[0]; ft=f_1[1]; fs=f_1[2];
            W33[index1]->GS_State(mu,dn,dt,ds,fn,ft,fs);

            //W33[index1]->BiPotential(mu,dn,dt,ds,fn,ft,fs);

            error += absError(dn,dt,ds,d_1[0],d_1[1],d_1[2]);


            set3Dof(f,index1,fn,ft,fs);

        }

        if (error < tol*(numContacts+1))
        {
            free(d);
            for (int i = 0; i < numContacts; i++)
                delete W33[i];
            free(W33);
            //printf("Convergence after %d iteration(s) with tolerance : %f and error : %f with dim : %d\n",it, tol, error, dim);
            //afficheLCP(dfree,W,f,dim);
            return 1;
        }
    }
    free(d);
    for (int i = 0; i < numContacts; i++)
        delete W33[i];
    free(W33);

    if (verbose)
    {
        std::cerr<<"\n No convergence in  nlcp_gaussseidel function : error ="<<error <<" after"<< it<<" iterations"<<std::endl;
        afficheLCP(dfree,W,f,dim);
    }

    return 0;

}

int nlcp_gaussseidelTimed(int dim, double *dfree, double**W, double *f, double mu, double tol, int numItMax, bool useInitialF, double timeout, bool verbose)
{
    double test = dim/3;
    double zero = 0.0;
    int numContacts =  (int) floor(test);
    test = dim/3 - numContacts;

    ctime_t t0 = CTime::getTime()/CTime::getTicksPerSec();

    if (test>0.01)
    {
        printf("\n WARNING dim should be dividable by 3 in nlcp_gaussseidel");
        return 0;
    }
    // iterators
    int it,c1,i;

    // memory allocation of vector d
    double *d;
    d = (double*)malloc(dim*sizeof(double));
    // put the vector force to zero
    if (!useInitialF)
        memset(f, 0, dim*sizeof(double));

    // previous value of the force and the displacment
    double f_1[3];
    double d_1[3];

    // allocation of the inverted system 3x3
    LocalBlock33 **W33;
    W33 = (LocalBlock33 **) malloc (dim*sizeof(LocalBlock33));
    for (c1=0; c1<numContacts; c1++)
        W33[c1] = new LocalBlock33();
    /*
    std::vector<listElem> sortedList;
    listElem buf;
    sortedList.clear();
    for (c1=0; c1<numContacts; c1++)
    {
    	buf.value = dfree[3*c1];
    	buf.index = c1;
    	sortedList.push_back(buf);
    }
    */

    //////////////
    // Beginning of iterative computations
    //////////////
    double error = 0;
    double dn, dt, ds, fn, ft, fs;

    for (it=0; it<numItMax; it++)
    {
        error =0;
        for (c1=0; c1<numContacts; c1++)
        {
            // index of contact
            int index1 = c1;

            // put the previous value of the contact force in a buffer and put the current value to 0
            f_1[0] = f[3*index1]; f_1[1] = f[3*index1+1]; f_1[2] = f[3*index1+2];
            set3Dof(f,index1,zero,zero,zero); //		f[3*index] = 0.0; f[3*index+1] = 0.0; f[3*index+2] = 0.0;

            // computation of actual d due to contribution of other contacts
            dn=dfree[3*index1]; dt=dfree[3*index1+1]; ds=dfree[3*index1+2];
            for (i=0; i<dim; i++)
            {
                dn += W[3*index1  ][i]*f[i];
                dt += W[3*index1+1][i]*f[i];
                ds += W[3*index1+2][i]*f[i];
            }
            d_1[0] = dn + W[3*index1  ][3*index1  ]*f_1[0]+W[3*index1  ][3*index1+1]*f_1[1]+W[3*index1  ][3*index1+2]*f_1[2];
            d_1[1] = dt + W[3*index1+1][3*index1  ]*f_1[0]+W[3*index1+1][3*index1+1]*f_1[1]+W[3*index1+1][3*index1+2]*f_1[2];
            d_1[2] = ds + W[3*index1+2][3*index1  ]*f_1[0]+W[3*index1+2][3*index1+1]*f_1[1]+W[3*index1+2][3*index1+2]*f_1[2];

            if(W33[index1]->computed==false)
            {
                W33[index1]->compute(W[3*index1][3*index1],W[3*index1][3*index1+1],W[3*index1][3*index1+2],
                        W[3*index1+1][3*index1+1], W[3*index1+1][3*index1+2],W[3*index1+2][3*index1+2]);
            }


            fn=f_1[0]; ft=f_1[1]; fs=f_1[2];
            W33[index1]->GS_State(mu,dn,dt,ds,fn,ft,fs);
            error += absError(dn,dt,ds,d_1[0],d_1[1],d_1[2]);


            set3Dof(f,index1,fn,ft,fs);

            ctime_t t1 = CTime::getTime()/CTime::getTicksPerSec();
            if((t1-t0) > timeout)
            {
                free(d);
                for (int i = 0; i < numContacts; i++)
                    delete W33[i];
                free(W33);
                //printf("Convergence after %d iteration(s) with tolerance : %f and error : %f with dim : %d\n",it, tol, error, dim);
                //afficheLCP(dfree,W,f,dim);
                return 1;
            }
        }

        if (error < tol)
        {
            free(d);
            for (int i = 0; i < numContacts; i++)
                delete W33[i];
            free(W33);
            //printf("Convergence after %d iteration(s) with tolerance : %f and error : %f with dim : %d\n",it, tol, error, dim);
            //afficheLCP(dfree,W,f,dim);
            return 1;
        }
    }
    free(d);
    for (int i = 0; i < numContacts; i++)
        delete W33[i];
    free(W33);

    if (verbose)
    {
        printf("\n No convergence in nlcp_gaussseidel function : error =%f after %d iterations", error, it);
        afficheLCP(dfree,W,f,dim);
    }

    return 0;

}


/* Resoud un LCP écrit sous la forme U = q + M.F
 * dim : dimension du pb
 * res[0..dim-1] = U
 * res[dim..2*dim-1] = F
 */
void gaussSeidelLCP1(int dim, FemClipsReal * q, FemClipsReal ** M, FemClipsReal * res, double tol, int numItMax)
{
    int compteur;	// compteur de boucle
    int compteur2, compteur3;	// compteur de boucle

    double f_1;
    double error=0.0;

    for (compteur=0; compteur<numItMax; compteur++)
    {

        error=0.0;
        for (compteur2=0; compteur2<dim; compteur2++)
        {
            //res[compteur2]=(FemClipsReal)0.0;
            res[compteur2]=q[compteur2];
            for (compteur3=0; compteur3<dim; compteur3++)
            {
                res[compteur2]+=M[compteur2][compteur3]* res[dim+compteur3]/* + q[compteur2]*/;
            }
            res[compteur2] -= M[compteur2][compteur2]* res[dim+compteur2];
            f_1 = res[dim+compteur2];

            if (res[compteur2]<0)
            {
                res[dim+compteur2]=-res[compteur2]/M[compteur2][compteur2];
            }
            else
            {
                res[dim+compteur2]=(FemClipsReal)0.0;
            }

            error +=fabs( M[compteur2][compteur2] * (res[dim+compteur2] - f_1) );


        }

        if (error < tol)
        {
            //	std::cout << "convergence in gaussSeidelLCP1 with " << compteur << " iterations\n";
            break;
        }

    }

    for (compteur=0; compteur<dim; compteur++)
        res[compteur] = res[compteur+dim];

    if (error >= tol)
    {
        std::cout << "No convergence in gaussSeidelLCP1 : error = " << error << std::endl;
        //	afficheLCP(q, M, res, dim);
    }
}

} // namespace helper

} // namespace sofa
