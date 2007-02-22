#include <sofa/helper/LCPcalc.h>

namespace sofa
{

namespace helper
{


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
    // si on est arriv??résoudre le pb, seules les variables en base sont non nulles
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
    static int dimTest=0;

    dim2 = 2*(dim+1);

    /*input*/

    itermax = MAX_BOU;
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

    itermax = MAX_BOU;
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

} // namespace helper

} // namespace sofa
