/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_HELPER_LCPSOLVER_INL
#define SOFA_HELPER_LCPSOLVER_INL

#include <sofa/helper/LCPSolver.h>

#ifdef PS3
#include <stdio.h>
#else
#include <cstdio>		// fopen and friends
#endif

namespace sofa
{

namespace helper
{

#define EPS     0.00001	// epsilon pour tests = 0
#define EPSP    0.00000000001	// epsilon pour pivot
#define MAX_BOU 50	// nombre maximal de boucles de calcul

// -----------------------------------------------------------------
// --- Resoud un LCP ecrit sous la forme U = q + M.F
// ---   dim : dimension du pb
// ---   res[0..dim-1] = U
// ---   res[dim..2*dim-1] = F
// -----------------------------------------------------------------
template <int dim>
bool LCPSolver<dim>::solve(const double *q, const Matrix &M, double *res)
{
    const int	dim_mult2 = 2 * dim;
    const int	dim_mult2_plus1 = dim_mult2 + 1;

    int         ii, jj;
    int         ligPiv;	// ligne du pivot
    int         colPiv;	// colonne du pivot
    double      pivot;	// pivot
    double      min;	// recherche du minimum pour le pivot
    double      coeff;	// valeur du coefficient de la combinaison lineaire
    int         boucles;	// ii du nombre de passages dans la boucle
    double      mat[dim][dim_mult2_plus1];
    int         base[dim];		// base des variables non nulles

    // matrix initialization
    for (ii = 0; ii < dim; ii++)
    {
        // colonnes correspondantes a w
        for (jj = 0; jj < dim; jj++)
        {
            if (jj == ii)
                mat[ii][jj] = 1;
            else
                mat[ii][jj] = 0;
        }

        // colonnes correspondantes a z
        for (; jj < dim_mult2; jj++)
            mat[ii][jj] = -(M[ii][jj - dim]);

        // colonne correspondante a q
        mat[ii][jj] = q[ii];
    }

    // initialisation de la base
    for (ii = 0; ii < dim; ii++)
        base[ii] = ii;

    // initialisation du nombre de boucles
    boucles = 0;

    // recherche de la ligne du pivot
    ligPiv = -1;
    min = -EPS;
    for (ii = 0; ii < dim; ii++)
    {
        if (mat[ii][dim_mult2] < min)
        {
            ligPiv = ii;
            min = mat[ii][dim_mult2];
        }
    }

    // tant que tous les q[i] ne sont pas > 0 et qu'on ne boucle pas trop
    while ((ligPiv >= 0) && (boucles < MAX_BOU))
    {
        // augmentation du nombre de passages dans cette boucle
        boucles++;

        // recherche de la colonne du pivot
        if (base[ligPiv] < dim)
        {
            // c'est un wi dans la base
            colPiv = dim + base[ligPiv];
        }
        else
        {
            // c'est un zi dans la base
            colPiv = base[ligPiv] - dim;
        }

        // stockage de la valeur du pivot
        pivot = mat[ligPiv][colPiv];

        // si le pivot est nul, le LCP echoue
        if (fabs(pivot) < EPSP)
        {
            printf("  + No solution to LCP + \n");
            boucles = MAX_BOU;
            return false;
        }
        else
        {
            // division de la ligne du pivot par le pivot
            for (ii = 0; ii < dim_mult2_plus1; ii++)
                mat[ligPiv][ii] /= pivot;

            // combinaisons lineaires mettant la colonne du pivot a 0
            for (ii = 0; ii < dim; ii++)
            {
                if (ii != ligPiv)
                {
                    coeff = mat[ii][colPiv];
                    for (jj = 0; jj < dim_mult2_plus1; jj++)
                        mat[ii][jj] -= coeff * mat[ligPiv][jj];
                }
            }

            // on rentre dans la base la nouvelle variable
            base[ligPiv] = colPiv;

            // recherche de la nouvelle ligne du pivot
            ligPiv = -1;
            min = -EPS;

            for (ii = 0; ii < dim; ii++)
            {
                if (mat[ii][dim_mult2] < min)
                {
                    ligPiv = ii;
                    min = mat[ii][dim_mult2];
                }
            }
        }
    }

    // stockage du resultat
    for (ii = 0; ii < dim_mult2; ii++)
        res[ii] = 0;

    // si on est arrive a resoudre le pb, seules les variables en base sont non nulles
    if (boucles < MAX_BOU)
    {
        for (ii = 0; ii < dim; ii++)
            res[base[ii]] = mat[ii][dim_mult2];
    }

    return true;
}


// -----------------------------------------------------------------
// ---
// -----------------------------------------------------------------
template <int dim> void LCPSolver<dim>::printInfo(double *q, Matrix &M)
{
    int ii, jj;

    // affichage de la matrice du LCP
    printf("M = [");
    for(ii=0; ii<dim; ii++)
    {
        for(jj=0; jj<dim; jj++)
        {
            printf("\t%.4f",M[ii][jj]);
        }
        printf("\n");
    }
    printf("      ]\n\n");

    // affichage de q
    printf("q = [");
    for (ii=0; ii<dim; ii++)
    {
        printf("\t%.4f\n",q[ii]);
    }
    printf("      ]\n\n");

    // afficahge base courante
    /*	printf("B = [");
    	for(ii=0;ii<dim;ii++) {
    		printf("\t%d",base[ii]);
    	}
    	printf("\t]\n\n");

    	// affichage matrice courante
    	printf("mat = [");
    	for(ii=0;ii<dim;ii++) {
    		for(jj=0;jj<2*dim+1;jj++) {
    			printf("\t%.4f",mat[ii][jj]);
    		}
    		printf("\n");
    	}
    	printf("      ]\n\n"); */
}

//template<> class LCPSolver<3>;
//template<> class LCPSolver<5>;

} // namespace helper

} // namespace sofa

#endif
