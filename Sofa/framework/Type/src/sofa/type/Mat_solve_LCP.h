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

#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>

namespace sofa::type
{

// -----------------------------------------------------------------
// --- Resoud un LCP ecrit sous la forme U = q + M.F
// ---   dim : dimension du pb
// ---   res[0..dim-1] = U
// ---   res[dim..2*dim-1] = F
// -----------------------------------------------------------------
template <Size dim, typename real>
[[nodiscard]] 
bool solveLCP(const Vec<dim,real> &q, const Mat<dim,dim,real> &M, Vec<dim * 2, real> &res)
{
    constexpr real EPSILON_ZERO  = 0.00001;	        // epsilon pour tests = 0
    constexpr real EPSILON_PIVOT = 0.00000000001;	// epsilon pour pivot
    constexpr Size MAX_NB_LOOP   = 50;	            // nombre maximal de boucles de calcul

    const Size	dim_mult2 = 2 * dim;
    const Size	dim_mult2_plus1 = dim_mult2 + 1;

    Index       ii, jj;
    int         ligPiv;	// ligne du pivot
    int         colPiv;	// colonne du pivot
    double      pivot;	// pivot
    double      min;	// recherche du minimum pour le pivot
    double      coeff;	// valeur du coefficient de la combinaison lineaire
    Index       boucles;	// ii du nombre de passages dans la boucle
    double      mat[dim][dim_mult2_plus1];
    Index       base[dim];		// base des variables non nulles

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
            mat[ii][jj] = -(M(ii,jj - dim));

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
    min = -EPSILON_ZERO;
    for (ii = 0; ii < dim; ii++)
    {
        if (mat[ii][dim_mult2] < min)
        {
            ligPiv = ii;
            min = mat[ii][dim_mult2];
        }
    }

    // tant que tous les q[i] ne sont pas > 0 et qu'on ne boucle pas trop
    while ((ligPiv >= 0) && (boucles < MAX_NB_LOOP))
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
        if (fabs(pivot) < EPSILON_PIVOT)
        {
            printf("  + No solution to LCP + \n");
            boucles = MAX_NB_LOOP;
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
                if (int(ii) != ligPiv)
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
            min = -EPSILON_ZERO;

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
    if (boucles < MAX_NB_LOOP)
    {
        for (ii = 0; ii < dim; ii++)
            res[base[ii]] = mat[ii][dim_mult2];
    }

    return true;
}

} // namespace sofa::type
