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
#include <sofa/helper/LCPcalc.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/logging/Messaging.h>
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
using namespace sofa::helper::system::thread;

LCP::LCP() : maxConst(0), tol(0.00001), numItMax(1000), useInitialF(true), mu(0.0), dim(0)
{

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
        dim = lcp.dim;

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
    delete [] d;
    delete [] f;
    delete [] f_1;
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
    memset(f, 0, (2 * maxConst + 1) * sizeof(double));


}


void LCP::allocate (unsigned int input_maxConst)
{
    this->maxConst = input_maxConst;

    W = new double*[maxConst];
    for (int i = 0; i < (int)maxConst; i++)
    {
        W[i] = new double[maxConst];
        memset(W[i], 0, maxConst * sizeof(double));
    }
    dfree = new double[maxConst];
    d = new double[maxConst];
    f = new double[2 * maxConst + 1];
    f_1= new double[maxConst];

    memset(dfree, 0, maxConst * sizeof(double));
    memset(f, 0, (2 * maxConst + 1) * sizeof(double));
    memset(f_1, 0, maxConst * sizeof(double));

}

void LCP::setLCP(unsigned int input_dim, double *input_dfree, double **input_W, double *input_f, double &input_mu, double &input_tol, int input_numItMax)
{
    dim = input_dim;
    dfree = input_dfree;
    W = input_W;
    f = input_f;
    numItMax = input_numItMax;
    tol = input_tol;
    mu = input_mu;
    maxConst = dim;

    d = new double[maxConst];
    f_1= new double[maxConst];
    memset(d, 0, maxConst * sizeof(double));
    memset(f_1, 0, maxConst * sizeof(double));

}

void LCP::solveNLCP(bool convergenceTest, std::vector<double>* residuals, std::vector<double>* violations)
{
    //double error;
    double f_1[3],dn, ds, dt;
    int numContacts = dim/3;
    const bool computeError = (convergenceTest || residuals);
    for (it=0; it<numItMax; it++)
    {
        error = 0;
        for (int c=0;  c<numContacts ; c++)
        {
            f_1[0] = f[3*c]; f_1[1] = f[3*c+1]; f_1[2] = f[3*c+2];
            dn =dfree[3*c];  dt=dfree[3*c+1]; ds =dfree[3*c+2];
            for (int i=0; i<dim; i++)
            {
                dn += W[3*c  ][i]*f[i];
                dt += W[3*c+1][i]*f[i];
                ds += W[3*c+2][i]*f[i];
            }


            // error measure
            double Ddn, Ddt, Dds;
            Ddn=0; Ddt=0; Dds=0;

            /////// CONTACT
            f[3*c] -= dn / W[3*c  ][3*c  ];
            if (f[3*c]<0)
            {

                if (f_1[0]>0 && computeError)  // the point was in contact and is no more in contact..
                {

                    for (int j=0; j<3; j++ )
                    {
                        Ddn -= W[3*c  ][3*c+j]*f_1[j];
                        Ddt -= W[3*c+1][3*c+j]*f_1[j];
                        Dds -= W[3*c+2][3*c+j]*f_1[j];
                    }
                    error += sqrt(Ddn*Ddn + Ddt*Ddt + Dds*Dds);
                }
                f[3*c  ]=0;
                f[3*c+1]=0;
                f[3*c+2]=0;

                continue;
            }

            ////// FRICTION

            // evaluation of the current tangent positions (motion du to force change along normal)
            dt +=  W[3*c+1][3*c]*(f[3*c]-f_1[0]);
            ds +=  W[3*c+2][3*c]*(f[3*c]-f_1[0]);

            // envaluation of the new fricton forces

            f[3*c+1] -= 2*dt/(W[3*c+1][3*c+1]+W[3*c+2][3*c+2]);
            f[3*c+2] -= 2*ds/(W[3*c+1][3*c+1]+W[3*c+2][3*c+2]);

            double normFt=sqrt(f[3*c+1]*f[3*c+1]+ f[3*c+2]* f[3*c+2]);

            if (normFt > mu*f[3*c])
            {
                f[3*c+1] *=mu*f[3*c]/normFt;
                f[3*c+2] *=mu*f[3*c]/normFt;
            }

            if(computeError)
            {

                for (int j=0; j<3; j++ )
                {
                    Ddn -= W[3*c  ][3*c+j]*(f[3*c+j]-f_1[j]);
                    Ddt -= W[3*c+1][3*c+j]*(f[3*c+j]-f_1[j]);
                    Dds -= W[3*c+2][3*c+j]*(f[3*c+j]-f_1[j]);
                }

                error += sqrt(Ddn*Ddn + Ddt*Ddt + Dds*Dds);
            }
        }
        if (residuals) residuals->push_back(error);
        if (violations)
        {
            double sum_d = 0;
            for (int c=0;  c<numContacts ; c++)
            {
                dn = dfree[3*c];  //dt = dfree[3*c+1]; ds = dfree[3*c+2];
                for (int i=0; i<dim; i++)
                {
                    dn += W[3*c  ][i]*f[i];
                    //dt += W[3*c+1][i]*f[i];
                    //ds += W[3*c+2][i]*f[i];
                }
                if (dn < 0)
                    sum_d += -dn;
            }
            violations->push_back(sum_d);
        }

        if (convergenceTest && error < tol*(numContacts+1))
        {
            return;
        }
    }

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
        double pivot=mat[ligPiv][colPiv];
        // et son affichage
        // printf("pivot=mat[%d][%d]=%f\n\n",ligPiv,colPiv,pivot);

        // si le pivot est nul, le LCP echoue
        if (fabs(pivot)<EPSILON_LCP)
        {
            afficheLCP(q,M,dim);
            printf("*** Pas de solution *** \n");
//            boucles=MAX_BOU;
            result=0;
            for(compteur=0; compteur<dim; compteur++)
            {
                free(mat[compteur]);
            }
            free(mat);
            free(base);
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


void afficheResult(double *f, int dim)
{
    int compteur;
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
        double normFt=sqrt(ft*ft+fs*fs);
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

    if (fn <= 0)
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

/*
typedef struct { double value; int index;} listElem;
struct listSortAscending
{
    bool operator()(const listElem& e1, const listElem& e2)
    {
        return e1.value < e2.value;
    }
};*/


/* Fonctions "de base" à définir pour le MultiGrid :
   - Projection -
   - Calcul à 1 niveau donné -
   - Prolongation -

   Classe
   - NLCP {dim , dfree, W , f

    void NLCPSolve( int numIteration, const double &tol, bool convergenceTest)
    }



   void projection(const NLCP &fineLevel, NLCP &coarseLevel, const std::vector<int> &projectionTable )
   *computation of W
   *computation of dfree

   void prolongation(const NLCP &coarseLevel, NLCP &fineLevel, const std::vector<int> &projectionTable )



*/

/// projection function
/// input values: LCP &fineLevel => LCP at the fine level
///               nbContactsCoarse => number of contacts wanted at the coarse level
///               projectionTable => Table (size = fine level) => for each contact at the fine level, provide the coarse contact
///               verbose =>
/// output values: LCP &coarseLevel
///                contact_is_projected => (size= fine level) => for each contact at the fine level, tell if the contact is projected or not


void projection(LCP &fineLevel, LCP &coarseLevel, int nbContactsCoarse, const std::vector<int> &projectionTable, const std::vector<int> &projectionConstraints, std::vector<double> & projectionValues, std::vector<bool> &contact_is_projected, bool verbose)

{
    // preliminary step: set values to 0

    if (3*nbContactsCoarse > (int) coarseLevel.getMaxConst())
    {
        msg_error("LCPcalc")<<"allocation pb for the coarseLevel. size needed : "<<3*nbContactsCoarse
                           <<" - size allocated : "<<coarseLevel.getMaxConst();
        return;
    }

    for (int c=0;  c<3*nbContactsCoarse ; c++)
    {
        memset(coarseLevel.getW()[c], 0, 3*nbContactsCoarse*sizeof(double));
    }

    memset(coarseLevel.getDfree(), 0, 3*nbContactsCoarse*sizeof(double));
    memset(coarseLevel.getF(), 0, 3*nbContactsCoarse*sizeof(double));
    memset(coarseLevel.getF_1(), 0, 3*nbContactsCoarse*sizeof(double));
    memset(coarseLevel.getD(), 0, 3*nbContactsCoarse*sizeof(double));


    // STEP1 => which contact is being projected ?
    // Only active or interpenetrated ones !!
    int numContactFine = (int)fineLevel.getDim()/3;

    std::vector<int> size_of_group;
    //std::vector<bool> contact_is_projected;
    contact_is_projected.clear();
    std::vector<bool> group_has_projection;
    contact_is_projected.resize(numContactFine);
    group_has_projection.resize(nbContactsCoarse);
    size_of_group.resize(nbContactsCoarse);

    for (int c=0;  c<nbContactsCoarse ; c++)
        group_has_projection[c] = false;

    for (int c1=0; c1<numContactFine; c1++)
    {
        double dn=fineLevel.getDfree()[3*c1];
        for (int i=0; i<(int)fineLevel.getDim(); i++)
        {
            dn += fineLevel.getW()[3*c1  ][i]*fineLevel.getF()[i];
        }
        fineLevel.getD()[3*c1]=dn;

        if (fineLevel.getF()[3*c1] > 0 || dn < 0)  //contact actif uniquement ???
        {
            contact_is_projected[c1]= true;
            size_of_group[projectionTable[c1]] +=1;
            group_has_projection[projectionTable[c1]]=true;
        }
        else
            contact_is_projected[c1]= false;

        //   std::cout<<"contact "<<c1<<" is in group "<< projectionTable[c1]<<std::endl;
    }


    // STEP2
    // For group with no active contact, the closest to the contact one is chosen
    for (int g=0;  g<nbContactsCoarse ; g++)
    {
        if (!group_has_projection[g])
        {
            std::cout<<"WARNING ! NO PROJECTION FOR GROUP "<<g<<" projection of the closest contact"<<std::endl;

            double dmin = 0.0;
            int projected_contact=-1;
            for (int c1=0; c1<numContactFine; c1++)
            {
                if (projectionTable[c1]==g && (projected_contact == -1 || dmin > fineLevel.getD()[3*c1] ))
                {
                    dmin = fineLevel.getD()[3*c1];
                    projected_contact = c1;
                    contact_is_projected[c1]= true;
                }

            }
            if (projected_contact >=0)
            {
                group_has_projection[g]=true;
                size_of_group[g] +=1;
            }
            else
            {
                msg_error("LCPcalc")<<"in nlcp_multiGrid: no projection found for group" << g ;
                return;
            }


        }
    }


    // STEP 3: set up the new coarse LCP
    double** fineW = fineLevel.getW();
    double** coarseW = coarseLevel.getW();
    for (int c1=0; c1<numContactFine; c1++)
    {
        if (contact_is_projected[c1])
        {

            int group = projectionTable[c1];
            int g_n_id = projectionConstraints[3*c1  ]; double g_n_f = projectionValues[3*c1  ];
            int g_t_id = projectionConstraints[3*c1+1]; double g_t_f = projectionValues[3*c1+1];
            int g_s_id = projectionConstraints[3*c1+2]; double g_s_f = projectionValues[3*c1+2];
            ////////////
            // on calcule le système grossier
            ////////////
            coarseLevel.getDfree()[g_n_id] += fineLevel.getDfree()[3*c1  ] * g_n_f;
            coarseLevel.getDfree()[g_t_id] += fineLevel.getDfree()[3*c1+1] * g_t_f;
            coarseLevel.getDfree()[g_s_id] += fineLevel.getDfree()[3*c1+2] * g_s_f;

            coarseLevel.getF_1()[g_n_id] += fineLevel.getF()[3*c1  ] * g_n_f/size_of_group[group];
            coarseLevel.getF_1()[g_t_id] += fineLevel.getF()[3*c1+1] * g_t_f/size_of_group[group];
            coarseLevel.getF_1()[g_s_id] += fineLevel.getF()[3*c1+2] * g_s_f/size_of_group[group];
            coarseLevel.getF()[g_n_id]   += fineLevel.getF()[3*c1  ] * g_n_f/size_of_group[group];
            coarseLevel.getF()[g_t_id]   += fineLevel.getF()[3*c1+1] * g_t_f/size_of_group[group];
            coarseLevel.getF()[g_s_id]   += fineLevel.getF()[3*c1+2] * g_s_f/size_of_group[group];

            for (int c2=0; c2<numContactFine; c2++)
            {
                if (contact_is_projected[c2])
                {
                    //int group2 = projectionTable[c2];
                    int g_n2_id = projectionConstraints[3*c2  ]; double g_n2_f = projectionValues[3*c2  ];
                    int g_t2_id = projectionConstraints[3*c2+1]; double g_t2_f = projectionValues[3*c2+1];
                    int g_s2_id = projectionConstraints[3*c2+2]; double g_s2_f = projectionValues[3*c2+2];

                    coarseW[g_n_id][g_n2_id] += fineW[3*c1  ][3*c2  ]*g_n_f*g_n2_f;   coarseW[g_n_id][g_t2_id] += fineW[3*c1  ][3*c2+1]*g_n_f*g_t2_f;   coarseW[g_n_id][g_s2_id] += fineW[3*c1  ][3*c2+2]*g_n_f*g_s2_f;
                    coarseW[g_t_id][g_n2_id] += fineW[3*c1+1][3*c2  ]*g_t_f*g_n2_f;   coarseW[g_t_id][g_t2_id] += fineW[3*c1+1][3*c2+1]*g_t_f*g_t2_f;   coarseW[g_t_id][g_s2_id] += fineW[3*c1+1][3*c2+2]*g_t_f*g_s2_f;
                    coarseW[g_s_id][g_n2_id] += fineW[3*c1+2][3*c2  ]*g_s_f*g_n2_f;   coarseW[g_s_id][g_t2_id] += fineW[3*c1+2][3*c2+1]*g_s_f*g_t2_f;   coarseW[g_s_id][g_s2_id] += fineW[3*c1+2][3*c2+2]*g_s_f*g_s2_f;
                }
            }
        }
    }

    if(verbose)
    {
        //std::cout<<"LCP at the COARSE LEVEL: "<<std::endl;
        //afficheLCP(coarseLevel.getDfree(), coarseLevel.getW(), coarseLevel.getF(), nbContactsCoarse*3);
    }



}

/// prolongation function
/// all parameters as input
/// output=> change value of F in fineLevel

void prolongation(LCP &fineLevel, LCP &coarseLevel, const std::vector<int> &projectionTable, const std::vector<int> &projectionConstraints, std::vector<double> & projectionValues, std::vector<bool> &contact_is_projected, bool verbose)

{

    int numContactsFine = fineLevel.getDim()/3;

    if (numContactsFine != (int)contact_is_projected.size() || numContactsFine != (int)projectionTable.size() )
    {
        msg_info("LCPcalc")<<"WARNING in prolongation: problem with the size of tables ";
    }

    // STEP 4: PROLONGATION DU RESULTAT AU NIVEAU FIN
    for (int c1=0; c1<numContactsFine; c1++)
    {
        if (contact_is_projected[c1])
        {
            //int group = projectionTable[c1];
            int g_n_id = projectionConstraints[3*c1  ]; double g_n_f = projectionValues[3*c1  ];
            int g_t_id = projectionConstraints[3*c1+1]; double g_t_f = projectionValues[3*c1+1];
            int g_s_id = projectionConstraints[3*c1+2]; double g_s_f = projectionValues[3*c1+2];

            fineLevel.getF()[3*c1  ]  +=  ( coarseLevel.getF()[g_n_id] - coarseLevel.getF_1()[g_n_id] ) * g_n_f;
            fineLevel.getF()[3*c1+1]  +=  ( coarseLevel.getF()[g_t_id] - coarseLevel.getF_1()[g_t_id] ) * g_t_f;
            fineLevel.getF()[3*c1+2]  +=  ( coarseLevel.getF()[g_s_id] - coarseLevel.getF_1()[g_s_id] ) * g_s_f;

            if ( fineLevel.getF()[3*c1  ] < 0)
            {
                fineLevel.getF()[3*c1  ]=0;  fineLevel.getF()[3*c1+1]=0;  fineLevel.getF()[3*c1+2]=0;
            }
        }
    }

    if(verbose)
    {
        // std::cout<<"projection at the finer LEVEL: "<<std::endl;
        // afficheResult(  fineLevel.getF(), fineLevel.getDim());
    }
}

/// new multigrid resolution of a problem with projection & prolongation

int nlcp_multiGrid_2levels(int dim, double *dfree, double**W, double *f, double mu,
        double tol, int numItMax, bool useInitialF,
        std::vector< int> &contact_group, unsigned int num_group,
        std::vector< int> &constraint_group, std::vector<double> &constraint_group_fact,
        bool verbose, std::vector<double>* residuals1, std::vector<double>* residuals2)
{

    LCP *fineLevel = new LCP();
    fineLevel->setLCP(dim,dfree, W, f, mu,tol,numItMax);


    if (!useInitialF)
        memset(fineLevel->getF(), 0, dim*sizeof(double));




    // iterations at the fine Level (no test of convergence)
    bool convergenceTest= false;
    fineLevel->setNumItMax(0);
    fineLevel->solveNLCP(convergenceTest, residuals1);
    if (residuals1 && residuals2) while (residuals2->size() < residuals1->size()) residuals2->push_back(pow(10.0,0.0));

    // projection step & construction of the coarse LCP
    LCP *coarseLevel = new LCP();
    if(verbose)
        std::cout<<"allocation of size"<<num_group<<" at coarse level"<<std::endl;
    coarseLevel->allocate(3*num_group); // allocation of the memory for the coarse LCP
    coarseLevel->setDim(3*num_group);


    std::vector<bool> contact_is_projected;
    projection((*fineLevel), (*coarseLevel), num_group, contact_group, constraint_group, constraint_group_fact, contact_is_projected, verbose);

    // iterations  at the coarse level (till convergence)
    convergenceTest = true;
    coarseLevel->setNumItMax(numItMax);
    coarseLevel->setTol(tol);
    coarseLevel->solveNLCP(convergenceTest, residuals1);
    if (residuals1 && residuals2) while (residuals2->size() < residuals1->size()) residuals2->push_back(pow(10.0,1.0));

    if(verbose)
    {
        std::cout<<"after  "<<coarseLevel->it<<" iteration(s) to solve NLCP at the coarse level: (dim = "<< coarseLevel->getDim()<<") "<<std::endl;
        afficheResult(  coarseLevel->getF(), coarseLevel->getDim());
    }

    // prolongation (interpolation) at the fine level
    prolongation((*fineLevel), (*coarseLevel), contact_group, constraint_group, constraint_group_fact, contact_is_projected, verbose);


    // iterations at the fine level (till convergence)
    convergenceTest = true;
    fineLevel->setNumItMax(1000);
    fineLevel->solveNLCP(convergenceTest, residuals1);
    if (residuals1 && residuals2) while (residuals2->size() < residuals1->size()) residuals2->push_back(pow(10.0,0.0));
    if(verbose)
    {
        std::cout<<"after  "<<fineLevel->it<<" iteration(s) to solve NLCP at the fine Level : (dim = "<< fineLevel->getDim()<<")  error ="<<fineLevel->error<<std::endl;
        afficheResult(  fineLevel->getF(), fineLevel->getDim());
    }

    return 1;

}


int nlcp_multiGrid_Nlevels(int dim, double *dfree, double**W, double *f, double mu, double tol, int numItMax, bool useInitialF, std::vector< std::vector< int> > &contact_group_hierarchy, std::vector<unsigned int> Tab_num_group, std::vector< std::vector< int> > &constraint_group_hierarchy, std::vector< std::vector< double> > &constraint_group_fact_hierarchy, bool verbose, std::vector<double> *residualsN, std::vector<double> *residualLevels, std::vector<double> *violations)
{
    if (dim == 0) return 1; // nothing to do
    std::size_t num_hierarchies = Tab_num_group.size();
    if (num_hierarchies != contact_group_hierarchy.size())
    {
        msg_info("LCPcalc")<<" in nlcp_multiGrid_Nlevels size of Tab_num_group must be equal to size of contact_group_hierarchy";
        return 0;
    }
    if (num_hierarchies != constraint_group_hierarchy.size())
    {
        msg_info("LCPcalc")<<" in nlcp_multiGrid_Nlevels size of Tab_num_group must be equal to size of constraint_group_hierarchy";
        return 0;
    }
    if (num_hierarchies != constraint_group_fact_hierarchy.size())
    {
        msg_info("LCPcalc")<<" in nlcp_multiGrid_Nlevels size of Tab_num_group must be equal to size of constraint_group_fact_hierarchy";
        return 0;
    }

    std::vector<LCP *> hierarchicalLevels;
    hierarchicalLevels.resize(num_hierarchies+1);

    hierarchicalLevels[0] = new LCP(); // finest level !
    hierarchicalLevels[0]->setLCP(dim,dfree, W, f, mu,tol,numItMax);

    if (!useInitialF)
        memset(hierarchicalLevels[0]->getF(), 0, dim*sizeof(double));



    /////////// projection (with few iterations before projection)

    std::vector< std::vector<bool> > contact_is_projected;
    contact_is_projected.resize(num_hierarchies+1);

    bool convergenceTest= false;
    for(std::size_t h = 0; h<num_hierarchies; h++)
    {
        // iterations at the fine Level (no test of convergence)

        hierarchicalLevels[h]->setNumItMax(0);
        hierarchicalLevels[h]->solveNLCP(convergenceTest, residualsN, violations);

        if (residualsN && residualLevels)
            while(residualsN->size() > residualLevels->size())
                residualLevels->push_back(pow(10.0,(double)h));

        // projection step & construction of the coarse LCP
        hierarchicalLevels[h+1] = new LCP();
        if(verbose)
            std::cout<<"Hierarchical level "<<h<<": allocation of size"<<Tab_num_group[h]<<" at coarse level"<<std::endl;
        hierarchicalLevels[h+1]->allocate(3*Tab_num_group[h]); // allocation of the memory for the coarse LCP
        hierarchicalLevels[h+1]->setDim(3*Tab_num_group[h]);

        // call to projection function

        projection((*hierarchicalLevels[h]), (*hierarchicalLevels[h+1]), Tab_num_group[h], contact_group_hierarchy[h], constraint_group_hierarchy[h], constraint_group_fact_hierarchy[h], contact_is_projected[h], verbose);
    }



    // iterations  at the coarse level (till convergence)
    convergenceTest = true;
    hierarchicalLevels[num_hierarchies]->setNumItMax(numItMax);
    hierarchicalLevels[num_hierarchies]->setTol((tol * (dim/3+1))/(hierarchicalLevels[num_hierarchies]->getDim()/3+1));
    hierarchicalLevels[num_hierarchies]->solveNLCP(convergenceTest, residualsN, violations);

    if (residualsN && residualLevels)
        while(residualsN->size() > residualLevels->size())
            residualLevels->push_back(pow(10.0,(double)num_hierarchies));

    if(verbose)
    {
        std::cout<<"after  "<<hierarchicalLevels[num_hierarchies]->it<<" iteration(s) to solve NLCP at the level "<<num_hierarchies<<" : (dim = "<< hierarchicalLevels[num_hierarchies]->getDim()<<") "<<std::endl;
        afficheResult(  hierarchicalLevels[num_hierarchies]->getF(), hierarchicalLevels[num_hierarchies]->getDim());
    }

    for(std::size_t idx = 1 ; idx<=num_hierarchies; idx++)
    {
        std::size_t h = num_hierarchies-idx;
        // prolongation (interpolation) at the fine level
        prolongation((*hierarchicalLevels[h]), (*hierarchicalLevels[h+1]), contact_group_hierarchy[h], constraint_group_hierarchy[h], constraint_group_fact_hierarchy[h], contact_is_projected[h], verbose);

        // iterations at the fine level (till convergence)
        convergenceTest = true;
        hierarchicalLevels[h]->setNumItMax(numItMax);
        //hierarchicalLevels[h]->setTol(tol);
        hierarchicalLevels[h]->setTol((tol * (dim/3+1))/(hierarchicalLevels[h]->getDim()/3+1));
        hierarchicalLevels[h]->solveNLCP(convergenceTest, residualsN, violations);
        if (residualsN && residualLevels)
            while(residualsN->size() > residualLevels->size())
                residualLevels->push_back(pow(10.0,(double)h));

        if(verbose)
        {
            std::cout<<"after  "<<hierarchicalLevels[h]->it<<" iteration(s) to solve NLCP at the Level "<<h<<" : (dim = "<< hierarchicalLevels[h]->getDim()<<")  error ="<<hierarchicalLevels[h]->error<<std::endl;
            afficheResult(  hierarchicalLevels[h]->getF(), hierarchicalLevels[h]->getDim());
        }
    }

    return 1;

}




int nlcp_multiGrid(int dim, double *dfree, double**W, double *f, double mu, double tol, int numItMax, bool useInitialF,
        double** W_coarse, std::vector< int> &contact_group, unsigned int num_group, bool verbose)
{


    msg_info("LCPcalc")<<"entering nlcp_multiGrid fct";

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
    memset(d, 0, dim*sizeof(double));
    // put the vector force to zero
    if (!useInitialF)
        memset(f, 0, dim*sizeof(double));

    msg_info("LCPcalc")<<"step 1 allocation ok";

    // previous value of the force and the displacment
    double f_1[3];
    double d_1[3];




    ////////////////////////////////
    // allocation du système grossier
    ////////////////////////////////
    //double **W_coarse;
    double *d_free_coarse;
    double *F_coarse_1, *F_coarse;
    double *d_coarse;


    // W_coarse = (double **) malloc (3*num_group * sizeof(double*));
    d_free_coarse= (double*) malloc (3*num_group * sizeof(double));
    F_coarse_1 = (double*) malloc (3*num_group * sizeof(double));
    F_coarse= (double*) malloc (3*num_group * sizeof(double));
    d_coarse= (double*) malloc (3*num_group * sizeof(double));
    msg_info("LCPcalc")<<"step 2 allocation ok";

    for (unsigned int g=0;  g<3*num_group ; g++)
    {
        W_coarse[g] = (double*) malloc(3*num_group * sizeof(double));
        memset(W_coarse[g], 0, 3*num_group*sizeof(double));
    }

    memset(d_free_coarse, 0, 3*num_group*sizeof(double));
    memset(F_coarse_1, 0, 3*num_group*sizeof(double));
    memset(F_coarse, 0, 3*num_group*sizeof(double));
    memset(d_coarse, 0, 3*num_group*sizeof(double));
    if(verbose)
    {
        std::cout<<"allocation ok"<<std::endl;
    }
    ////////////////////////////////
    /////////// CALCUL EN V /////////
    ////////////////////////////////

    // STEP1: 3 premières itérations au niveau fin

    double dn, dt, ds, fn, ft, fs;
    // allocation of the inverted system 3x3
    LocalBlock33 **W33;
    W33 = (LocalBlock33 **) malloc (dim*sizeof(LocalBlock33));
    for (c1=0; c1<numContacts; c1++)
        W33[c1] = new LocalBlock33();
    double error = 0;


    for (int it_fin =0; it_fin<0; it_fin++)
    {
        error=0;
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
        }
    }

    if(verbose)
    {
        std::cout<<"initial steps at the finest level "<<std::endl;
        afficheLCP(dfree, W, f, dim);
    }

    // STEP 2: DESCENTE AU NIVEAU GROSSIER => PROJECTION


    std::vector<int> size_of_group;
    std::vector<bool> contact_is_projected;
    std::vector<bool> group_has_projection;
    contact_is_projected.resize(numContacts);
    group_has_projection.resize(num_group);
    size_of_group.resize(num_group);

    for (unsigned int g=0;  g<num_group ; g++)
        group_has_projection[g] = false;




    for (c1=0; c1<numContacts; c1++)
    {
        size_of_group[contact_group[c1]] +=1;
        dn=dfree[3*c1];
        for (i=0; i<dim; i++)
        {
            dn += W[3*c1  ][i]*f[i];
        }
        d[3*c1]=dn;

        if (f[3*c1] > 0 || dn < 0)  //contact actif uniquement ???
        {
            contact_is_projected[c1]= true;
            group_has_projection[contact_group[c1]]=true;
        }
        else
            contact_is_projected[c1]= false;

        //      std::cout<<"contact "<<c1<<" is in group "<< contact_group[c1]<<std::endl;
    }
    std::cout<<"STEP 2, d = "<<std::endl;
    afficheResult(d,dim);


    for (unsigned int g=0;  g<num_group ; g++)
    {
        if (!group_has_projection[g])
        {
            std::cout<<"WARNING ! NO PROJECTION FOR GROUP "<<g<<" projection of the closest contact"<<std::endl;

            double dmin = 9.9e99;
            int projected_contact=-1;
            for (c1=0; c1<numContacts; c1++)
            {
                if (contact_group[c1]==(int)g && dmin > d[3*c1] )
                {
                    dmin = d[3*c1];
                    projected_contact = c1;
                    contact_is_projected[c1]= true;
                }
                if(contact_group[c1]==7)
                    std::cout<<"dmin > d["<<3*c1<<"] (=" << d[3*c1] << ")"<<std::endl;

            }
            if (projected_contact >=0)
            {
                group_has_projection[g]=true;


            }
            else
            {
                msg_error("LCPcalc")<<"in nlcp_multiGrid: no projection found for group" << g;
                free(d_free_coarse);
                free(F_coarse_1);
                free(F_coarse);
                free(d_coarse);

                for (int i = 0; i < numContacts; i++)
                    delete W33[i];
                free(W33);

                return 0;
            }


        }
    }



    for (c1=0; c1<numContacts; c1++)
    {
        if (contact_is_projected[c1])
        {

            int group = contact_group[c1];

            ////////////
            // on calcule le système grossier
            ////////////
            d_free_coarse[3*group  ] += dfree[3*c1  ];
            d_free_coarse[3*group+1] += dfree[3*c1+1];
            d_free_coarse[3*group+2] += dfree[3*c1+2];

            F_coarse_1[3*group  ] += f[3*c1  ]/size_of_group[group];
            F_coarse_1[3*group+1] += f[3*c1+1]/size_of_group[group];
            F_coarse_1[3*group+2] += f[3*c1+2]/size_of_group[group];
            F_coarse[3*group  ] += f[3*c1  ]/size_of_group[group];
            F_coarse[3*group+1] += f[3*c1+1]/size_of_group[group];
            F_coarse[3*group+2] += f[3*c1+2]/size_of_group[group];


            for (int c2=0; c2<numContacts; c2++)
            {
                if (contact_is_projected[c2])
                {
                    int group2 = contact_group[c2];
                    W_coarse[3*group  ][3*group2] += W[3*c1  ][3*c2];   W_coarse[3*group  ][3*group2+1] += W[3*c1  ][3*c2+1];   W_coarse[3*group  ][3*group2+2] += W[3*c1  ][3*c2+2];
                    W_coarse[3*group+1][3*group2] += W[3*c1+1][3*c2];   W_coarse[3*group+1][3*group2+1] += W[3*c1+1][3*c2+1];   W_coarse[3*group+1][3*group2+2] += W[3*c1+1][3*c2+2];
                    W_coarse[3*group+2][3*group2] += W[3*c1+2][3*c2];   W_coarse[3*group+2][3*group2+1] += W[3*c1+2][3*c2+1];   W_coarse[3*group+2][3*group2+2] += W[3*c1+2][3*c2+2];

                }

            }


        }
    }

    if(verbose)
    {
        std::cout<<"LCP at the COARSE LEVEL: "<<std::endl;
        afficheLCP(d_free_coarse, W_coarse, F_coarse,num_group*3);
    }


    // STEP 3: CALCUL GS AU NIVEAU GROSSIER !!
    int dim_coarse = 3*num_group;
    for (it=0; it<numItMax; it++)
    {
        error =0;
        for (unsigned int g1=0;  g1<num_group ; g1++)
        {
            f_1[0] = F_coarse[3*g1]; f_1[1] = F_coarse[3*g1+1]; f_1[2] = F_coarse[3*g1+2];
            dn =d_free_coarse[3*g1];  dt=d_free_coarse[3*g1+1]; ds =d_free_coarse[3*g1+2];
            for (i=0; i<dim_coarse; i++)
            {
                dn += W_coarse[3*g1  ][i]*F_coarse[i];
                dt += W_coarse[3*g1+1][i]*F_coarse[i];
                ds += W_coarse[3*g1+2][i]*F_coarse[i];
            }

            // error measure
            double Ddn, Ddt, Dds;
            Ddn=0; Ddt=0; Dds=0;

            /////// CONTACT
            F_coarse[3*g1] -= dn / W_coarse[3*g1  ][3*g1  ];
            if (F_coarse[3*g1]<0)
            {

                if (f_1[0]>0)  // the point was in contact and is no more in contact..
                {

                    for (int j=0; j<3; j++ )
                    {
                        Ddn -= W_coarse[3*g1  ][3*g1+j]*f_1[j];
                        Ddt -= W_coarse[3*g1+1][3*g1+j]*f_1[j];
                        Dds -= W_coarse[3*g1+2][3*g1+j]*f_1[j];
                    }
                    error += sqrt(Ddn*Ddn + Ddt*Ddt + Dds*Dds);
                }
                F_coarse[3*g1  ]=0;
                F_coarse[3*g1+1]=0;
                F_coarse[3*g1+2]=0;

                continue;
            }

            ////// FRICTION

            // evaluation of the current tangent positions (motion du to force change along normal)
            dt +=  W_coarse[3*g1+1][3*g1]*(F_coarse[3*g1]-f_1[0]);
            ds +=  W_coarse[3*g1+2][3*g1]*(F_coarse[3*g1]-f_1[0]);

            // envaluation of the new fricton forces

            F_coarse[3*g1+1] -= 2*dt/(W_coarse[3*g1+1][3*g1+1]+W_coarse[3*g1+2][3*g1+2]);
            F_coarse[3*g1+2] -= 2*ds/(W_coarse[3*g1+1][3*g1+1]+W_coarse[3*g1+2][3*g1+2]);

            double normFt=sqrt(F_coarse[3*g1+1]*F_coarse[3*g1+1]+ F_coarse[3*g1+2]* F_coarse[3*g1+2]);

            if (normFt > mu*F_coarse[3*g1])
            {
                F_coarse[3*g1+1] *=mu*F_coarse[3*g1]/normFt;
                F_coarse[3*g1+2] *=mu*F_coarse[3*g1]/normFt;
            }

            for (int j=0; j<3; j++ )
            {
                Ddn -= W_coarse[3*g1  ][3*g1+j]*(F_coarse[3*g1+j]-f_1[j]);
                Ddt -= W_coarse[3*g1+1][3*g1+j]*(F_coarse[3*g1+j]-f_1[j]);
                Dds -= W_coarse[3*g1+2][3*g1+j]*(F_coarse[3*g1+j]-f_1[j]);
            }

            error += sqrt(Ddn*Ddn + Ddt*Ddt + Dds*Dds);
        }

        if (error < tol*(numContacts+1))
        {
            continue;
        }
    }

    if(verbose)
    {
        std::cout<<"Result at the COARSE LEVEL: "<<std::endl;
        afficheResult( F_coarse,num_group*3);
    }


    // STEP 4: PROLONGATION DU RESULTAT AU NIVEAU FIN
    for (c1=0; c1<numContacts; c1++)
    {
        if (contact_is_projected[c1])
        {
            int group = contact_group[c1];
            f[3*c1  ]  +=  ( F_coarse[3*group]   - F_coarse_1[3*group] );
            f[3*c1+1]  +=  ( F_coarse[3*group+1] - F_coarse_1[3*group+1] );
            f[3*c1+2]  +=  ( F_coarse[3*group+2] - F_coarse_1[3*group+2] );

            if (f[3*c1  ] < 0)
            {
                f[3*c1  ]=0; f[3*c1+1]=0; f[3*c1+2]=0;
            }
        }
    }

    if(verbose)
    {
        std::cout<<"projection at the finer LEVEL: "<<std::endl;
        afficheResult( f,dim);
    }


    // STEP 5: RELAXATION AU NIVEAU FIN : 10 iterations
    int result =  nlcp_gaussseidel(dim, dfree, W, f,  mu,  tol, 10 , true, true);


    if(verbose)
    {
        std::cout<<"after 10 iteration at the finer LEVEL: "<<std::endl;
        afficheResult( f,dim);
    }

    free(d_free_coarse);
    free(F_coarse_1);
    free(F_coarse);
    free(d_coarse);

    for (int i = 0; i < numContacts; i++)
        delete W33[i];
    free(W33);

    return result;



}

int nlcp_gaussseidel(int dim, double *dfree, double**W, double *f, double mu, double tol, int numItMax, bool useInitialF, bool verbose, double minW, double maxF, std::vector<double>* residuals, std::vector<double>* violations)

{
    const int numContacts =  dim/3;

    if (dim % 3)
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
    // double d_1b[3] = {0.0, 0.0, 0.0};
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
            set3Dof(f,index1,0.0,0.0,0.0); //		f[3*index] = 0.0; f[3*index+1] = 0.0; f[3*index+2] = 0.0;

            // computation of actual d due to contribution of other contacts
            dn=dfree[3*index1]; dt=dfree[3*index1+1]; ds=dfree[3*index1+2];
            for (i=0; i<dim; i++)
            {
                dn += W[3*index1  ][i]*f[i];
                dt += W[3*index1+1][i]*f[i];
                ds += W[3*index1+2][i]*f[i];
            }
            // if (maxF != 0.0)
            // {
            //     d_1b[0] = dn;
            //     d_1b[1] = dt;
            //     d_1b[2] = ds;
            // }
            d_1[0] = dn + W[3*index1  ][3*index1  ]*f_1[0]+W[3*index1  ][3*index1+1]*f_1[1]+W[3*index1  ][3*index1+2]*f_1[2];
            d_1[1] = dt + W[3*index1+1][3*index1  ]*f_1[0]+W[3*index1+1][3*index1+1]*f_1[1]+W[3*index1+1][3*index1+2]*f_1[2];
            d_1[2] = ds + W[3*index1+2][3*index1  ]*f_1[0]+W[3*index1+2][3*index1+1]*f_1[1]+W[3*index1+2][3*index1+2]*f_1[2];
            if (minW != 0.0 && fabs(W[3*index1  ][3*index1  ]) <= minW)
            {
                // constraint compliance is too small
                if (it == 0) std::cout<<"NLCP WARNING: compliance too small for contact " << index1 << ": |" << std::scientific << W[3*index1  ][3*index1  ] << "| < " << minW << std::fixed << std::endl;

                fn=0; ft=0; fs=0;
            }
            else
            {
                if(W33[index1]->computed==false)
                {
                    W33[index1]->compute(W[3*index1][3*index1],W[3*index1][3*index1+1],W[3*index1][3*index1+2],
                            W[3*index1+1][3*index1+1], W[3*index1+1][3*index1+2],W[3*index1+2][3*index1+2]);
                }

                fn=f_1[0]; ft=f_1[1]; fs=f_1[2];
                W33[index1]->GS_State(mu,dn,dt,ds,fn,ft,fs);
                //W33[index1]->BiPotential(mu,dn,dt,ds,fn,ft,fs);
                // if (maxF != 0.0 && it >= 3 && fabs(fn) >= maxF)
                // { // constraint force is too large
                //     std::cout<<"NLCP WARNING: force too large for contact " << index1 << " at iteration " << it << ": |" << std::scientific << fn << "| > " << maxF << std::fixed << std::endl;
                //     fn=0; ft=0; fs=0;
                //     dn = d_1b[0];
                //     dt = d_1b[1];
                //     ds = d_1b[2];
                // }
            }
            error += absError(dn,dt,ds,d_1[0],d_1[1],d_1[2]);


            set3Dof(f,index1,fn,ft,fs);

        }
        if (residuals) residuals->push_back(error);
        if (violations)
        {
            double sum_d = 0;
            for (int c=0;  c<numContacts ; c++)
            {
                dn = dfree[3*c];  //dt = dfree[3*c+1]; ds = dfree[3*c+2];
                for (int i=0; i<dim; i++)
                {
                    dn += W[3*c  ][i]*f[i];
                    //dt += W[3*c+1][i]*f[i];
                    //ds += W[3*c+2][i]*f[i];
                }
                if (dn < 0)
                    sum_d += -dn;
            }
            violations->push_back(sum_d);
        }

        if (error < tol*(numContacts+1))
        {


            if (maxF != 0.0)
            {
                for (c1=0; c1<numContacts; c1++)
                {
                    // index of contact
                    int index1 = c1;
                    if (fabs(f[3*index1]) >= maxF)
                    {
                        // constraint force is too large
                        std::cout<<"NLCP WARNING: force too large for contact " << index1 << " : |" << std::scientific << f[3*index1] << "| > " << maxF << std::fixed << std::endl;
                        f[3*index1  ] = 0;
                        f[3*index1+1] = 0;
                        f[3*index1+2] = 0;
                    }
                }
            }

            free(d);
            for (int i = 0; i < numContacts; i++)
                delete W33[i];
            free(W33);
            if (verbose)
                printf("Convergence after %d iteration(s) with tolerance : %f and error : %f with dim : %d\n",it, tol, error, dim);
            //afficheLCP(dfree,W,f,dim);
            sofa::helper::AdvancedTimer::valSet("GS iterations", it+1);
            return 1;
        }
    }
    sofa::helper::AdvancedTimer::valSet("GS iterations", it);

    free(d);
    for (int i = 0; i < numContacts; i++)
        delete W33[i];
    free(W33);

    if (verbose)
    {
        msg_warning("LCPcalc")<<"No convergence in  nlcp_gaussseidel function : error ="<<error <<" after"<< it<<" iterations";
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

    ctime_t t0 = CTime::getTime();
    ctime_t tdiff = (ctime_t)(timeout*CTime::getTicksPerSec());

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

            ctime_t t1 = CTime::getTime();
            if((t1-t0) > tdiff)
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
            sofa::helper::AdvancedTimer::valSet("GS iterations", it+1);
            return 1;
        }
    }
    sofa::helper::AdvancedTimer::valSet("GS iterations", it);
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
void gaussSeidelLCP1(int dim, FemClipsReal * q, FemClipsReal ** M, FemClipsReal * res, double tol, int numItMax, double minW, double maxF, std::vector<double>* residuals)
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

            if (minW != 0.0 && fabs(M[compteur2][compteur2]) <= minW)
            {
                // constraint compliance is too small
                if (compteur == 0) std::cout<<"LCP WARNING: compliance too small for constraint " << compteur2 << ": |" << std::scientific << M[compteur2][compteur2] << "| < " << minW << std::fixed << std::endl;
                res[dim+compteur2]=(FemClipsReal)0.0;
            }
            else if (res[compteur2]<0)
            {
                res[dim+compteur2]=-res[compteur2]/M[compteur2][compteur2];
                // if (maxF != 0.0 && compteur >= 3 && fabs(res[dim+compteur2]) >= maxF)
                // { // constraint force is too large
                //     std::cout<<"LCP WARNING: force too large for constraint " << compteur2 << " at iteration " << compteur << " : |" << std::scientific << res[dim+compteur2] << "| > " << maxF << std::fixed << std::endl;
                //     res[dim+compteur2]=(FemClipsReal)0.0;
                // }
            }
            else
            {
                res[dim+compteur2]=(FemClipsReal)0.0;
            }

            error +=fabs( M[compteur2][compteur2] * (res[dim+compteur2] - f_1) );


        }
        if (residuals) residuals->push_back(error);
        if (error < tol)
        {
            //	std::cout << "convergence in gaussSeidelLCP1 with " << compteur << " iterations\n";
            break;
        }

    }
    sofa::helper::AdvancedTimer::valSet("GS iterations", (compteur < numItMax) ? compteur+1 : compteur);

    if (maxF != 0.0)
    {
        for (compteur2=0; compteur2<dim; compteur2++)
        {
            if (fabs(res[dim+compteur2]) >= maxF)
            {
                // constraint force is too large
                std::cout<<"LCP WARNING: force too large for constraint " << compteur2 << " : |" << std::scientific << res[dim+compteur2] << "| > " << maxF << std::fixed << std::endl;
                res[dim+compteur2]=(FemClipsReal)0.0;
            }
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
