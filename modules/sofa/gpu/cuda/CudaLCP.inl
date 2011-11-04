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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
int CudaLCP_MultVector_ResSize(unsigned int dim)
{
    return (dim+BSIZE-1)/BSIZE;
}

int CudaNLCP_MultVector_ResSize(unsigned int dim)
{
    return (dim+MBSIZE-1)/MBSIZE;
}

///////////////////////////////////////1er version

void CudaLCP_MultVectorf(int dim,int index, const void * m,const void * f,void * r)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    CudaLCP_MultVector_kernel<float><<< grid, threads, threads.x*sizeof(float) >>>(dim, index, (const float*)m, (const float*)f, (float*)r, BSIZE/2);
}
void CudaLCP_MultVectord(int dim,int index, const void * m,const void * f,void * r)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    CudaLCP_MultVector_kernel<double><<< grid, threads, threads.x*sizeof(double) >>>(dim, index, (const double*)m, (const double*)f, (double*)r, BSIZE/2);
#endif
}

void CudaLCP_ComputeErrorf(int compteur2,int sizeTmp, const void * tmp, const void * M,const void * q,void * f,void * error)
{
    dim3 threads(sizeTmp,1);
    dim3 grid(1,1);
    int offset;
    if (sizeTmp==1) offset = 0;
    else
    {
        offset = 1;
        while (offset*2 < sizeTmp)	offset *= 2;
    }

    CudaLCP_ComputeError_kernel<float><<< grid, threads, threads.x*sizeof(float) >>>(compteur2,(const float*)tmp, (const float*)M,(const float*)q, (float*)f,(float*)error,offset);
}
void CudaLCP_ComputeErrord(int compteur2,int sizeTmp, const void * tmp, const void * M,const void * q,void * f,void * error)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(sizeTmp,1);
    dim3 grid(1,1);
    int offset;
    if (sizeTmp==1) offset = 0;
    else
    {
        offset = 1;
        while (offset*2 < sizeTmp)	offset *= 2;
    }

    CudaLCP_ComputeError_kernel<double><<< grid, threads, threads.x*sizeof(double) >>>(compteur2,(const double*)tmp, (const double*)M,(const double*)q, (double*)f,(double*)error,offset);
#endif
}

//////////////////////////////2em version

/*
for (int i=0;i<dim;i++) {
	cuda_res[i] = cuda_q[i];
	for (int j=0;j<dim;j++) {
		if (j>i) cuda_res[i] += cuda_M[i][j] * cuda_f[j];
}
}
*/
void CudaLCP_MultIndepf(int dim,const void * m,int pM,const void * f,void * tmp,int pTmp)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,dim);

    CudaLCP_MultIndep_kernel<float><<< grid, threads,0>>>(dim, (const float*)m,pM,(const float*)f,(float*)tmp,pTmp,BSIZE/2);
}
void CudaLCP_MultIndepd(int dim,const void * m,int pM,const void * f,void * tmp,int pTmp)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,dim);

    CudaLCP_MultIndep_kernel<double><<< grid, threads,0>>>(dim, (const double*)m,pM,(const double*)f,(double*)tmp,pTmp,BSIZE/2);
#endif
}

/*
for (int j=0;j<dim;j++) {
	cuda_res[j] = cuda_q[i];
	for (int i=0;i<tmpsize;i++) {
		cuda_res[j] += cuda_tmp[i][j];
}
}
*/
void CudaLCP_AddIndepf(int dim,int tmpsize,const void * tmp,int pTmp,void * res)
{
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaLCP_AddIndep_kernel<float><<< grid, threads,threads.x*sizeof(float)>>>(tmpsize,(const float*)tmp,pTmp,(float*)res);
}
void CudaLCP_AddIndepd(int dim,int tmpsize,const void * tmp,int pTmp,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaLCP_AddIndep_kernel<double><<< grid, threads,threads.x*sizeof(double)>>>(tmpsize,(const double*)tmp,pTmp,(double*)res);
#endif
}

void CudaLCP_AddIndepAndUpdatef(int dim,int tmpsize,const void * m,const void * q,const void * tmp,int pTmp,void * f,void * res,void * err)
{
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaLCP_AddIndepAndUpdate_kernel<float><<< grid, threads,threads.x*sizeof(float)>>>(dim,tmpsize,(const float*)m,(const float*)q,(const float*)tmp,pTmp,(float*)f,(float*)res,(float*)err,tmpsize);
}
void CudaLCP_AddIndepAndUpdated(int dim,int tmpsize,const void * m,const void * q,const void * tmp,int pTmp,void * f,void * res,void * err)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaLCP_AddIndepAndUpdate_kernel<double><<< grid, threads,threads.x*sizeof(double)>>>(dim,tmpsize,(const double*)m,(const double*)q,(const double*)tmp,pTmp,(double*)f,(double*)res,(double*)err,tmpsize);
#endif
}

/*
float f_1 = cuda_f[compteur2];
if (cuda_res[compteur2]<0) cuda_f[compteur2] = -cuda_res[compteur2]/cuda_M[compteur2][compteur2];
else cuda_f[compteur2]=0.0;
cuda_err[0] += fabs(cuda_M[compteur2][compteur2] * (cuda_f[compteur2] - f_1));
cuda_err[1] = f_1;

for (int compteur3=0;compteur3<dim;compteur3++) {
	if (compteur3!=compteur2) cuda_res[compteur3] += cuda_M[compteur3][compteur2] * (cuda_f[compteur2] - cuda_err[1]);
}
*/
void CudaLCP_ComputeNextIter_V2f(int dim,int compteur2,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    CudaLCP_ComputeNextIter_kernel_V2<float><<< grid, threads,0 >>>(dim,compteur2,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
void CudaLCP_ComputeNextIter_V2d(int dim,int compteur2,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    CudaLCP_ComputeNextIter_kernel_V2<double><<< grid, threads,0 >>>(dim,compteur2,(const double*) m,mP,(const double *) q,(double*)f, (double*) err,(double *)res);
#endif
}

/////////////////////////////////3em version

void CudaLCP_ComputeNextIter_V3_OneKernelf(int dim,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(dim,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V3_OneKernel_kernel<float><<< grid, threads,0 >>>(dim,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
void CudaLCP_ComputeNextIter_V3_OneKerneld(int dim,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(dim,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V3_OneKernel_kernel<double><<< grid, threads,0 >>>(dim,(const double*) m,mP,(const double *) q,(double*)f, (double*) err,(double *)res);
#endif
}

////////////////////////////4em version

/*
for (int block=0; block<d; block++) {
	int compteur2 = debutblock+block;

	float f_1 = cuda_f[compteur2];

	float f_2;
	if (cuda_res[compteur2]<0) f_2 = -cuda_res[compteur2]/cuda_M[compteur2][compteur2];
	else f_2=0.0;

	cuda_f[compteur2] = f_2;
	cuda_res[compteur2] = cuda_q[compteur2];
	cuda_err[0] += fabs(cuda_M[compteur2][compteur2] * (f_2 - f_1));

	for (int compteur3=debutblock;compteur3<debutblock+d;compteur3++) {
		if (compteur3!=compteur2) cuda_res[compteur3] += cuda_M[compteur2][compteur3] * f_2;
}
}
*/
void CudaLCP_ComputeNextIter_V4_DepKernelf(int d,int debutblock,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(BSIZE,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V4_DepKernel_kernel<float><<< grid, threads,0>>>(d,debutblock,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
void CudaLCP_ComputeNextIter_V4_DepKerneld(int d,int debutblock,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V4_DepKernel_kernel<double><<< grid, threads,0>>>(d,debutblock,(const double*) m,mP,(const double *) q,(double*)f, (double*) err,(double *)res);
#endif
}

/*
for (int compteur4=0;compteur4<dim-d;compteur4++) {
	int ligne = compteur4;
	if (compteur4>=debutblock) ligne+=BSIZE;

	for (int k=debutblock;k<debutblock+d;k++) {
		cuda_res[ligne]+= cuda_M[k][ligne] * cuda_f[k];
}
}
*/
void CudaLCP_ComputeNextIter_V4_InDepKernelf(int dim,int d,int debutBlock,const void * m,int mP,const void * f,void * res)
{
    dim3 threads(BSIZE,1);
    dim3 grid(1,V4_MAX_LINE);

    CudaLCP_ComputeNextIter_V4_InDepKernel_kernel<float><<< grid, threads,0>>>(dim,debutBlock,(const float*) m,mP,(const float*)f, (float *)res,BSIZE/2);
}
void CudaLCP_ComputeNextIter_V4_InDepKerneld(int dim,int d,int debutBlock,const void * m,int mP,const void * f,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid(1,V4_MAX_LINE);

    CudaLCP_ComputeNextIter_V4_InDepKernel_kernel<double><<< grid, threads,0>>>(dim,debutBlock,(const double*) m,mP,(const double*)f, (double *)res,BSIZE/2);
#endif
}

////////////////////////////////////////5em version

/*
for (int compteur2=0; compteur2<BSIZE;compteur2++) {	//calcule le bloc en haut a gauche pour lancer le calcul
	float f_2;
	if (cuda_res[compteur2]<0) f_2 = -cuda_res[compteur2]/cuda_M[compteur2][compteur2];
	else f_2=0.0;

	cuda_f[compteur2] = f_2;
	cuda_res[compteur2] = cuda_q[compteur2];

	for (int compteur3=0;compteur3<BSIZE;compteur3++) {
		if (compteur3!=compteur2) cuda_res[compteur3] += cuda_M[compteur3][compteur2] * f_2;
}
}
*/
void CudaLCP_ComputeNextIter_V5_FirstKernelf(const void * m,int mP,const void * q,void * f,void * res)
{
    dim3 threads(BSIZE,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V5_FirstKernel_kernel<float><<< grid, threads,0 >>>((const float*) m,mP,(const float *) q,(float*)f,(float *)res);

}
void CudaLCP_ComputeNextIter_V5_FirstKerneld(const void * m,int mP,const void * q,void * f,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V5_FirstKernel_kernel<double><<< grid, threads,0 >>>((const double*) m,mP,(const double *) q,(double*)f,(double *)res);
#endif
}

/*
for (int j=0;j<d;j++) {
	int compteur2=ligne+j;//ligne du bloc en cours

	float acc=0.0;

	for (int index1=0;index1<BSIZE;index1++) {//calcul du 1er bloc independant
		int dc = index1+colone;
		if (dc<dim) acc += cuda_M[dc][compteur2] * cuda_f[dc];
}

	float r_calc = cuda_res[compteur2]+acc;
	float f_1 = cuda_f[compteur2];

	float f_2;
	if (r_calc<0) f_2 = -r_calc/cuda_M[compteur2][compteur2];
	else f_2=0.0;

	cuda_f[compteur2] = f_2;
	cuda_res[compteur2] = cuda_q[compteur2];
	cuda_err[0] += fabs(cuda_M[compteur2][compteur2] * (f_2 - f_1));

	for (int index1=0;index1<BSIZE;index1++) {//calcul du 1er bloc independant
		int compteur3 = ligne+index1;
		if ((compteur3<dim) && (compteur3!=compteur2)) cuda_res[compteur3] += cuda_M[compteur2][compteur3] * f_2;
}
}

for (int j=0;j<nbth;j++) {
	int dl=j;
	if (ligne-BSIZE<0) dl+=BSIZE;
	else if (j>=ligne-BSIZE) dl+=2*BSIZE;

	for (int i=0;i<BSIZE;i++) {
		int dc = colone+i;
		if (dc<dim) cuda_res[dl]+= cuda_M[dc][dl] * cuda_f[dc];
}
}
*/
void CudaLCP_ComputeNextIter_V5_SecondKernelf(int dim,int nbth,int d,int ligne,int colone,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(BSIZE,1);
    dim3 grid(1,nbth+1);

    CudaLCP_ComputeNextIter_V5_SecondKernel_kernel<float><<< grid, threads,0 /*,threads.x*sizeof(float)*/ >>>(dim,d,ligne,colone,(const float*) m,mP,(const float *) q,(float*)f,(float*)err,(float *)res,BSIZE/2);
}
void CudaLCP_ComputeNextIter_V5_SecondKerneld(int dim,int nbth,int d,int ligne,int colone,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid(1,nbth+1);

    CudaLCP_ComputeNextIter_V5_SecondKernel_kernel<double><<< grid, threads,0 /*,threads.x*sizeof(double)*/ >>>(dim,d,ligne,colone,(const double*) m,mP,(const double *) q,(double*)f,(double*)err,(double *)res,BSIZE/2);
#endif
}

///////////////////////////////////////////////////6em version

void CudaLCP_FullKernel_V6f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    dim3 threads(BSIZE_C,1);
    dim3 grid(1,NB_MULTIPROC);

    CudaLCP_FullKernel_V6_kernel<<< grid, threads,threads.x*threads.y*sizeof(float)>>>(dim,dim*itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);

}
void CudaLCP_FullKernel_V6d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE_C,1);
    dim3 grid(1,NB_MULTIPROC);

    CudaLCP_FullKernel_V6_kernel<<< grid, threads,threads.x*threads.y*sizeof(double)>>>(dim,dim*itMax,(double)tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);
#endif
}

/////////////////////////////////////////////////7 em version

void CudaLCP_FullKernel_V7f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    dim3 threads(BSIZE_C,BSIZE_L);
    dim3 grid(1,NB_MULTIPROC);
    int dim_n = (dim+BSIZE_L-1)/BSIZE_L * BSIZE_L;

    CudaLCP_FullKernel_V7_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaLCP_FullKernel_V7d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE_C,BSIZE_L);
    dim3 grid(1,NB_MULTIPROC);
    int dim_n = (dim+BSIZE_L-1)/BSIZE_L * BSIZE_L;

    CudaLCP_FullKernel_V7_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,(double)tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

#endif
}

//////////////////version 8

void CudaLCP_FullKernel_V8f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{

    dim3 threads(V8_BSIZE,V8_BSIZE);
    dim3 grid(1,V8_NBPROC);
    int dim_n = (dim+V8_BSIZE-1)/V8_BSIZE * V8_BSIZE;

    CudaLCP_FullKernel_V8_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaLCP_FullKernel_V8d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(V8_BSIZE,V8_BSIZE);
    dim3 grid(1,V8_NBPROC);
    int dim_n = (dim+V8_BSIZE-1)/V8_BSIZE * V8_BSIZE;

    CudaLCP_FullKernel_V8_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,(double)tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

#endif
}

//////////////////version 9
void CudaLCP_FullKernel_V9f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    double dim_n_d = (dim + V9_NBPROC * 2.0 - 1.0) / (V9_NBPROC * 2.0);
    double alloc_d = 5 * dim_n_d + dim * dim_n_d * 2 + dim_n_d * dim_n_d;

    unsigned dim_n = (unsigned) dim_n_d;
    unsigned alloc = (unsigned) alloc_d;

    if (dim_n*dim_n*V9_NBREG_USED>V9_NBREG)
    {
        myprintf("Utilisation de la version 8 car il y a trop de registres utilisés (used = %d , max = %d)\n",dim_n*dim_n*V9_NBREG_USED,V9_NBREG);
        //CudaLCP_FullKernel_V8f(dim,itMax,tol,m,mP,q,f,err,share);
    }
    else if (dim>V9_SZMAX)
    {
        myprintf("Utilisation de la version 8 car il y a trop de contacts (max = %d , dim = %d)\n",V9_SZMAX,dim);
        //CudaLCP_FullKernel_V8f(dim,itMax,tol,m,mP,q,f,err,share);
    }
    else
    {


        //unsigned dim_n = (dim + V9_NBPROC * 2 - 1) / (V9_NBPROC * 2);
        //unsigned alloc = dim_n + dim_n + dim_n + dim * dim_n + dim * dim_n + dim_n * dim_n;

        dim3 threads(dim_n,dim_n);
        dim3 grid(1,V9_NBPROC);

        //printf("\nallocSize %d maxSize = %d blocsize= %d\n",alloc,V9_SZMAX,dim_n);
        switch(dim_n)
        {
#define CASE(N) \
			case N: \
			CudaLCP_FullKernel_V9_kernel<float,N><<< grid, threads, alloc *  sizeof(float)>>>(dim,itMax*V9_NBPROC_2*dim_n,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share); \
				break
            CASE(1);
            CASE(2);
            CASE(3);
            CASE(4);
            CASE(5);
            CASE(6);
            CASE(7);
            CASE(8);
            CASE(9);
            CASE(10);
            CASE(11);
            CASE(12);
            CASE(13);
            CASE(14);
            CASE(15);
            CASE(16);
            CASE(17);
            CASE(18);
            CASE(19);
            CASE(20);
            CASE(21);
            CASE(22);
#undef CASE
        }
        //CudaLCP_FullKernel_V9_kernel<<< grid, threads, alloc *  sizeof(float)>>>(dim,dim_n,V9_NBPROC*2,itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
    }
}
void CudaLCP_FullKernel_V9d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{

}

/*
void CudaLCP_FullKernel_V10f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share) {
	unsigned dim_n = 9;

	if ((dim>0) && (dim<56)) dim_n = 9;
	else dim_n = 9;

	unsigned alloc = 5 * dim_n + dim * dim_n * 2 + dim_n * dim_n;
	unsigned nbBlock = (dim + dim_n*2 - 1) / (dim_n*2);

	if (nbBlock<=V10_NB_PROC) {
		dim3 threads(dim_n,dim_n);
		dim3 grid(1,nbBlock);

		switch(dim_n)
		{
#define CASE(N) \
			case N: \
			CudaLCP_FullKernel_V10_kernel<float,N><<< grid, threads, alloc *  sizeof(float)>>>(dim,itMax*nbBlock*2*dim_n,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share); \
			break
			CASE(1);
			CASE(2);
			CASE(3);
			CASE(4);
			CASE(5);
			CASE(6);
			CASE(7);
			CASE(8);
			CASE(9);
			CASE(10);
			CASE(11);
			CASE(12);
			CASE(13);
			CASE(14);
			CASE(15);
			CASE(16);
			CASE(17);
			CASE(18);
			CASE(19);
			CASE(20);
			CASE(21);
			CASE(22);
#undef CASE
		}

		//CudaLCP_FullKernel_V10_kernel<float,dim_n><<< grid, threads, alloc *  sizeof(float)>>>(dim,itMax*nbBlock*2*dim_n,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
	} else {
		myprintf("Utilisation de la version 8 car il y a trop de multiprocesseurs utilisés (max = %d , dim = %d)\n",V10_NB_PROC,nbBlock);
		//CudaLCP_FullKernel_V8f(dim,itMax,tol,m,mP,q,f,err,share);
	}
}
*/

void CudaLCP_FullKernel_V10f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    unsigned alloc = 5 * V10_DIM_N + dim * V10_DIM_N * 2 + V10_DIM_N * V10_DIM_N;
    unsigned nbBlock = (dim + V10_DIM_N*2 - 1) / (V10_DIM_N*2);

    if (nbBlock<=V10_NB_PROC)
    {
        dim3 threads(V10_DIM_N,V10_DIM_N);
        dim3 grid(1,nbBlock);

        CudaLCP_FullKernel_V10_kernel<float,V10_DIM_N><<< grid, threads, alloc *  sizeof(float)>>>(dim,itMax*nbBlock*2*V10_DIM_N,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
    }
    else
    {
        myprintf("Utilisation de la version 8 car il y a trop de multiprocesseurs utilisés (max = %d , dim = %d)\n",V10_NB_PROC,nbBlock);
        //CudaLCP_FullKernel_V8f(dim,itMax,tol,m,mP,q,f,err,share);
    }
}
void CudaLCP_FullKernel_V10d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{

}

void CudaLCP_FullKernel_V11f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{

    int nb = (dim+V11_BSIZE-1)/V11_BSIZE;
    if (nb > V11_NBPROC) nb = V11_NBPROC;
    dim3 threads(V11_BSIZE,V11_BSIZE/2);
    dim3 grid(1, nb);
    int dim_n = (dim+V11_BSIZE-1)/V11_BSIZE * V11_BSIZE;

    CudaLCP_FullKernel_V11_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaLCP_FullKernel_V11d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    int nb = (dim+V11_BSIZE-1)/V11_BSIZE;
    if (nb > V11_NBPROC) nb = V11_NBPROC;
    dim3 threads(V11_BSIZE,V11_BSIZE/2);
    dim3 grid(1, nb);
    int dim_n = (dim+V11_BSIZE-1)/V11_BSIZE * V11_BSIZE;

    CudaLCP_FullKernel_V11_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,(double)tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

#endif
}

void CudaLCP_FullKernel_V12f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    dim3 threads(V12_BSIZE,V12_BSIZE/2);
    dim3 grid(1,V12_NBPROC);
    int dim_n = (dim+V12_BSIZE-1)/V12_BSIZE * V12_BSIZE;

    CudaLCP_FullKernel_V12_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaLCP_FullKernel_V12d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(V12_BSIZE,V12_BSIZE/2);
    dim3 grid(1,V12_NBPROC);
    int dim_n = (dim+V12_BSIZE-1)/V12_BSIZE * V12_BSIZE;

    CudaLCP_FullKernel_V12_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,(double)tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

#endif
}

void CudaLCP_FullKernel_V13f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    if (dim < 2*V13_NBPROC*V13_BSIZE)
    {
        CudaLCP_FullKernel_V12f(dim, itMax, tol, m, mP, q, f, err, share);
        return;
    }
    dim3 threads(V13_BSIZE,V13_BSIZE/2);
    dim3 grid(1,V13_NBPROC);
    int dim_n = (dim+V13_BSIZE-1)/V13_BSIZE * V13_BSIZE;

    CudaLCP_FullKernel_V13_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaLCP_FullKernel_V13d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    if (dim < 2*V13_NBPROC*V13_BSIZE)
    {
        CudaLCP_FullKernel_V12d(dim, itMax, tol, m, mP, q, f, err, share);
        return;
    }
    dim3 threads(V13_BSIZE,V13_BSIZE/2);
    dim3 grid(1,V13_NBPROC);
    int dim_n = (dim+V13_BSIZE-1)/V13_BSIZE * V13_BSIZE;

    CudaLCP_FullKernel_V13_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,(double)tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

#endif
}
//////////////////////////////////////////////nlcp

/*
	for (int i=0;i<numContacts;i++) {
		int index0 = 3*i;
		int index1 = index0+1;
		int index2 = index0+2;

		cuda_res[index0] = cuda_q[index0];
		cuda_res[index1] = cuda_q[index1];
		cuda_res[index2] = cuda_q[index2];

		for (int j=index2+1;j<dim;j++) {
			cuda_res[index0] += cuda_M[j][index0] * cuda_f[j];
			cuda_res[index1] += cuda_M[j][index1] * cuda_f[j];
			cuda_res[index2] += cuda_M[j][index2] * cuda_f[j];
}
}
*/
void CudaNLCP_MultIndepf(int dim,const void * m,int pM,const void * f,void * tmp,int pTmp)
{
    dim3 threads(MBSIZE,1);
    dim3 grid((dim+MBSIZE-1)/MBSIZE,dim);

    CudaNLCP_MultIndep_kernel<float><<< grid, threads,0>>>(dim, (const float*)m,pM,(const float*)f,(float*)tmp,pTmp);
}
void CudaNLCP_MultIndepd(int dim,const void * m,int pM,const void * f,void * tmp,int pTmp)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(MBSIZE,1);
    dim3 grid((dim+MBSIZE-1)/MBSIZE,dim);

    CudaNLCP_MultIndep_kernel<double><<< grid, threads,0>>>(dim, (const double*)m,pM,(const double*)f,(double*)tmp,pTmp);
#endif
}

void CudaNLCP_AddIndepf(int dim,int tmpsize,const void * q,const void * tmp,int pTmp,void * res)
{
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaNLCP_AddIndep_kernel<float><<< grid, threads,threads.x*sizeof(float)>>>(dim,tmpsize,(const float*)q,(const float*)tmp,pTmp,(float*)res);
}
void CudaNLCP_AddIndepd(int dim,int tmpsize,const void * q,const void * tmp,int pTmp,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaNLCP_AddIndep_kernel<double><<< grid, threads,threads.x*sizeof(double)>>>(dim,tmpsize,(const double*)q,(const double*)tmp,pTmp,(double*)res);
#endif
}

void CudaNLCP_ComputeNextIter_V1_InDepKernelf(int dim,int d,int debutBlock,const void * m,int mP,const void * f,void * res)
{
    dim3 threads(MBSIZE,1);
    dim3 grid(1,dim-d);

    CudaNLCP_ComputeNextIter_V1_InDepKernel_kernel<float><<< grid, threads,0>>>(dim,debutBlock,(const float*) m,mP,(const float*)f, (float *)res);
}
void CudaNLCP_ComputeNextIter_V1_InDepKerneld(int dim,int d,int debutBlock,const void * m,int mP,const void * f,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(MBSIZE,1);
    dim3 grid(1,dim-d);

    CudaNLCP_ComputeNextIter_V1_InDepKernel_kernel<double><<< grid, threads,0>>>(dim,debutBlock,(const double*) m,mP,(const double*)f, (double *)res);
#endif
}

void CudaNLCP_ComputeNextIter_V1_DepKernelf(int d,int debutblock,float mu,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(MBSIZE,1);
    dim3 grid(1,1);

    CudaNLCP_ComputeNextIter_V1_DepKernel_kernel<float><<< grid, threads,0>>>(d,debutblock,mu,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
void CudaNLCP_ComputeNextIter_V1_DepKerneld(int d,int debutblock,float mu,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(MBSIZE,1);
    dim3 grid(1,1);

    CudaNLCP_ComputeNextIter_V1_DepKernel_kernel<double><<< grid, threads,0>>>(d,debutblock,(double)mu,(const double*) m,mP,(const double *) q,(double*)f, (double*) err,(double *)res);
#endif
}

void CudaNLCP_FullKernel_V2f(int dim,int itMax,float tol,float mu,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    dim3 threads(MBSIZE_C,MBSIZE_C);
    dim3 grid(1,NB_MULTIPROC);
    int dim_n = (dim+MBSIZE_C-1)/MBSIZE_C * MBSIZE_C;

    CudaNLCP_FullKernel_V2_kernel<float><<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,mu,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaNLCP_FullKernel_V2d(int dim,int itMax,float tol,float mu,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(MBSIZE_C,MBSIZE_C);
    dim3 grid(1,NB_MULTIPROC);
    int dim_n = (dim+MBSIZE_C-1)/MBSIZE_C * MBSIZE_C;

    CudaNLCP_FullKernel_V2_kernel<double><<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,mu,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

#endif
}

void CudaNLCP_FullKernel_V12f(int dim,int itMax,float tol,float mu,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    dim3 threads(V12_BSIZE,V12_BSIZE/2);
    dim3 grid(1,V12_NBPROC);
    int dim_n = (dim+V12_BSIZE-1)/V12_BSIZE * V12_BSIZE;

    CudaNLCP_FullKernel_V12_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,mu,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaNLCP_FullKernel_V12d(int dim,int itMax,float tol,float mu,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else

    dim3 threads(V12_BSIZE,V12_BSIZE/2);
    dim3 grid(1,V12_NBPROC);
    int dim_n = (dim+V12_BSIZE-1)/V12_BSIZE * V12_BSIZE;

    CudaNLCP_FullKernel_V12_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,(double)tol,(double) mu,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

#endif
}

void CudaNLCP_FullKernel_V13f(int dim,int itMax,float tol,float mu,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    const int V_NBPROC = mycudaGetMultiProcessorCount() * 2;

    dim3 threads(V12_BSIZE,V12_BSIZE/2);
    dim3 grid(1,V_NBPROC);
    int dim_n = (dim+V12_BSIZE-1)/V12_BSIZE * V12_BSIZE;

    CudaNLCP_FullKernel_V13_kernel<<< grid, threads>>>(V_NBPROC,dim,dim_n,dim_n*itMax,tol,mu,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaNLCP_FullKernel_V13d(int dim,int itMax,float tol,float mu,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    const int V_NBPROC = mycudaGetMultiProcessorCount() * 2;

#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else

    dim3 threads(V12_BSIZE,V12_BSIZE/2);
    dim3 grid(1,V_NBPROC);
    int dim_n = (dim+V12_BSIZE-1)/V12_BSIZE * V12_BSIZE;

    CudaNLCP_FullKernel_V13_kernel<<< grid, threads>>>(V_NBPROC,dim,dim_n,dim_n*itMax,(double)tol,(double) mu,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

#endif
}
