/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

int CudaLCP_MultVector_ResSize_V7(unsigned int dim)
{
    return (dim+32-1)/32;
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
void CudaLCP_AddIndepf(int dim,int tmpsize,const void * q,const void * tmp,int pTmp,void * res)
{
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaLCP_AddIndep_kernel<float><<< grid, threads,threads.x>>>(dim,tmpsize,(const float*)q,(const float*)tmp,pTmp,(float*)res,tmpsize);
}
void CudaLCP_AddIndepd(int dim,int tmpsize,const void * q,const void * tmp,int pTmp,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaLCP_AddIndep_kernel<double><<< grid, threads,threads.x>>>(dim,tmpsize,(const double*)q,(const double*)tmp,pTmp,(double*)res,tmpsize);
#endif
}

void CudaLCP_AddIndepAndUpdatef(int dim,int tmpsize,const void * m,const void * q,const void * tmp,int pTmp,void * f,void * res,void * err)
{
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaLCP_AddIndepAndUpdate_kernel<float><<< grid, threads,threads.x>>>(dim,tmpsize,(const float*)m,(const float*)q,(const float*)tmp,pTmp,(float*)f,(float*)res,(float*)err,tmpsize);
}
void CudaLCP_AddIndepAndUpdated(int dim,int tmpsize,const void * m,const void * q,const void * tmp,int pTmp,void * f,void * res,void * err)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaLCP_AddIndepAndUpdate_kernel<double><<< grid, threads,threads.x>>>(dim,tmpsize,(const double*)m,(const double*)q,(const double*)tmp,pTmp,(float*)f,(double*)res,(double*)err,tmpsize);
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

    CudaLCP_ComputeNextIter_kernel_V2<float><<< grid, threads >>>(dim,compteur2,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
void CudaLCP_ComputeNextIter_V2d(int dim,int compteur2,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,1);

    CudaLCP_ComputeNextIter_kernel_V2<double><<< grid, threads >>>(dim,compteur2,(const double*) m,mP,(const double *) q,(double*)f, (double*) err,(double *)res);
#endif
}

/////////////////////////////////3em version

void CudaLCP_ComputeNextIter_V3_OneKernelf(int dim,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(dim,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V3_OneKernel_kernel<float><<< grid, threads >>>(dim,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
void CudaLCP_ComputeNextIter_V3_OneKerneld(int dim,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(dim,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V3_OneKernel_kernel<double><<< grid, threads >>>(dim,(const double*) m,mP,(const double *) q,(double*)f, (double*) err,(double *)res);
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
    dim3 grid(1,dim-d);

    CudaLCP_ComputeNextIter_V4_InDepKernel_kernel<float><<< grid, threads,0>>>(dim,debutBlock,(const float*) m,mP,(const float*)f, (float *)res,BSIZE/2);
}
void CudaLCP_ComputeNextIter_V4_InDepKerneld(int dim,int d,int debutBlock,const void * m,int mP,const void * f,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid(1,dim-d);

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

    CudaLCP_ComputeNextIter_V5_FirstKernel_kernel<float><<< grid, threads >>>((const float*) m,mP,(const float *) q,(float*)f,(float *)res);

}
void CudaLCP_ComputeNextIter_V5_FirstKerneld(const void * m,int mP,const void * q,void * f,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V5_FirstKernel_kernel<double><<< grid, threads >>>((const double*) m,mP,(const double *) q,(double*)f,(double *)res);
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

    CudaLCP_ComputeNextIter_V5_SecondKernel_kernel<float><<< grid, threads /*,threads.x*sizeof(float)*/ >>>(dim,d,ligne,colone,(const float*) m,mP,(const float *) q,(float*)f,(float*)err,(float *)res,BSIZE/2);
}
void CudaLCP_ComputeNextIter_V5_SecondKerneld(int dim,int nbth,int d,int ligne,int colone,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid(1,nbth+1);

    CudaLCP_ComputeNextIter_V5_SecondKernel_kernel<double><<< grid, threads /*,threads.x*sizeof(double)*/ >>>(dim,d,ligne,colone,(const double*) m,mP,(const double *) q,(double*)f,(double*)err,(double *)res,BSIZE/2);
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

    CudaLCP_FullKernel_V6_kernel<<< grid, threads,threads.x*threads.y*sizeof(double)>>>(dim,dim*itMax,tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);
#endif
}

/////////////////////////////////////////////////7 em version

void CudaLCP_FullKernel_V7f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    dim3 threads(BSIZE_L,BSIZE_L);
    dim3 grid(1,NB_MULTIPROC);
    int dim_n = (dim+BSIZE_L-1)/BSIZE_L * BSIZE_L;

    CudaLCP_FullKernel_V7_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaLCP_FullKernel_V7d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE_L,BSIZE_L);
    dim3 grid(1,NB_MULTIPROC);
    int dim_n = (dim+BSIZE_L-1)/BSIZE_L * BSIZE_L;

    CudaLCP_FullKernel_V7_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

#endif
}

//////////////////version 8

void CudaLCP_FullKernel_V8f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    dim3 threads(BSIZE_C,BSIZE_C);
    dim3 grid(1,NB_MULTIPROC);
    int dim_n = (dim+BSIZE_C-1)/BSIZE_C * BSIZE_C;

    CudaLCP_FullKernel_V8_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaLCP_FullKernel_V8d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE_C,BSIZE_C);
    dim3 grid(1,NB_MULTIPROC);
    int dim_n = (dim+BSIZE_C-1)/BSIZE_C * BSIZE_C;

    CudaLCP_FullKernel_V8_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

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

    CudaNLCP_AddIndep_kernel<float><<< grid, threads,threads.x>>>(dim,tmpsize,(const float*)q,(const float*)tmp,pTmp,(float*)res,tmpsize);
}
void CudaNLCP_AddIndepd(int dim,int tmpsize,const void * q,const void * tmp,int pTmp,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaNLCP_AddIndep_kernel<double><<< grid, threads,threads.x>>>(dim,tmpsize,(const float*)q,(const float*)tmp,pTmp,(float*)res,tmpsize);
#endif
}

void CudaNLCP_ComputeNextIter_V1_InDepKernelf(int dim,int d,int debutBlock,const void * m,int mP,const void * f,void * res)
{
    dim3 threads(MBSIZE,1);
    dim3 grid(1,dim-d);

    CudaNLCP_ComputeNextIter_V1_InDepKernel_kernel<float><<< grid, threads,0>>>(dim,debutBlock,(const float*) m,mP,(const float*)f, (float *)res);

//	printf("fin\n");
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

void CudaNLCP_ComputeNextIter_V1_DepKernelf(int dim,int d,int debutblock,float mu,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(MBSIZE,1);
    dim3 grid(1,1);

    CudaNLCP_ComputeNextIter_V1_DepKernel_kernel<float><<< grid, threads,0>>>(dim,d,debutblock,mu,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
void CudaNLCP_ComputeNextIter_V1_DepKerneld(int dim,int d,int debutblock,float mu,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(MBSIZE,1);
    dim3 grid(1,1);

    CudaNLCP_ComputeNextIter_V1_DepKernel_kernel<double><<< grid, threads,0>>>(dim,d,debutblock,mu,(const double*) m,mP,(const double *) q,(double*)f, (double*) err,(double *)res);
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


/*
template<class real>
__global__ void CudaNLCP_FullKernel_V2_kernel(int dim,int dim_n,int countMax,real tol,real mu,const real * m,int mPitch,const real * q,real * f,volatile real * err,volatile int * share) {
	__shared__ real temp[MBSIZE_L][MBSIZE_L];
	__shared__ real f_i[MBSIZE_L];
	__shared__ real q_s[MBSIZE_L];
	__shared__ real error;
	__shared__ real last_share;
	__shared__ real f_1[3];
	real m_i;

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int count = blockIdx.y * MBSIZE_L + ty;

	if ((tx==0) && (ty==0)) last_share=0;

	__syncthreads();

	while ((last_share<countMax) && (count<countMax)) {
		temp[ty][tx] = 0.0;

		int ligne = count % dim_n;
		int bl = ligne/MBSIZE_L * MBSIZE_L;

	//fin de la ligne il n'y a pas de bloc diago
		int i = bl + MBSIZE_L;
		while ((i<dim_n) && (last_share<countMax)) {
			m_i = ((real *) ((char*) m + (mPitch*ligne)))[i+tx];

			if ((tx==0) && (ty==0)) {
				int dep;

				if (i+MBSIZE_L<dim_n) dep = count-ligne-dim_n+i+MBSIZE_L;
				else dep = count-ligne;

				while (last_share<dep) last_share = share[0]; // boucle de syncho
}

#if MBSIZE_L>32
			__syncthreads();
#endif

			if (ty==0) f_i[tx] = f[i+tx];

			__syncthreads();

			if (i+tx<dim) temp[ty][tx] += m_i * f_i[tx]; //même si ligne>dim mais on ecrira pas a la fin
			i+=MBSIZE_L;
}

	//avant le bloc colone qui contient la diago
		i=0;
		while ((i<bl) && (last_share<countMax)) {
			m_i = ((real *) ((char*) m + (mPitch*ligne)))[i+tx];

			if ((tx==0) && (ty==0)) {
				int dep = count-ligne+i+MBSIZE_L;

				while (last_share<dep) last_share = share[0]; // boucle de syncho
}

#if MBSIZE_L>32
			__syncthreads();
#endif

			if (ty==0) f_i[tx] = f[i+tx];

			__syncthreads();

			temp[ty][tx] += m_i * f_i[tx];
			i+=MBSIZE_L;
}

		if (last_share<countMax) { //on ne fait plus rien
			m_i = ((real *) ((char*) m + (mPitch*ligne)))[bl+tx];

		//debut du bloc diago
			if (ty==0) {
				f_i[tx] = f[bl+tx];

				if (tx<MBSIZE_L) q_s[tx] = q[ligne+tx];

				if (tx==0) {
					if (ligne==0) error=0.0;
					else error=err[0];
}
}

			__syncthreads();

			i=0;
			while ((i<MBSIZE_L) && (bl+i<dim)){
				if (((ty>=i) && (ty<=i+2)) && ((tx<i) || (tx>i+2))) temp[ty][tx] += m_i * f_i[tx]; // mise à jour de tous les valeur du bloc diago

#if MBSIZE_L>32
				__syncthreads();
#endif

				if ((ty==0) && (tx<=2)) { //on prends
					real r_tmp = q_s[i];

					for (int k=1;k<MBSIZE_L;k++) temp[0][bl+ty] += temp[i][bl+ty];

					if (tx==0) f_1[ty] = f_i[bl+ty];

					real d0 = -(temp[0][ty] + temp[0][ty+1]*f_1[1] + temp[0][ty+2]*f_1[2]) / temp[0][bl+i];

					real f_2;

					if (r_tmp<0) f_2 = -r_tmp/m_i;
					else f_2=0.0;

					//error += fabs(m_i * (f_2 - f_1));

					f_i[tx] = f_2;
}

				i+=3;

				__syncthreads();//ligne suivante
}

			if (ty==0) f[bl+tx] = f_i[tx];

#if MBSIZE_L>32
			__syncthreads();
#endif

			if ((ty==0) && (tx==0)) {
				err[0] = error;

				if (ligne+MBSIZE_L==dim_n) {
					if (error<tol)	share[0] = countMax;
					else share[0] += MBSIZE_L;
} else share[0] += MBSIZE_L;
}

			__syncthreads();//ligne suivante

			count += NB_MULTIPROC*MBSIZE_L;
}
}
}
*/
