
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

    CudaLCP_MultIndep_kernel<float><<< grid, threads,threads.x*sizeof(float)>>>(dim, (const float*)m,pM,(const float*)f,(float*)tmp,pTmp,BSIZE/2);
}
void CudaLCP_MultIndepd(int dim,const void * m,int pM,const void * f,void * tmp,int pTmp)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,dim);

    CudaLCP_MultIndep_kernel<double><<< grid, threads,threads.x*sizeof(double)>>>(dim, (const double*)m,pM,(const double*)f,(double*)tmp,pTmp,BSIZE/2);
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

    CudaLCP_AddIndep_kernel<float><<< grid, threads,threads.x>>>(dim,tmpsize,(const float*)q,(const float*)tmp,pTmp,(float*)res,BSIZE/2);
}
void CudaLCP_AddIndepd(int dim,int tmpsize,const void * q,const void * tmp,int pTmp,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);

    CudaLCP_AddIndep_kernel<double><<< grid, threads,threads.x>>>(dim,tmpsize,(const double*)q,(const double*)tmp,pTmp,(double*)res,BSIZE/2);
#endif
}

void CudaLCP_AddIndepAndUpdatef(int dim,int tmpsize,const void * m,const void * q,const void * tmp,int pTmp,void * f,void * res,void * err)
{
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);
    CudaLCP_AddIndepAndUpdate_kernel<float><<< grid, threads,threads.x>>>(dim,tmpsize,(const float*)m,(const float*)q,(const float*)tmp,pTmp,(float*)f,(float*)res,(float*)err,BSIZE/2);
}
void CudaLCP_AddIndepAndUpdated(int dim,int tmpsize,const void * m,const void * q,const void * tmp,int pTmp,void * f,void * res,void * err)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(tmpsize,1);
    dim3 grid(dim,1);
    CudaLCP_AddIndepAndUpdate_kernel<double><<< grid, threads,threads.x>>>(dim,tmpsize,(const double*)m,(const double*)q,(const double*)tmp,pTmp,(float*)f,(double*)res,(double*)err,BSIZE/2);
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

    CudaLCP_ComputeNextIter_V4_DepKernel_kernel<float><<< grid, threads >>>(d,debutblock,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
void CudaLCP_ComputeNextIter_V4_DepKerneld(int d,int debutblock,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V4_DepKernel_kernel<double><<< grid, threads >>>(d,debutblock,(const double*) m,mP,(const double *) q,(double*)f, (double*) err,(double *)res);
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

    CudaLCP_ComputeNextIter_V4_InDepKernel_kernel<float><<< grid, threads,threads.x*sizeof(float)>>>(dim,debutBlock,(const float*) m,mP,(const float*)f, (float *)res,BSIZE/2);
}
void CudaLCP_ComputeNextIter_V4_InDepKerneld(int dim,int d,int debutBlock,const void * m,int mP,const void * f,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid(1,dim-d);

    CudaLCP_ComputeNextIter_V4_InDepKernel_kernel<double><<< grid, threads,threads.x*sizeof(double)>>>(dim,debutBlock,(const double*) m,mP,(const double*)f, (double *)res,BSIZE/2);
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

////////////////////////////////////////////6em version

void CudaLCP_ComputeNextIter_V6_DepKernelf(int d,int debutblock,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(BSIZE,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V6_DepKernel_kernel<float><<< grid, threads,(threads.x*2)*sizeof(float)>>>(d,debutblock,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
void CudaLCP_ComputeNextIter_V6_DepKerneld(int d,int debutblock,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V6_DepKernel_kernel<double><<< grid, threads,(threads.x*2)*sizeof(double)>>>(d,debutblock,(const double*) m,mP,(const double *) q,(double*)f, (double*) err,(double *)res);
#endif
}

////////////////////////////////////////////7em version

/*
for (int i=0;i<dim;i++) {
	cuda_res[i] = cuda_q[i];
	for (int j=0;j<dim;j++) {
		if (j>i) cuda_res[i] += cuda_M[i][j] * cuda_f[j];
}
}
*/
void CudaLCP_MultIndep_V7f(int dim,const void * m,int pM,const void * f,void * tmp,int pTmp)
{
    dim3 threads(32,1);
    dim3 grid((dim+32-1)/32,dim);

    CudaLCP_MultIndep_V7_kernel<float><<< grid, threads,threads.x*sizeof(float)>>>(dim, (const float*)m,pM,(const float*)f,(float*)tmp,pTmp);
}
void CudaLCP_MultIndep_V7d(int dim,const void * m,int pM,const void * f,void * tmp,int pTmp)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(32,1);
    dim3 grid((dim+32-1)/32,dim);

    CudaLCP_MultIndep_V7_kernel<double><<< grid, threads,threads.x*sizeof(double)>>>(dim, (const double*)m,pM,(const double*)f,(double*)tmp,pTmp);
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
void CudaLCP_AddIndep_V7f(int dim,int tmpsize,const void * q,const void * tmp,int pTmp,void * res)
{
    dim3 threads(32,1);
    dim3 grid((dim+32-1)/32,1);

    CudaLCP_AddIndep_V7_kernel<float><<< grid, threads,0>>>(dim,tmpsize,(const float*)q,(const float*)tmp,pTmp,(float*)res);
}
void CudaLCP_AddIndep_V7d(int dim,int tmpsize,const void * q,const void * tmp,int pTmp,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(32,1);
    dim3 grid((dim+32-1)/32,1);

    CudaLCP_AddIndep_V7_kernel<double><<< grid, threads,0>>>(dim,tmpsize,(const double*)q,(const double*)tmp,pTmp,(double*)res);
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
void CudaLCP_ComputeNextIter_V7_InDepKernelf(int dim,int d,int debutBlock,const void * m,int mP,const void * f,void * res)
{
    dim3 threads(32,1);
    dim3 grid(1,dim-d);

    CudaLCP_ComputeNextIter_V7_InDepKernel_kernel<float><<< grid, threads,threads.x*sizeof(float)>>>(dim,debutBlock,(const float*) m,mP,(const float*)f, (float *)res);
}
void CudaLCP_ComputeNextIter_V7_InDepKerneld(int dim,int d,int debutBlock,const void * m,int mP,const void * f,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(32,1);
    dim3 grid(1,dim-d);

    CudaLCP_ComputeNextIter_V7_InDepKernel_kernel<double><<< grid, threads,threads.x*sizeof(double)>>>(dim,debutBlock,(const double*) m,mP,(const double*)f, (double *)res);
#endif
}

void CudaLCP_ComputeNextIter_V7_DepKernelf(int d,int debutblock,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(32,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V7_DepKernel_kernel<float><<< grid, threads,(threads.x*2)*sizeof(float)>>>(d,debutblock,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
void CudaLCP_ComputeNextIter_V7_DepKerneld(int d,int debutblock,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(32,1);
    dim3 grid(1,1);

    CudaLCP_ComputeNextIter_V7_DepKernel_kernel<double><<< grid, threads,(threads.x*2)*sizeof(double)>>>(d,debutblock,(const double*) m,mP,(const double *) q,(double*)f, (double*) err,(double *)res);
#endif
}

/////////////////////////////////////////////////8 em version

void CudaLCP_FullKernel_V8f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    dim3 threads(BSIZE_C,1);
    dim3 grid(1,NB_MULTIPROC);

    CudaLCP_FullKernel_V8_kernel<<< grid, threads,threads.x*threads.y*sizeof(float)>>>(dim,dim*itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);

}
void CudaLCP_FullKernel_V8d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE_C,1);
    dim3 grid(1,NB_MULTIPROC);

    //CudaLCP_FullKernel_V8_kernel<<< grid, threads,threads.x*threads.y*sizeof(double)>>>(dim,dim*itMax,tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);
#endif
}

//////////////////version 9
/*
template<class real>
__global__ void CudaLCP_FullKernel_V9_kernel(int dim,int dim_n,int countMax,real tol,const real * m,int mPitch,const real * q,real * f,volatile real * err,volatile int * share) {
	__shared__ real temp[BSIZE_L][BSIZE_C];
	__shared__ real f_i[BSIZE_C];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int count = blockIdx.y * BSIZE_L + ty;
	int last_share = 0;

	while ((last_share<countMax) && (count<countMax)) {
		temp[ty][tx] = 0.0;

		int ligne = count % dim_n;
		int bl = ligne/BSIZE_C * BSIZE_C;
		int bd = ligne/BSIZE_L * BSIZE_L;

		//fin de la ligne il n'y a pas de bloc diago
		int i = bl + BSIZE_C;
		while (i<dim) {
			real m_i = ((real *) ((char*) m + (mPitch*ligne)))[i+tx];
			if ((tx==0) && (ty==0)) {
				int dep;

				if (i+BSIZE_C<dim_n) dep = count-ligne-dim_n+i+BSIZE_C;
				else dep = count-ligne;

				while (last_share<dep) last_share = share[0]; // boucle de syncho
}

			if (BSIZE_C>32)	__syncthreads();
			if (ty==0) f_i[tx] = f[i+tx];
			__syncthreads();

			if (i+tx<dim) temp[ty][tx] += m_i * f_i[tx]; //on calcule mm si ligne>dim mais on ecrira pas a la fin
			i+=BSIZE_C;
}

		//avant le bloc colone qui contient la diago
		i=0;
		while (i<bl) {
			real m_i = ((real *) ((char*) m + (mPitch*ligne)))[i+tx];
			if ((tx==0) && (ty==0)) {
				int dep = count-ligne+i+BSIZE_C;

				while (last_share<dep) last_share = share[0]; // boucle de syncho
}

			if (BSIZE_C>32)	__syncthreads();
			if (ty==0) f_i[tx] = f[i+tx];
			__syncthreads();

			temp[ty][tx] += m_i * f_i[tx];
			i+=BSIZE_C;
}

		if (last_share<countMax) { //on ne fait plus rien
			real m_i = ((real *) ((char*) m + (mPitch*ligne)))[bl+tx]; // on a assez de threads pour tout lire
			if ((tx==0) && (ty==0))
{
				int dep = count-ligne+bd;
				while (last_share<dep) last_share = share[0]; // boucle de syncho
}
			if (BSIZE_C>32)	__syncthreads();

			//debut du bloc diago
			if (ty==0) f_i[tx] = f[bl+tx];

			__syncthreads();

			i=0;
			while ((i<BSIZE_L) && (bl+i<dim)){
				if (ty==i) {
					if ((bl+tx!=bd+i) && (bl+tx<dim)) temp[ty][tx] += m_i * f_i[tx]; // mis a jour de tous les valeur du bloc diago
}
				if (BSIZE_C>32)	__syncthreads();
				if (ty==i) {

					if (bl+tx==bd+i) {
						real r_tmp = q[ligne];
						for (int k=0;k<BSIZE_C;k++) r_tmp += temp[ty][k];

						real f_1 = f_i[tx];

						real f_2;
						if (r_tmp<0) f_2 = -r_tmp/m_i;
						else f_2=0.0;

						real error = fabs(m_i * (f_2 - f_1));

						f_i[tx] = f_2;

						if (ligne>0) error += err[0];
						err[0] = error;

						if (ligne==dim-1) {
							if (error<tol)	share[0] = countMax;
}
}
}

				i++;

				__syncthreads();//ligne suivante
}

			if (ty==0) {
				f[bl+tx] = f_i[tx];
}
			if (BSIZE_C>32)	__syncthreads();
			if (ty==0) {
					if (tx==0) share[0] += BSIZE_L;
}
			__syncthreads();//ligne suivante

			count += NB_MULTIPROC*BSIZE_L;
}
}
}
*/
void CudaLCP_FullKernel_V9f(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
    dim3 threads(BSIZE_C,BSIZE_L);
    dim3 grid(1,NB_MULTIPROC);
    int dim_n = (dim+BSIZE_L-1)/BSIZE_L * BSIZE_L;

    CudaLCP_FullKernel_V9_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const float *) m,mP,(const float *) q,(float *) f,(float *) err,(int *) share);
}
void CudaLCP_FullKernel_V9d(int dim,int itMax,float tol,const void * m,int mP,const void * q,void * f,void * err,void * share)
{
#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ < 130
    myprintf("CUDA ERROR: double precision not supported.\n");
#else
    dim3 threads(BSIZE_C,BSIZE_L);
    dim3 grid(1,NB_MULTIPROC);
    int dim_n = (dim+BSIZE_L-1)/BSIZE_L * BSIZE_L;

    //CudaLCP_FullKernel_V9_kernel<<< grid, threads,0>>>(dim,dim_n,dim_n*itMax,tol,(const double *) m,mP,(const double *) q,(double *) f,(double *) err,(int *) share);

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
void CudaNLCP_MultIndep(int dim,const void * m,int pM,const void * f,void * tmp,int pTmp)
{
    dim3 threads(BSIZE,1);
    dim3 grid((dim+BSIZE-1)/BSIZE,dim);

    CudaNLCP_MultIndep_kernel<<< grid, threads,threads.x*sizeof(float)>>>(dim, (const float*)m,pM,(const float*)f,(float*)tmp,pTmp,BSIZE/2);
}

void CudaNLCP_ComputeNextIter_V1_InDepKernel(int dim,int d,int debutBlock,const void * m,int mP,const void * f,void * res)
{
    dim3 threads(MBSIZE,1);
    dim3 grid(1,dim-d);

    CudaNLCP_ComputeNextIter_V1_InDepKernel_kernel<<< grid, threads,threads.x*sizeof(float)>>>(dim,debutBlock,(const float*) m,mP,(const float*)f, (float *)res,48);
}

void CudaNLCP_ComputeNextIter_V1_DepKernel(int dim,int d,int debutblock,float mu,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(MBSIZE,1);
    dim3 grid(1,1);

    CudaNLCP_ComputeNextIter_V1_DepKernel_kernel<<< grid, threads, threads.x*5*sizeof(float) >>>(dim,d,debutblock,mu,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}

void CudaNLCP_ComputeNextIter_V2_DepKernel(int dim,int d,int debutblock,float mu,const void * m,int mP,const void * q,void * f,void * err,void * res)
{
    dim3 threads(MBSIZE,1);
    dim3 grid(1,1);

    CudaNLCP_ComputeNextIter_V2_DepKernel_kernel<<< grid, threads, threads.x*5*sizeof(float) >>>(dim,d,debutblock,mu,(const float*) m,mP,(const float *) q,(float*)f, (float*) err,(float *)res);
}
