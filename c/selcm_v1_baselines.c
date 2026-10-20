/* SELCM-AE and baselines in C, for timing and cross-implementation verification.
   Build: gcc -O2 -o bench selcm.c -lcrypto -lm
   Usage: bench <in.raw int16> <key32.bin> <nonce16.bin> <out.raw>   (writes SELCM-AE ciphertext, prints timings) */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <openssl/evp.h>

#define L 256
#define LOG2L 8
#define BURN 32

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double frac_(double v){ return v - floor(v); }
static double h_(double u){ return exp(u) * (1.0 + u * (1.0 - u)); }
static double u53(const uint8_t *b){ uint64_t v; memcpy(&v,b,8); return (double)(v>>11) / 9007199254740992.0; }

static void lattice(double *X, double *Y, double a, double b, double eps, uint16_t *out, size_t nwords){
    double xn[L], yn[L]; size_t k = 0; long step = 0;
    while (k < nwords) {
        for (int j=0;j<L;j++){ double x=X[j], y=Y[j];
            double x1 = frac_(a*sin(M_PI*y) + b*h_(x));
            double y1 = frac_(a*sin(M_PI*x1) + b*h_(y)); xn[j]=x1; yn[j]=y1; }
        for (int j=0;j<L;j++){
            X[j] = frac_((1.0-eps)*xn[j] + eps*xn[(j+1)%L]);
            Y[j] = frac_((1.0-eps)*yn[j] + eps*yn[(j-1+L)%L]); }
        step++; if (step <= BURN) continue;
        for (int j=0;j<L && k<nwords;j++){
            uint64_t mx=(uint64_t)(X[j]*4503599627370496.0), my=(uint64_t)(Y[j]*4503599627370496.0);
            uint16_t w[4]={(uint16_t)(mx>>12),(uint16_t)(my>>12),(uint16_t)(mx>>28),(uint16_t)(my>>28)};
            for (int t=0;t<4 && k<nwords;t++) out[k++]=w[t];
        }
    }
}
static inline uint32_t rotl16(uint32_t v,int r){ return ((v<<r)|(v>>(16-r)))&0xFFFF; }

static void diffuse(uint16_t *M, long R, const long *order, const long *krow, long nsteps,
                    const uint16_t *K1, const uint16_t *K2, const uint16_t *S, const uint16_t *IV){
    uint32_t prev[L], cur[L]; for (int j=0;j<L;j++) prev[j]=IV[j];
    for (long k=0;k<nsteps;k++){ long r=order[k], kr=krow[k]; int s=1<<(k%LOG2L);
        for (int j=0;j<L;j++){ uint32_t m=(prev[j]+rotl16(prev[(j-s+L)%L],7))&0xFFFF;
            uint32_t t=(M[r*L+j]+m+K1[kr*L+j])&0xFFFF; cur[j]=S[t]^K2[kr*L+j]; }
        for (int j=0;j<L;j++){ M[r*L+j]=(uint16_t)cur[j]; prev[j]=cur[j]; } }
}

/* returns padded length; out must hold Np samples */
static long selcm_encrypt(const int16_t *in, long n, const uint8_t key[32], const uint8_t nonce[16], uint16_t *out){
    long Np=((n+L-1)/L)*L, R=Np/L;
    size_t need=24+16*L+4*L; uint8_t *d=malloc(need);
    uint8_t msg[12+32+16+8]; memcpy(msg,"SELCM-AE|v1|",12); memcpy(msg+12,key,32); memcpy(msg+44,nonce,16);
    uint64_t nn=(uint64_t)n; memcpy(msg+60,&nn,8);
    EVP_MD_CTX *c=EVP_MD_CTX_new(); EVP_DigestInit_ex(c,EVP_shake256(),NULL); EVP_DigestUpdate(c,msg,sizeof msg);
    EVP_DigestFinalXOF(c,d,need); EVP_MD_CTX_free(c);
    double a=0.5+7.5*u53(d), b=10.0+90.0*u53(d+8), eps=0.05+0.10*u53(d+16);
    double X[L],Y[L]; for(int i=0;i<L;i++){X[i]=u53(d+24+8*i); Y[i]=u53(d+24+8*L+8*i);}
    uint16_t iv1[L],iv2[L]; memcpy(iv1,d+24+16*L,2*L); memcpy(iv2,d+24+16*L+2*L,2*L); free(d);
    size_t total=65536+6*(size_t)Np; uint16_t *ks=malloc(total*2); lattice(X,Y,a,b,eps,ks,total);
    /* stable argsort of 65536 words = counting sort */
    static uint16_t S[65536]; static uint32_t cnt[65537]; memset(cnt,0,sizeof cnt);
    for(int i=0;i<65536;i++) cnt[ks[i]+1]++; for(int v=0;v<65536;v++) cnt[v+1]+=cnt[v];
    for(int i=0;i<65536;i++) S[cnt[ks[i]]++]=(uint16_t)i;
    /* Fisher-Yates */
    long *perm=malloc(Np*sizeof(long)); for(long i=0;i<Np;i++) perm[i]=i; const uint16_t *w=ks+65536;
    for(long i=Np-1;i>0;i--){ uint64_t v=((uint64_t)w[2*i]<<16)|w[2*i+1]; long j=(long)((v*(uint64_t)(i+1))>>32);
        long t=perm[i]; perm[i]=perm[j]; perm[j]=t; }
    const uint16_t *K=ks+65536+2*Np;
    for(long i=0;i<Np;i++){ long p=perm[i]; out[i]= p<n ? (uint16_t)in[p] : 0; }
    long W = R>LOG2L ? LOG2L : 0, ns=R+W;
    long *fo=malloc(ns*sizeof(long)), *fk=malloc(ns*sizeof(long)), *bo=malloc(ns*sizeof(long)), *bk=malloc(ns*sizeof(long));
    for(long i=0;i<R;i++){ fo[i]=i; fk[i]=i; bo[i]=R-1-i; bk[i]=R-1-i; }
    for(long i=0;i<W;i++){ fo[R+i]=i; fk[R+i]=R-1-i; bo[R+i]=R-1-i; bk[R+i]=i; }
    diffuse(out,R,fo,fk,ns,K,K+Np,S,iv1); diffuse(out,R,bo,bk,ns,K+2*Np,K+3*Np,S,iv2);
    free(fo);free(fk);free(bo);free(bk);free(perm);free(ks); return Np;
}

/* ---- baselines ---- */
static void b1(const int16_t *in,long n,double x,double mu,uint16_t *out){
    for(int i=0;i<200;i++) x=mu*x*(1-x);
    for(long i=0;i<n;i++){ x=mu*x*(1-x); out[i]=(uint16_t)in[i]^(uint16_t)((uint64_t)(x*1e14)%65536); } }
static void b3(const int16_t *in,long n,double x,double mu,uint16_t *out){
    for(int i=0;i<200;i++) x=mu*x*(1-x); int64_t prev=12345;
    for(long i=0;i<n;i++){ x=mu*x*(1-x); uint64_t k1=(uint64_t)(x*1e14)%65536; x=mu*x*(1-x); uint64_t k2=(uint64_t)(x*1e14)%65536;
        int64_t t=((uint16_t)in[i]+prev+k1)&0xFFFF; t=((t<<5)|(t>>11))&0xFFFF; prev=t^k2; out[i]=(uint16_t)prev; } }
static double *hv; static int cmpidx(const void *a,const void *b){ long i=*(const long*)a,j=*(const long*)b;
    if(hv[i]<hv[j])return -1; if(hv[i]>hv[j])return 1; return (i>j)-(i<j); }
static void b2(const int16_t *in,long n,double x,double y,double lx,double ly,double lz,uint16_t *out){
    hv=malloc(n*sizeof(double)); for(int i=0;i<200;i++){ double t=1-1.4*x*x+y; y=0.3*x; x=t; }
    for(long i=0;i<n;i++){ double t=1-1.4*x*x+y; y=0.3*x; x=t; hv[i]=x; }
    long *p=malloc(n*sizeof(long)); for(long i=0;i<n;i++) p[i]=i; qsort(p,n,sizeof(long),cmpidx);
    double h=0.005;
    for(long i=0;i<n+500;i++){ double dx=10*(ly-lx),dy=lx*(28-lz)-ly,dz=lx*ly-8.0/3.0*lz; lx+=h*dx; ly+=h*dy; lz+=h*dz;
        if(i>=500) out[i-500]=(uint16_t)in[p[i-500]]^(uint16_t)((uint64_t)(fabs(lx)*1e12)%65536); }
    free(p); free(hv); }
static void aes_ctr(const int16_t *in,long n,const uint8_t *key,const uint8_t *iv,uint16_t *out){
    EVP_CIPHER_CTX *c=EVP_CIPHER_CTX_new(); int l; EVP_EncryptInit_ex(c,EVP_aes_256_ctr(),NULL,key,iv);
    EVP_EncryptUpdate(c,(uint8_t*)out,&l,(const uint8_t*)in,(int)(2*n)); EVP_CIPHER_CTX_free(c); }

static int cmpd(const void*a,const void*b){ double x=*(double*)a,y=*(double*)b; return (x>y)-(x<y); }
int main(int argc,char**argv){
    FILE *f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); long n=ftell(f)/2; rewind(f);
    int16_t *in=malloc(n*2); fread(in,2,n,f); fclose(f);
    uint8_t key[32],nonce[16]; f=fopen(argv[2],"rb"); fread(key,1,32,f); fclose(f); f=fopen(argv[3],"rb"); fread(nonce,1,16,f); fclose(f);
    uint16_t *out=malloc(((n+L-1)/L)*L*2+16);
    long Np=selcm_encrypt(in,n,key,nonce,out); f=fopen(argv[4],"wb"); fwrite(out,2,Np,f); fclose(f);
    int RUNS=31; double t[31]; const char *names[5]={"SELCM-AE","B1 Logistic-XOR","B2 Henon-perm + Lorenz-XOR","B3 Forward chained","B4 AES-256-CTR"};
    for(int m=0;m<5;m++){
        for(int r=0;r<RUNS;r++){ double t0=now();
            switch(m){case 0: selcm_encrypt(in,n,key,nonce,out); break; case 1: b1(in,n,0.37,3.995,out); break;
                case 2: b2(in,n,0.05,0.04,1.3,1.4,1.5,out); break; case 3: b3(in,n,0.37,3.995,out); break; case 4: aes_ctr(in,n,key,nonce,out); break;}
            t[r]=now()-t0; }
        qsort(t,RUNS,sizeof(double),cmpd);
        printf("%s\t%.6f\t%.2f\n",names[m],t[RUNS/2],(2.0*n/1e6)/t[RUNS/2]);
    }
    return 0;
}
