/* SELCM-AE v2 in portable C (bit-identical to selcm2.py on any IEEE-754 platform).
   Build:  gcc -O3 -march=native -ffp-contract=off -o bench2 selcm2.c -lcrypto -lm
   (-ffp-contract=off is required: fused multiply-add changes the bits.)
   Usage:  bench2 <in.raw int16> <key32.bin> <nonce16.bin> <out.raw> [runs]
           bench2 -d <cipher.raw> <key32.bin> <nonce16.bin> <n> <out.raw>      (decrypt) */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <openssl/evp.h>

#define L 256
#define LOG2L 8
#define BURN 32
#define SBOX_WORDS 128
#define NPASS 3
#define MIN_ROWS 16

static const double COS_C[11] = {0x1.0000000000000p+0,-0x1.0000000000000p-1,0x1.5555555555555p-5,-0x1.6c16c16c16c17p-10,
  0x1.a01a01a01a01ap-16,-0x1.27e4fb7789f5cp-22,0x1.1eed8eff8d898p-29,-0x1.93974a8c07c9dp-37,0x1.ae7f3e733b81fp-45,
  -0x1.6827863b97d97p-53,0x1.e542ba4020225p-62};
static const double EXP_C[18] = {0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p-1,0x1.5555555555555p-3,
  0x1.5555555555555p-5,0x1.1111111111111p-7,0x1.6c16c16c16c17p-10,0x1.a01a01a01a01ap-13,0x1.a01a01a01a01ap-16,
  0x1.71de3a556c734p-19,0x1.27e4fb7789f5cp-22,0x1.ae64567f544e4p-26,0x1.1eed8eff8d898p-29,0x1.6124613a86d09p-33,
  0x1.93974a8c07c9dp-37,0x1.ae7f3e733b81fp-41,0x1.ae7f3e733b81fp-45,0x1.952c77030ad4ap-49};
#define PI 3.141592653589793

#pragma omp declare simd notinbranch
static inline double psin(double y){ double t=PI*(y-0.5), w=t*t, c=COS_C[10];
  #pragma GCC unroll 16
  for(int k=9;k>=0;k--) c=c*w+COS_C[k]; return c; }
#pragma omp declare simd notinbranch
static inline double pexp(double u){ double c=EXP_C[17];
  #pragma GCC unroll 20
  for(int k=16;k>=0;k--) c=c*u+EXP_C[k]; return c; }
static inline double hfun(double u){ return pexp(u)*(1.0+u*(1.0-u)); }
/* arguments are always >= 0 and < 2^31, so truncation == floor */
static inline double fracp(double v){ return v-(double)(int32_t)v; }

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double u53(const uint8_t *b){ uint64_t v; memcpy(&v,b,8); return (double)(v>>11)/9007199254740992.0; }

static void lattice(double *restrict X,double *restrict Y,double a,double b,double eps,uint16_t *restrict out,size_t nwords){
    double xn[L+1] __attribute__((aligned(64))), yn[L+1] __attribute__((aligned(64))); size_t k=0; long step=0; const double ce=1.0-eps;
    while(k<nwords){
        #pragma omp simd
        for(int j=0;j<L;j++){ xn[j]=fracp(a*psin(Y[j])+b*hfun(X[j])); }
        #pragma omp simd
        for(int j=0;j<L;j++){ yn[j+1]=fracp(a*psin(xn[j])+b*hfun(Y[j])); }
        xn[L]=xn[0]; yn[0]=yn[L];
        for(int j=0;j<L;j++){ X[j]=fracp(ce*xn[j]+eps*xn[j+1]); Y[j]=fracp(ce*yn[j+1]+eps*yn[j]); }
        step++; if(step<=BURN) continue;
        for(int j=0;j<L && k<nwords;j++){
            uint64_t mx=(uint64_t)(X[j]*4503599627370496.0), my=(uint64_t)(Y[j]*4503599627370496.0);
            uint16_t w[4]={(uint16_t)(mx>>12),(uint16_t)(my>>12),(uint16_t)(mx>>28),(uint16_t)(my>>28)};
            for(int t=0;t<4&&k<nwords;t++) out[k++]=w[t];
        }
    }
}

/* ---------- GF(2^16) inversion and keyed affine S-box ---------- */
static uint16_t GFINV[65536];
static void gf_init(void){ static uint16_t ex[65535]; static int32_t lg[65536]; uint32_t v=1;
  for(int i=0;i<65535;i++){ ex[i]=(uint16_t)v; lg[v]=i; v<<=1; if(v&0x10000) v^=0x1100B; }
  GFINV[0]=0; for(uint32_t x=1;x<65536;x++) GFINV[x]=ex[(65535-lg[x])%65535]; }
static void matrix(const uint16_t *w,uint32_t cols[16]){
  uint32_t Lm[16],Um[16],LU[16],rows[16]; int bits[256],bi=0,perm[16];
  for(int i=0;i<16;i++) for(int t=0;t<16;t++) bits[16*i+t]=(w[i]>>t)&1;
  for(int i=0;i<16;i++){ Lm[i]=1u<<i; Um[i]=1u<<i;
    for(int j=0;j<i;j++){ if(bits[bi]) Lm[i]|=1u<<j; bi++; }
    for(int j=i+1;j<16;j++){ if(bits[bi]) Um[i]|=1u<<j; bi++; } }
  for(int i=0;i<16;i++) perm[i]=i;
  for(int i=15;i>0;i--){ int jj=(int)(((uint32_t)w[16+i]*(uint32_t)(i+1))>>16); int t=perm[i]; perm[i]=perm[jj]; perm[jj]=t; }
  for(int i=0;i<16;i++){ uint32_t r=0; for(int j=0;j<16;j++) if((Lm[i]>>j)&1) r^=Um[j]; LU[i]=r; }
  for(int i=0;i<16;i++) rows[i]=LU[perm[i]];
  for(int c=0;c<16;c++){ uint32_t v=0; for(int i=0;i<16;i++) if((rows[i]>>c)&1) v|=1u<<i; cols[c]=v; } }
static inline uint32_t apply(const uint32_t *cols,uint32_t x){ uint32_t v=0; for(int i=0;i<16;i++) if((x>>i)&1) v^=cols[i]; return v; }
static void build_sbox(const uint16_t *w,uint16_t *S,uint16_t *Si){
  uint32_t A[16],B[16]; matrix(w,A); matrix(w+32,B); uint32_t al=w[64],be=w[65];
  /* byte tables for fast matrix application */
  uint32_t Alo[256],Ahi[256],Blo[256],Bhi[256];
  for(int v=0;v<256;v++){ Alo[v]=apply(A,v); Ahi[v]=apply(A,v<<8); Blo[v]=apply(B,v); Bhi[v]=apply(B,v<<8); }
  for(uint32_t x=0;x<65536;x++){ uint32_t t=(Blo[x&255]^Bhi[x>>8])^be; uint32_t g=GFINV[t];
    uint32_t y=(Alo[g&255]^Ahi[g>>8])^al; S[x]=(uint16_t)y; if(Si) Si[y]=(uint16_t)x; } }

/* ---------- diffusion ---------- */
static inline uint32_t rotl16(uint32_t v,int r){ return ((v<<r)|(v>>(16-r)))&0xFFFF; }
static inline void mix(const uint32_t *p,int s,uint32_t *m){
  for(int lo=0;lo<L;lo++){ if(lo&s) continue; int hi=lo|s;
    uint32_t uh=(p[hi]+rotl16(p[lo],7))&0xFFFF; uint32_t ul=(p[lo]+rotl16(uh,11))&0xFFFF; m[hi]=uh; m[lo]=ul; } }
static void diffuse(uint16_t *M,const long *ord,const long *kr,long ns,const uint16_t *K0,const uint16_t *K1,
                    const uint16_t *S,const uint16_t *IV){
  uint32_t prev[L],m[L]; for(int j=0;j<L;j++) prev[j]=IV[j];
  for(long k=0;k<ns;k++){ long r=ord[k],q=kr[k]; mix(prev,1<<(k%LOG2L),m);
    uint16_t *row=M+r*L; const uint16_t *k0=K0+q*L,*k1=K1+q*L;
    for(int j=0;j<L;j++){ uint32_t z=S[row[j]^k0[j]]; uint32_t v=S[(z+m[j]+k1[j])&0xFFFF]; row[j]=(uint16_t)v; prev[j]=v; } } }
static void undiffuse(uint16_t *M,const long *ord,const long *kr,long ns,const uint16_t *K0,const uint16_t *K1,
                      const uint16_t *Si,const uint16_t *IV){
  uint32_t prev[L],m[L];
  for(long k=ns-1;k>=0;k--){ long r=ord[k],q=kr[k];
    if(k==0) for(int j=0;j<L;j++) prev[j]=IV[j]; else { const uint16_t *pr=M+ord[k-1]*L; for(int j=0;j<L;j++) prev[j]=pr[j]; }
    mix(prev,1<<(k%LOG2L),m); uint16_t *row=M+r*L; const uint16_t *k0=K0+q*L,*k1=K1+q*L;
    for(int j=0;j<L;j++){ uint32_t z=(Si[row[j]]-m[j]-k1[j])&0xFFFF; row[j]=(uint16_t)(Si[z]^k0[j]); } } }

typedef struct { long Np,R,ns; uint16_t *ks,*S,*Si,iv[3][L]; long *perm,*ord[3],*kr[3]; } Ctx;

static void setup(Ctx *c,const uint8_t key[32],const uint8_t nonce[16],long n,int want_inv){
  long rows=(n+L-1)/L; if(rows<MIN_ROWS) rows=MIN_ROWS; c->R=rows; c->Np=rows*L;
  size_t need=24+16*L+4*L; uint8_t *d=malloc(need), msg[12+32+16+8];
  memcpy(msg,"SELCM-AE|v2|",12); memcpy(msg+12,key,32); memcpy(msg+44,nonce,16); uint64_t nn=(uint64_t)n; memcpy(msg+60,&nn,8);
  EVP_MD_CTX *h=EVP_MD_CTX_new(); EVP_DigestInit_ex(h,EVP_shake256(),NULL); EVP_DigestUpdate(h,msg,sizeof msg);
  EVP_DigestFinalXOF(h,d,need); EVP_MD_CTX_free(h);
  double a=0.5+7.5*u53(d), b=10.0+90.0*u53(d+8), eps=0.05+0.10*u53(d+16), X[L], Y[L];
  for(int i=0;i<L;i++){ X[i]=u53(d+24+8*i); Y[i]=u53(d+24+8*L+8*i); }
  memcpy(c->iv[0],d+24+16*L,2*L); memcpy(c->iv[1],d+24+16*L+2*L,2*L); for(int j=0;j<L;j++) c->iv[2][j]=c->iv[0][j]^c->iv[1][j];
  free(d);
  size_t total=SBOX_WORDS+(2+2*NPASS)*(size_t)c->Np; c->ks=malloc(total*2); lattice(X,Y,a,b,eps,c->ks,total);
  c->S=malloc(65536*2); c->Si=want_inv?malloc(65536*2):NULL; build_sbox(c->ks,c->S,c->Si);
  long Np=c->Np; c->perm=malloc(Np*sizeof(long)); for(long i=0;i<Np;i++) c->perm[i]=i;
  const uint16_t *w=c->ks+SBOX_WORDS;
  for(long i=Np-1;i>0;i--){ uint64_t v=((uint64_t)w[2*i]<<16)|w[2*i+1]; long j=(long)((v*(uint64_t)(i+1))>>32);
    long t=c->perm[i]; c->perm[i]=c->perm[j]; c->perm[j]=t; }
  long R=c->R, W=R>LOG2L?LOG2L:0; c->ns=R+W;
  for(int p=0;p<3;p++){ c->ord[p]=malloc(c->ns*sizeof(long)); c->kr[p]=malloc(c->ns*sizeof(long)); }
  for(long i=0;i<R;i++){ c->ord[0][i]=i; c->kr[0][i]=i; c->ord[1][i]=R-1-i; c->kr[1][i]=R-1-i; }
  for(long i=0;i<W;i++){ c->ord[0][R+i]=i; c->kr[0][R+i]=R-1-i; c->ord[1][R+i]=R-1-i; c->kr[1][R+i]=i; }
  memcpy(c->ord[2],c->ord[0],c->ns*sizeof(long)); memcpy(c->kr[2],c->kr[0],c->ns*sizeof(long));
}
static void freectx(Ctx *c){ free(c->ks); free(c->S); free(c->Si); free(c->perm); for(int p=0;p<3;p++){ free(c->ord[p]); free(c->kr[p]); } }

static long encrypt(const int16_t *in,long n,const uint8_t *key,const uint8_t *nonce,uint16_t *out){
  Ctx c; setup(&c,key,nonce,n,0); const uint16_t *K=c.ks+SBOX_WORDS+2*c.Np;
  for(long i=0;i<c.Np;i++){ long p=c.perm[i]; out[i]= p<n?(uint16_t)in[p]:0; }
  for(int p=0;p<NPASS;p++) diffuse(out,c.ord[p],c.kr[p],c.ns,K+(2*p)*c.Np,K+(2*p+1)*c.Np,c.S,c.iv[p]);
  long Np=c.Np; freectx(&c); return Np; }
static void decrypt(const uint16_t *cin,long n,const uint8_t *key,const uint8_t *nonce,int16_t *out){
  Ctx c; setup(&c,key,nonce,n,1); const uint16_t *K=c.ks+SBOX_WORDS+2*c.Np; uint16_t *M=malloc(c.Np*2); memcpy(M,cin,c.Np*2);
  for(int p=NPASS-1;p>=0;p--) undiffuse(M,c.ord[p],c.kr[p],c.ns,K+(2*p)*c.Np,K+(2*p+1)*c.Np,c.Si,c.iv[p]);
  for(long i=0;i<c.Np;i++){ long p=c.perm[i]; if(p<n) out[p]=(int16_t)M[i]; }
  free(M); freectx(&c); }

static int cmpd(const void*a,const void*b){ double x=*(double*)a,y=*(double*)b; return (x>y)-(x<y); }
static void *readall(const char *fn,long *bytes){ FILE *f=fopen(fn,"rb"); fseek(f,0,SEEK_END); *bytes=ftell(f); rewind(f);
  void *p=malloc(*bytes+16); if(fread(p,1,*bytes,f)!=(size_t)*bytes) exit(2); fclose(f); return p; }
int main(int argc,char**argv){
  gf_init(); long kb,nb; uint8_t *key,*nonce;
  if(!strcmp(argv[1],"-d")){ long cb; uint16_t *ci=readall(argv[2],&cb); key=readall(argv[3],&kb); nonce=readall(argv[4],&nb);
    long n=atol(argv[5]); int16_t *o=malloc(n*2); decrypt(ci,n,key,nonce,o); FILE *f=fopen(argv[6],"wb"); fwrite(o,2,n,f); fclose(f); return 0; }
  long bytes; int16_t *in=readall(argv[1],&bytes); long n=bytes/2; key=readall(argv[2],&kb); nonce=readall(argv[3],&nb);
  uint16_t *out=malloc((((n+L-1)/L)+MIN_ROWS)*L*2); long Np=encrypt(in,n,key,nonce,out);
  FILE *f=fopen(argv[4],"wb"); fwrite(out,2,Np,f); fclose(f);
  int runs=argc>5?atoi(argv[5]):0; if(!runs) return 0;
  double *t=malloc(runs*sizeof(double)), *td=malloc(runs*sizeof(double)); int16_t *dec=malloc(n*2);
  for(int r=0;r<runs;r++){ double t0=now(); encrypt(in,n,key,nonce,out); t[r]=now()-t0; t0=now(); decrypt(out,n,key,nonce,dec); td[r]=now()-t0; }
  qsort(t,runs,sizeof(double),cmpd); qsort(td,runs,sizeof(double),cmpd);
  printf("SELCM-AE v2\tenc %.6f s\t%.2f MB/s\tdec %.6f s\t%.2f MB/s\tlossless %d\n",t[runs/2],2.0*n/1e6/t[runs/2],td[runs/2],2.0*n/1e6/td[runs/2],!memcmp(dec,in,2*n));
  return 0; }
