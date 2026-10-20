/* TestU01 driver: SmallCrush / Crush on 32-bit words from the SELCM-AE v2 lattice keystream and baselines.
   Build: gcc -O3 -march=native -ffp-contract=off -fopenmp-simd -I../ext/testu01/include tu_selcm.c ../ext/testu01/libtu01.a -lcrypto -lm -o tu_selcm
   Usage: tu_selcm <selcm|logistic|lorenz|aes> <small|crush> <seed> */
#include "unif01.h"
#include "bbattery.h"
#define main selcm_main_unused
#include "selcm2.c"
#undef main

static double SX[L], SY[L], Sa, Sb, Seps; static uint16_t buf16[4 * L]; static int bpos = 4 * L; static long stp = 0;
static void lattice_step_words(void){
    double xn[L+1], yn[L+1]; const double ce = 1.0 - Seps;
    for(;;){
        for(int j=0;j<L;j++) xn[j]=fracp(Sa*psin(SY[j])+Sb*hfun(SX[j]));
        for(int j=0;j<L;j++) yn[j+1]=fracp(Sa*psin(xn[j])+Sb*hfun(SY[j]));
        xn[L]=xn[0]; yn[0]=yn[L];
        for(int j=0;j<L;j++){ SX[j]=fracp(ce*xn[j]+Seps*xn[j+1]); SY[j]=fracp(ce*yn[j+1]+Seps*yn[j]); }
        stp++; if(stp>BURN) break;
    }
    for(int j=0;j<L;j++){ uint64_t mx=(uint64_t)(SX[j]*4503599627370496.0), my=(uint64_t)(SY[j]*4503599627370496.0);
        buf16[4*j]=(uint16_t)(mx>>12); buf16[4*j+1]=(uint16_t)(my>>12); buf16[4*j+2]=(uint16_t)(mx>>28); buf16[4*j+3]=(uint16_t)(my>>28); }
    bpos=0;
}
static unsigned int selcm_bits(void){ if(bpos>=4*L) lattice_step_words(); unsigned int a=buf16[bpos++]; if(bpos>=4*L) lattice_step_words(); return (a<<16)|buf16[bpos++]; }

static double lx; static double lmu;
static unsigned int logistic16(void){ lx=lmu*lx*(1-lx); return (unsigned int)((uint64_t)(lx*1e14)%65536); }
static unsigned int logistic_bits(void){ unsigned int a=logistic16(); return (a<<16)|logistic16(); }
static double ox,oy,oz;
static unsigned int lorenz16(void){ double h=0.005,dx=10*(oy-ox),dy=ox*(28-oz)-oy,dz=ox*oy-8.0/3.0*oz; ox+=h*dx; oy+=h*dy; oz+=h*dz;
    return (unsigned int)((uint64_t)(fabs(ox)*1e12)%65536); }
static unsigned int lorenz_bits(void){ unsigned int a=lorenz16(); return (a<<16)|lorenz16(); }
static EVP_CIPHER_CTX *actx; static uint32_t abuf[4096]; static int apos=4096;
static unsigned int aes_bits(void){ if(apos>=4096){ static uint8_t z[16384]; int l; EVP_EncryptUpdate(actx,(uint8_t*)abuf,&l,z,16384); apos=0; } return abuf[apos++]; }

int main(int argc,char**argv){
    gf_init(); const char *g=argv[1]; int crush=!strcmp(argv[2],"crush"); unsigned seed=(unsigned)atoi(argv[3]);
    uint8_t key[32]={0}, nonce[16]={0}; memcpy(key,&seed,4);
    unif01_Gen *gen;
    if(!strcmp(g,"selcm")){
        size_t need=24+16*L+4*L; uint8_t *d=malloc(need), msg[12+32+16+8]; memcpy(msg,"SELCM-AE|v2|",12); memcpy(msg+12,key,32); memcpy(msg+44,nonce,16);
        uint64_t nn=1000; memcpy(msg+60,&nn,8);
        EVP_MD_CTX *h=EVP_MD_CTX_new(); EVP_DigestInit_ex(h,EVP_shake256(),NULL); EVP_DigestUpdate(h,msg,sizeof msg); EVP_DigestFinalXOF(h,d,need); EVP_MD_CTX_free(h);
        Sa=0.5+7.5*u53(d); Sb=10.0+90.0*u53(d+8); Seps=0.05+0.10*u53(d+16);
        for(int i=0;i<L;i++){ SX[i]=u53(d+24+8*i); SY[i]=u53(d+24+8*L+8*i); }
        gen=unif01_CreateExternGenBits("SELCM-AE v2 lattice keystream", selcm_bits);
    } else if(!strcmp(g,"logistic")){ lx=0.1+0.8*((seed%1000)/1000.0); lmu=3.995; for(int i=0;i<200;i++) logistic16();
        gen=unif01_CreateExternGenBits("Logistic map (B1/B3)", logistic_bits);
    } else if(!strcmp(g,"lorenz")){ ox=1.1+seed%7*0.01; oy=1.2; oz=1.5; for(int i=0;i<500;i++) lorenz16();
        gen=unif01_CreateExternGenBits("Lorenz system (B2)", lorenz_bits);
    } else { actx=EVP_CIPHER_CTX_new(); EVP_EncryptInit_ex(actx,EVP_aes_256_ctr(),NULL,key,nonce);
        gen=unif01_CreateExternGenBits("AES-256-CTR", aes_bits); }
    if(crush) bbattery_Crush(gen); else bbattery_SmallCrush(gen);
    return 0;
}
