# SELCM-AE v2 C builds. -ffp-contract=off is REQUIRED for bit-identical output across platforms.
CC ?= gcc
CFLAGS = -O3 -ffp-contract=off -fopenmp-simd
LIBS = -lcrypto -lm
TU01 ?= ../ext/testu01   # path to a built TestU01 1.2.3 (include/ and libtu01.a), only for the testu01 target

all: selcm2 selcm2_avx2 selcm2_sse2
selcm2: selcm2.c                 # tuned for this machine (AVX-512 if available)
	$(CC) $(CFLAGS) -march=native -o $@ $< $(LIBS)
selcm2_avx2: selcm2.c
	$(CC) $(CFLAGS) -mavx2 -o $@ $< $(LIBS)
selcm2_sse2: selcm2.c
	$(CC) $(CFLAGS) -msse2 -mno-avx -o $@ $< $(LIBS)
selcm_v1_baselines: selcm_v1_baselines.c
	$(CC) $(CFLAGS) -march=native -o $@ $< $(LIBS)
testu01: tu_selcm.c tu_savir.c
	$(CC) $(CFLAGS) -march=native -I$(TU01)/include tu_selcm.c $(TU01)/libtu01.a $(LIBS) -o tu_selcm
	$(CC) $(CFLAGS) -march=native -I$(TU01)/include tu_savir.c $(TU01)/libtu01.a $(LIBS) -o tu_savir
clean:
	rm -f selcm2 selcm2_avx2 selcm2_sse2 selcm_v1_baselines tu_selcm tu_savir
.PHONY: all clean testu01
