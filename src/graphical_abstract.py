from paths import DATA, RES, FIG
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'Liberation Sans','pdf.fonttype':42})
blue='#2f6fb3'; dark='#222'
D=json.load(open(RES + '/dyn2.json'))['le_grid']
A=sorted({float(k.split(',')[0]) for k in D}); B=sorted({float(k.split(',')[1]) for k in D})
M=np.array([[D[f'{a:.3f},{b:.2f}'][1] for b in B] for a in A])
fig=plt.figure(figsize=(8,4),dpi=300)
fig.text(0.5,0.93,'Hyperchaotic lattice audio cipher with provable trail bounds',ha='center',fontsize=15,fontweight='bold',color=dark)
ax=fig.add_axes([0.06,0.14,0.30,0.62]); im=ax.imshow(M,origin='lower',aspect='auto',extent=[B[0],B[-1],A[0],A[-1]],cmap='Blues',vmin=0)
ax.set_xlabel('b',fontsize=11); ax.set_ylabel('a',fontsize=11); ax.set_title('2D-SELCM: λ2 > 0 everywhere\n(min 0.29, whole key range)',fontsize=11)
cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.02); cb.ax.tick_params(labelsize=8); ax.tick_params(labelsize=8)
ax2=fig.add_axes([0.44,0.08,0.54,0.74]); ax2.set_xlim(0,10); ax2.set_ylim(0,10); ax2.axis('off')
steps=['Key + nonce → SHAKE-256','256-lane chaotic lattice','Keyed GF(2¹⁶) S-box (DU 4)','3-pass cyclic diffusion']
for i,s in enumerate(steps):
    y=8.6-i*2.1; ax2.add_patch(FancyBboxPatch((0.0,y-0.55),4.6,1.1,boxstyle='round,pad=0,rounding_size=0.25',fc='#e8f0f9',ec=blue,lw=1.2)); ax2.text(2.3,y,s,ha='center',va='center',fontsize=10)
    if i<3: ax2.annotate('',xy=(2.25,y-1.55),xytext=(2.25,y-0.55),arrowprops=dict(arrowstyle='-|>',color=dark,lw=1))
res=[('≥ 13 active S-boxes',r'trail probability $\leq 2^{-182}$'),('NSCR 99.9985 %','156 files, 30 keys'),('CPA / KPA ≤ 0.003 %','vs 100 % for baselines'),('51.5 MB/s in C','292× real time')]
for i,(a,b) in enumerate(res):
    y=8.6-i*2.1; ax2.text(5.1,y+0.18,a,fontsize=12,fontweight='bold',color=blue,va='center'); ax2.text(5.1,y-0.38,b,fontsize=9.5,color='#444',va='center')
fig.savefig(FIG + '/GraphicalAbstract.png',dpi=300); fig.savefig(FIG + '/GraphicalAbstract.pdf'); print('ok')
