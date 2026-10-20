from paths import DATA, RES, FIG
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.size': 8, 'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.6})
blue='#2f6fb3'; grey='#9aa0a6'; dark='#333333'
# ---- Fig 1: Lyapunov comparison
rows=[('2D-SELCM, a = 2, b = 100',5.15,4.89,1),('2D-SELCM, a = 2, b = 10',3.14,2.28,1),('2D-SELCM, a = 8, b = 10 (weakest)',5.13,0.29,1),
('Lorenz-type keystream [6]*',0.42,0.01,0),('Logistic map, μ = 4',0.69,None,0),('Sine map',0.69,None,0),('Hénon map (1.4, 0.3)',0.42,-1.62,0)]
fig,ax=plt.subplots(figsize=(6.85,2.9)); h=0.36
for i,(n,l1,l2,o) in enumerate(rows):
    y=-i
    ax.barh(y+h/2,l1,h,color='#c9ccd1',edgecolor='none'); ax.text(l1+0.06,y+h/2,f'{l1:.2f}',va='center',fontsize=7.5,color=dark)
    if l2 is None: ax.text(0.06,y-h/2,'1D map, no λ2',va='center',fontsize=7.5,color='#555')
    else:
        ax.barh(y-h/2,l2,h,color=blue if o else '#6b7178',edgecolor='none')
        ax.text(l2+(0.06 if l2>=0 else -0.06),y-h/2,f'{l2:.2f}',va='center',ha='left' if l2>=0 else 'right',fontsize=7.5,color=dark,fontweight='bold' if o else 'normal')
ax.set_yticks([-i for i in range(len(rows))]); ax.set_yticklabels([r[0] for r in rows])
for t,o in zip(ax.get_yticklabels(),[r[3] for r in rows]): t.set_fontweight('bold' if o else 'normal')
ax.axvline(0,color=dark,lw=0.8); ax.set_xlim(-2.5,6); ax.set_xlabel('Lyapunov exponent (per iteration; * per unit time)')
ax.spines[['top','right','left']].set_visible(False); ax.tick_params(axis='y',length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#c9ccd1',label='λ1 (largest)'),Patch(color='#6b7178',label='λ2 (second)'),Patch(color=blue,label='λ2 of 2D-SELCM')],frameon=False,loc='lower right',fontsize=7.5)
fig.tight_layout(); [fig.savefig(f'{FIG}/Fig1.{e}') for e in ('pdf','eps')]; plt.close(fig)
# ---- Fig 3: pipeline
fig,ax=plt.subplots(figsize=(6.85,3.4)); ax.set_xlim(0,760); ax.set_ylim(404,40); ax.axis('off')
def box(cx,cy,w,t1,t2,main=False,H=56):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-H/2),w,H,boxstyle='round,pad=0,rounding_size=8',fc='#e8f0f9' if main else 'white',ec=blue if main else dark,lw=1.2 if main else 0.8))
    ax.text(cx,cy-6,t1,ha='center',va='center',fontweight='bold',fontsize=8); ax.text(cx,cy+12,t2,ha='center',va='center',fontsize=7.5,color='#444')
def arr(pts):
    xs,ys=zip(*pts); ax.plot(xs[:-1]+(xs[-1],),ys,color=dark,lw=0.8)
    ax.annotate('',xy=pts[-1],xytext=pts[-2],arrowprops=dict(arrowstyle='-|>',color=dark,lw=0.8,mutation_scale=8))
W=112;rA,rB,rC=104,214,334;c1,c2,c3,c4,c6,c5,cs,cl=70,196,322,448,574,700,260,470
box(88,rA,136,'Key K + nonce ν','256 + 128 bits'); box(cs,rA,136,'SHAKE-256','a, b, ε, seeds, IVs'); box(cl,rA,160,'SELCM lattice','portable, 256 lanes')
box(c2,rB,136,'Permutation π','Fisher–Yates, O(N)'); box(cl,rB,282,'Keyed inversion S-box; keys K0, K1','DU 4, NL 32,512; 2 key matrices per pass')
box(c1,rC,W,'Plaintext P','16-bit PCM'); box(c2,rC,W,'Pad + permute','≥ 16 rows of 256'); box(c3,rC,W,'Forward pass','rows 0→R−1')
box(c4,rC,W,'Backward pass','rows R−1→0'); box(c6,rC,W,'Forward pass','rows 0→R−1'); box(c5,rC,W,'Ciphertext C','+ ν, N header',True)
arr([(156,rA),(192,rA)]); arr([(328,rA),(390,rA)]); arr([(cl,rA+28),(cl,rB-28)]); arr([(cl,rA+46),(c2,rA+46),(c2,rB-28)])
ax.text(cl+8,rA+42,'keystream W',fontsize=7.5,color='#444')
arr([(c2,rB+28),(c2,rC-28)]); arr([(350,rB+28),(350,262),(c3,262),(c3,rC-28)]); arr([(c4,rB+28),(c4,rC-28)]); arr([(c6,rB+28),(c6,rC-28)])
ax.text(c3-8,rB+70,'S, K0, K1 per pass',fontsize=7.5,color='#444',ha='right')
for a,b in [(c1,c2),(c2,c3),(c3,c4),(c4,c6),(c6,c5)]: arr([(a+W/2,rC),(b-W/2,rC)])
ax.text(380,392,'Each pass ends with an 8-row wrap. Each step: z = S(Q $\\oplus$ K0), then row ← S(z + Mix(prev) + K1).',ha='center',fontsize=7.5,color='#444')
fig.tight_layout(pad=0.2); [fig.savefig(f'{FIG}/Fig3.{e}') for e in ('pdf','eps')]; plt.close(fig)
print('ok')
