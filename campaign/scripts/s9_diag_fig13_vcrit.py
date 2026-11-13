"""Stage 9: her Fig. 13 v_crit (dashed) and v_lim (dash-dot) lines from the PDF vectors, against her nominal adot(m)."""
import pymupdf, glob, numpy as np, collections
d=pymupdf.open(glob.glob('literature/*/Pirani+2019a_*.pdf')[0]); pg=d[10]
x0,x1,yb,yt=335.2,532.9,489.1,369.4
fx=lambda x: 10**(-1+(x-x0)/(x1-x0)*3); fy=lambda y: 10**(-1+(yb-y)/(yb-yt)*4)
G=collections.defaultdict(list)
for dr in pg.get_drawings():
    c=dr.get("color")
    if c is None: continue
    for it in dr["items"]:
        if it[0]!="l": continue
        for p in (it[1],it[2]):
            if x0+1<p.x<x1-1 and 370.5<p.y<497:
                G[(tuple(round(v,2) for v in c), str(dr.get("dashes")))].append((fx(p.x),fy(p.y)))
nom=np.loadtxt("campaign/configs/tracks/pirani19a/fig13_nominal.csv",delimiter=",",skiprows=1)[:,:2]; nom=nom[np.argsort(nom[:,0])]
for k,v in G.items():
    v=np.array(sorted(set(v)));
    print(k, len(v), "m %.2f-%.2f"%(v[:,0].min(),v[:,0].max()))
    if k[0]==(0.0,0.0,0.0) and len(v)>=2 and v[:,0].max()/v[:,0].min()>10:
        # fit log v = a + b log m
        b,a=np.polyfit(np.log10(v[:,0]),np.log10(v[:,1]),1); print(f"   fit: v = {10**a:.3f} m^{b:.3f}")
        for m in (0.3,1,2,3,5,10,20):
            vb=10**(a+b*np.log10(m)); vn=10**np.interp(np.log10(m),np.log10(nom[:,0]),np.log10(nom[:,1]))
            print(f"   m={m:4}: her nominal {vn:6.2f}  her line {vb:6.2f}  nominal/line {vn/vb:.2f}")
