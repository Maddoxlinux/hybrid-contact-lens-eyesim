/* eyesim.js — client-side port of the Python eyesim engine.
   Navarro schematic eye + hybrid refractive/diffractive contact lens.
   Pure JS, no deps. Validated in Node against the Python reference. */
(function (global) {
"use strict";

// ---------- dispersion ----------
var LAM_D=0.58756, LAM_F=0.48613, LAM_C=0.65627, GREEN=0.55;
function cauchyAB(nd, Vd){
  var inv=1/(LAM_F*LAM_F)-1/(LAM_C*LAM_C);
  var B=(nd-1)/(Vd*inv); var A=nd-B/(LAM_D*LAM_D); return [A,B];
}
function Material(name,nd,Vd){ this.name=name; this.nd=nd; this.Vd=Vd; var ab=cauchyAB(nd,Vd); this.A=ab[0]; this.B=ab[1]; }
Material.prototype.index=function(lam){ return this.A+this.B/(lam*lam); };
var MEDIA={
  air:new Material("air",1.0,1e9),
  cornea:new Material("cornea",1.3760,55.0),
  aqueous:new Material("aqueous",1.3374,52.0),
  lens:new Material("lens",1.4201,48.0),
  vitreous:new Material("vitreous",1.3360,52.0),
};
var LENS_MATERIALS={
  "silicone-hydrogel":new Material("silicone-hydrogel",1.430,45.0),
  "hydrogel":new Material("hydrogel",1.400,50.0),
  "PMMA":new Material("PMMA",1.492,57.4),
};
var TEAR=MEDIA.aqueous;

// photopic V(lambda), 400..700nm/10nm (CIE 1924)
var PHOT=[0.0004,0.0012,0.0040,0.0116,0.0230,0.0380,0.0600,0.0910,0.1390,0.2080,
0.3230,0.5030,0.7100,0.8620,0.9540,0.9950,0.9950,0.9520,0.8700,0.7570,0.6310,
0.5030,0.3810,0.2650,0.1750,0.1070,0.0610,0.0320,0.0170,0.0082,0.0041];
function photopic(lam){ var nm=lam*1000, i=(nm-400)/10; if(i<=0)return PHOT[0]; if(i>=30)return PHOT[30];
  var lo=Math.floor(i), t=i-lo; return PHOT[lo]*(1-t)+PHOT[lo+1]*t; }

// ---------- surfaces ----------
// surface: {z, mat (material after), sd, R (Infinity=plane), k, kind, name, Pd0, a4, lam0, order}
function conicIntersect(s, p, d){
  var x0=p[0], y0=p[1], z0=p[2]-s.z, L=d[0], M=d[1], N=d[2];
  if(!isFinite(s.R)){ if(Math.abs(N)<1e-12) return NaN; return (s.z-p[2])/N; }
  var K=1+s.k, R=s.R;
  var a=L*L+M*M+K*N*N;
  var b=2*(x0*L+y0*M+K*z0*N-R*N);
  var c=x0*x0+y0*y0+K*z0*z0-2*R*z0;
  var eps=1e-9, t;
  if(Math.abs(a)<1e-12){ t=(Math.abs(b)>1e-30)?(-c/b):NaN; }
  else{
    var disc=b*b-4*a*c; if(disc<0) return NaN;
    var sq=Math.sqrt(disc), t1=(-b-sq)/(2*a), t2=(-b+sq)/(2*a);
    var tp1=(t1>eps)?t1:Infinity, tp2=(t2>eps)?t2:Infinity;
    t=Math.min(tp1,tp2); if(!isFinite(t)) return NaN;
  }
  return (t>eps)?t:NaN;
}
function conicNormal(s,p){
  if(!isFinite(s.R)) return [0,0,1];
  var Z=p[2]-s.z; var g=[p[0],p[1],(1+s.k)*Z-s.R];
  var n=Math.hypot(g[0],g[1],g[2]); if(n===0)n=1; return [g[0]/n,g[1]/n,g[2]/n];
}
function refract(d,n,mu){ // returns {d, tir}
  var cosi=-(d[0]*n[0]+d[1]*n[1]+d[2]*n[2]);
  if(cosi<0){ n=[-n[0],-n[1],-n[2]]; cosi=-cosi; }
  var s2=mu*mu*(1-cosi*cosi);
  if(s2>1) return {tir:true,d:d};
  var cost=Math.sqrt(1-s2), f=mu*cosi-cost;
  var dt=[mu*d[0]+f*n[0], mu*d[1]+f*n[1], mu*d[2]+f*n[2]];
  var L=Math.hypot(dt[0],dt[1],dt[2]);
  return {tir:false,d:[dt[0]/L,dt[1]/L,dt[2]/L]};
}
function diffAddPower(s,lam){ return s.Pd0*(lam/s.lam0); }
function diffract(s,p,d,nAfter,lam){
  var lam_mm=lam*1e-3, lam0_mm=s.lam0*1e-3;
  var r2=p[0]*p[0]+p[1]*p[1];
  var c2=-(2*Math.PI/lam0_mm)*0.5*(s.Pd0/1000.0);
  var c4=-(2*Math.PI/lam0_mm)*(s.a4||0);
  var dPhi_over_r=2*c2+4*c4*r2;
  var gx=dPhi_over_r*p[0], gy=dPhi_over_r*p[1];
  var fac=(s.order||1)*lam_mm/(2*Math.PI)/nAfter;
  var L=d[0]+fac*gx, M=d[1]+fac*gy;
  var N2=1-L*L-M*M, Ns=(d[2]<0?-1:1);
  var N=Ns*Math.sqrt(Math.max(N2,0));
  var nn=Math.hypot(L,M,N); return [L/nn,M/nn,N/nn];
}

// ---------- system helpers ----------
function indexBefore(sys,i,lam){ return i===0?sys.matBefore.index(lam):sys.surf[i-1].mat.index(lam); }

function paraxialPower(sys,lam){
  var y=1.0, omega=0.0, nb=indexBefore(sys,0,lam);
  var surf=sys.surf, zs=surf.map(function(s){return s.z;});
  for(var i=0;i<surf.length;i++){
    var s=surf[i]; if(s.kind==="image") break;
    var na=s.mat.index(lam);
    if(i>0){ var t=zs[i]-zs[i-1]; y=y+(omega/nb)*t; }
    var K=isFinite(s.R)?(na-nb)/s.R:0; omega=omega-y*K;
    if(s.kind==="diffractive"){ omega=omega-y*(diffAddPower(s,lam)/1000.0); }
    nb=na;
  }
  var power=-omega*1000.0;
  var bfd=(omega===0)?Infinity:(y/(-omega))*nb;
  return {power:power, bfd:bfd};
}

// trace one ray from (x0,y0) collimated; returns {x,y,valid} at image plane, or path
function traceRay(sys,x0,y0,lam,recordPath){
  var zStart=sys.surf[0].z-3.0;
  var p=[x0,y0,zStart], d=[0,0,1], nb=indexBefore(sys,0,lam), valid=true;
  var path=recordPath?[[round4(zStart),round4(y0)]]:null;
  for(var i=0;i<sys.surf.length;i++){
    var s=sys.surf[i];
    var t=conicIntersect(s,p,d);
    if(!(t>0)||!isFinite(t)){ valid=false; break; }
    p=[p[0]+t*d[0],p[1]+t*d[1],p[2]+t*d[2]];
    if(path) path.push([round4(p[2]),round4(p[1])]);
    var r=Math.hypot(p[0],p[1]);
    if(r>s.sd+1e-9) valid=false;
    var na=s.mat.index(lam);
    if(s.kind==="image") break;
    var nrm=conicNormal(s,p); var rr=refract(d,nrm,nb/na);
    if(rr.tir){ valid=false; break; } d=rr.d;
    if(s.kind==="diffractive") d=diffract(s,p,d,na,lam);
    nb=na;
  }
  if(recordPath) return path;
  return {x:p[0],y:p[1],valid:valid};
}
function round4(v){ return Math.round(v*1e4)/1e4; }

// hexapolar pupil
function hexapolar(semi,rings){
  var pts=[[0,0]];
  for(var i=1;i<=rings;i++){ var r=semi*i/rings, n=6*i;
    for(var j=0;j<n;j++){ var a=2*Math.PI*j/n; pts.push([r*Math.cos(a),r*Math.sin(a)]); } }
  return pts;
}
function spotRMS(sys,pupil,lam,rings){
  rings=rings||14; var pts=hexapolar(pupil/2,rings), xs=[],ys=[];
  for(var i=0;i<pts.length;i++){ var r=traceRay(sys,pts[i][0],pts[i][1],lam,false);
    if(r.valid){ xs.push(r.x); ys.push(r.y); } }
  if(!xs.length) return {rms:NaN,x:[],y:[]};
  var cx=mean(xs), cy=mean(ys), s=0, dxs=[],dys=[];
  for(var k=0;k<xs.length;k++){ var dx=xs[k]-cx, dy=ys[k]-cy; s+=dx*dx+dy*dy; dxs.push(dx*1000); dys.push(dy*1000); }
  return {rms:Math.sqrt(s/xs.length)*1000, x:dxs, y:dys, cx:cx, cy:cy};
}
function mean(a){ var s=0; for(var i=0;i<a.length;i++)s+=a[i]; return s/a.length; }

// ---------- eye ----------
var NAVARRO=[
  {R:7.72,k:-0.26,t:0.55,mat:MEDIA.cornea,sd:5.5},
  {R:6.50,k:0.0,t:3.05,mat:MEDIA.aqueous,sd:5.5},
  {R:10.20,k:-3.1316,t:4.00,mat:MEDIA.lens,sd:4.5},
  {R:-6.00,k:-1.00,t:16.3203,mat:MEDIA.vitreous,sd:4.5},
];
function eyeSurfaces(pupilSemi){
  var out=[], z=0;
  for(var i=0;i<NAVARRO.length;i++){ var s=NAVARRO[i];
    var sd=(i===2)?pupilSemi:s.sd;
    out.push({z:z,mat:s.mat,sd:Math.max(sd,pupilSemi),R:s.R,k:s.k,kind:"conic",name:"navarro-"+i});
    z+=s.t;
  }
  return {surf:out, zEnd:z};
}
function buildEye(errorD,pupil,frontOptics){
  var pupilSemi=pupil/2, es=eyeSurfaces(pupilSemi);
  // emmetropic retina from bare-eye paraxial image
  var bare={surf:es.surf.concat([{z:es.zEnd,mat:MEDIA.vitreous,sd:6,R:Infinity,k:0,kind:"image",name:"retina"}]),matBefore:MEDIA.air};
  var pp=paraxialPower(bare,GREEN);
  var zPost=es.surf[es.surf.length-1].z;
  var retinaEmm=zPost+pp.bfd;
  var nVit=MEDIA.vitreous.index(GREEN), F=pp.power;
  var delta=(F===0)?0:(-errorD*nVit/(F*F)*1000.0);
  var retinaZ=retinaEmm+delta;
  var surf=[];
  if(frontOptics) surf=surf.concat(frontOptics);
  surf=surf.concat(es.surf);
  surf.push({z:retinaZ,mat:MEDIA.vitreous,sd:6,R:Infinity,k:0,kind:"image",name:"retina"});
  return {system:{surf:surf,matBefore:MEDIA.air}, retinaZ:retinaZ, axial:retinaZ, error:errorD, pupil:pupil};
}

// ---------- lens ----------
function frontRadius(powerD,mat,Rback){
  var n=mat.nd, ntear=TEAR.nd;
  var rhs=powerD/1000.0-(ntear-n)/Rback;
  if(Math.abs(rhs)<1e-12) return Infinity;
  return (n-1)/rhs;
}
function buildLens(refrD,diffD,designLam,matName,a4){
  var mat=LENS_MATERIALS[matName]||LENS_MATERIALS["silicone-hydrogel"];
  var Rback=7.80, thickness=0.15, tearGap=0.05, sd=6.0;
  var Rfront=frontRadius(refrD,mat,Rback);
  var zBack=-tearGap, zFront=zBack-thickness;
  var front={z:zFront,mat:mat,sd:sd,R:Rfront,k:0,kind:"diffractive",name:"cl-front-diffractive",Pd0:diffD,a4:a4||0,lam0:designLam,order:1};
  var back={z:zBack,mat:TEAR,sd:sd,R:Rback,k:0,kind:"conic",name:"cl-back"};
  return {surfaces:[front,back],Rfront:Rfront,Rback:Rback,refr:refrD,diff:diffD};
}
function bisect(f,lo,hi){
  var flo=f(lo), fhi=f(hi);
  if(Math.sign(flo)===Math.sign(fhi)){ var best=lo,bv=Math.abs(flo);
    for(var i=0;i<=200;i++){ var x=lo+(hi-lo)*i/200, v=Math.abs(f(x)); if(v<bv){bv=v;best=x;} } return best; }
  for(var k=0;k<100;k++){ var mid=0.5*(lo+hi), fm=f(mid);
    if(Math.abs(fm)<1e-12||(hi-lo)<1e-3) return mid;
    if(Math.sign(fm)===Math.sign(flo)){lo=mid;flo=fm;}else{hi=mid;fhi=fm;} }
  return 0.5*(lo+hi);
}
function golden(f,a,b,tol){ tol=tol||0.05; var gr=(Math.sqrt(5)-1)/2;
  var c=b-gr*(b-a), d=a+gr*(b-a), fc=f(c), fd=f(d);
  for(var i=0;i<80;i++){ if(Math.abs(b-a)<tol) break;
    if(fc<fd){ b=d; d=c; fd=fc; c=b-gr*(b-a); fc=f(c); }
    else{ a=c; c=d; fc=fd; d=a+gr*(b-a); fd=f(d); } }
  return 0.5*(a+b);
}
function fitBase(errorD,pupil,mat,designLam,diffD){
  var ref=buildEye(errorD,pupil,null); var retina=ref.retinaZ;
  var zPost=ref.system.surf[ref.system.surf.length-2].z;
  function defocus(P){ var lens=buildLens(P,diffD,designLam,mat,0); var eye=buildEye(errorD,pupil,lens.surfaces);
    var sysR={surf:eye.system.surf.slice(0,-1),matBefore:MEDIA.air};
    return zPost+paraxialPower(sysR,GREEN).bfd-retina; }
  var Ppar=bisect(defocus,-20,120);
  function rms(P){ var lens=buildLens(P,diffD,designLam,mat,0); var eye=buildEye(errorD,pupil,lens.surfaces);
    var r=spotRMS(eye.system,pupil,GREEN,8).rms; return isFinite(r)?r:1e6; }
  return golden(rms,Ppar-20,Ppar+10,0.05);
}
function fitDiff(errorD,pupil,base,mat,designLam){
  function lca(Pd){ var lens=buildLens(base,Pd,designLam,mat,0); var eye=buildEye(errorD,pupil,lens.surfaces);
    var sysR={surf:eye.system.surf.slice(0,-1),matBefore:MEDIA.air};
    return paraxialPower(sysR,0.45).power-paraxialPower(sysR,0.65).power; }
  var f0=lca(0), f1=lca(1), slope=f1-f0;
  if(Math.abs(slope)<1e-9) return 0; return -f0/slope;
}
function autoFit(errorD,pupil,mat,designLam,useDiff){
  var baseEst=fitBase(errorD,pupil,mat,designLam,0);
  var diff=useDiff?fitDiff(errorD,pupil,baseEst,mat,designLam):0;
  var base=fitBase(errorD,pupil,mat,designLam,diff);
  return {base:base,diff:diff};
}

// ---------- metrics ----------
var LCA_LAMS=[0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65];
function lcaCurve(sys){
  var sysR={surf:sys.surf.slice(0,-1),matBefore:sys.matBefore};
  var powers=LCA_LAMS.map(function(l){return paraxialPower(sysR,l).power;});
  var Pref=interp(LCA_LAMS,powers,GREEN);
  var out=LCA_LAMS.map(function(l,i){return [Math.round(l*1000),round4(powers[i]-Pref)];});
  var lca=powers[0]-powers[powers.length-1];
  return {curve:out, lca:round1(lca)};
}
function interp(xs,ys,x){ for(var i=1;i<xs.length;i++){ if(x<=xs[i]){ var t=(x-xs[i-1])/(xs[i]-xs[i-1]); return ys[i-1]*(1-t)+ys[i]*t; } } return ys[ys.length-1]; }
function round1(v){return Math.round(v*100)/100;}

// simple 1-D geometric MTF from the line-spread of ray hits (photopic-weighted)
var MTF_LAMS=[0.45,0.5,0.55,0.6,0.65];
var MM_PER_DEG=16.7*Math.tan(Math.PI/180);
function geomMTF(sys,pupil){
  var N=256, fov=0.0; // adaptive fov from worst spot
  for(var w=0;w<MTF_LAMS.length;w++){ var s=spotRMS(sys,pupil,MTF_LAMS[w],10);
    if(isFinite(s.rms)){ var mx=0; for(var q=0;q<s.x.length;q++){mx=Math.max(mx,Math.abs(s.x[q]),Math.abs(s.y[q]));} fov=Math.max(fov,mx/1000*2.2); } }
  fov=Math.max(fov,0.02);
  var lsf=new Float64Array(N); var wsum=0;
  var semi=pupil/2, rings=26;
  var pts=hexapolar(semi,rings);
  for(var wi=0;wi<MTF_LAMS.length;wi++){ var lam=MTF_LAMS[wi], wt=photopic(lam); wsum+=wt;
    var xs=[]; var cx=0,cnt=0;
    for(var i=0;i<pts.length;i++){ var r=traceRay(sys,pts[i][0],pts[i][1],lam,false); if(r.valid){ xs.push(r.x); cx+=r.x; cnt++; } }
    if(!cnt) continue; cx/=cnt;
    for(var j=0;j<xs.length;j++){ var b=Math.floor((xs[j]-cx+fov/2)/fov*N); if(b>=0&&b<N) lsf[b]+=wt; }
  }
  // normalize
  var tot=0; for(var a=0;a<N;a++) tot+=lsf[a]; if(tot===0) return [[0,0]];
  for(var a2=0;a2<N;a2++) lsf[a2]/=tot;
  // 1-D DFT magnitude -> MTF vs cycles/mm -> cycles/deg
  var pixel=fov/N; var out=[]; var nf=60; // up to 60 c/deg
  for(var fi=0;fi<=nf;fi++){
    var fc=fi; // cycles/deg
    var fmm=fc/MM_PER_DEG; // cycles/mm
    var re=0, im=0;
    for(var x=0;x<N;x++){ var pos=(x-N/2)*pixel; var ph=2*Math.PI*fmm*pos; re+=lsf[x]*Math.cos(ph); im-=lsf[x]*Math.sin(ph); }
    out.push([fc, Math.min(1,Math.hypot(re,im))]);
  }
  return out;
}

// ---------- viz ----------
function surfaceGeom(sys){ return sys.surf.map(function(s){return {z:round4(s.z),R:isFinite(s.R)?round4(s.R):null,sd:round4(s.sd),kind:s.kind,name:s.name};}); }
function rayPaths(sys,pupil){
  var semi=pupil/2, ys=[]; for(var i=0;i<9;i++) ys.push(-semi*0.92+ (semi*0.92*2)*i/8);
  var out={};
  [0.45,0.55,0.65].forEach(function(lam){ var key=String(Math.round(lam*1000));
    out[key]=ys.map(function(y){ return traceRay(sys,0,y,lam,true); }); });
  return out;
}

// ---------- top-level simulate ----------
function spotsFor(sys,pupil){
  var o={}; [0.45,0.55,0.65].forEach(function(lam){ var s=spotRMS(sys,pupil,lam,12), pts=[];
    for(var i=0;i<s.x.length;i++) pts.push([round2(s.x[i]),round2(s.y[i])]); o[String(Math.round(lam*1000))]=pts; }); return o; }
function round2(v){return Math.round(v*100)/100;}
function statePayload(eye,pupil){
  var lca=lcaCurve(eye.system);
  var rmsWl={}; [0.45,0.55,0.65].forEach(function(l){ rmsWl[String(Math.round(l*1000))]=round1(spotRMS(eye.system,pupil,l,14).rms); });
  return {
    spots:spotsFor(eye.system,pupil),
    mtf:geomMTF(eye.system,pupil),
    lca:lca.curve, lca_D:lca.lca,
    rms_um:round1(spotRMS(eye.system,pupil,GREEN,14).rms),
    rms_wl:rmsWl,
    viz:{surfaces:surfaceGeom(eye.system), retina_z:round4(eye.retinaZ), rays:rayPaths(eye.system,pupil)}
  };
}
function simulate(p){
  p=p||{};
  var errorD=(p.error_D!==undefined)?+p.error_D:-3.0;
  var pupil=(p.pupil!==undefined)?+p.pupil:4.0;
  var useLens=(p.use_lens!==undefined)?!!p.use_lens:true;
  var useDiff=(p.use_diffractive!==undefined)?!!p.use_diffractive:true;
  var mat=p.material||"silicone-hydrogel";
  var designLam=(p.design_lam_nm!==undefined)?(+p.design_lam_nm)/1000:GREEN;
  var normal=buildEye(0,pupil,null), uncorr=buildEye(errorD,pupil,null), corrected=uncorr, lensInfo=null;
  if(useLens){
    var fit=autoFit(errorD,pupil,mat,designLam,useDiff);
    var lens=buildLens(fit.base,fit.diff,designLam,mat,0);
    corrected=buildEye(errorD,pupil,lens.surfaces);
    lensInfo={base:round1(fit.base),diff:round1(fit.diff),R_front:round1(lens.Rfront),R_back:round1(lens.Rback),design_nm:Math.round(designLam*1000)};
  }
  return {
    params:{error_D:errorD,pupil:pupil,use_lens:useLens,use_diffractive:useDiff,material:mat,design_lam_nm:designLam*1000},
    lens:lensInfo,
    states:{normal:statePayload(normal,pupil),uncorrected:statePayload(uncorr,pupil),corrected:statePayload(corrected,pupil)}
  };
}

var API={simulate:simulate, _buildEye:buildEye, _paraxialPower:paraxialPower, _spotRMS:spotRMS, _autoFit:autoFit, _lcaCurve:lcaCurve, MEDIA:MEDIA};
if(typeof module!=="undefined"&&module.exports) module.exports=API;
global.EyeSim=API;
})(typeof window!=="undefined"?window:globalThis);
