"""Independent NumPy checks; no training data, no claim of rerunning old CPU bundles."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20260924)
D = 8
def norm(q):
    return q / np.sqrt(np.mean(q*q, axis=-1, keepdims=True))
def jac(q, eps=0.0):
    s = np.sqrt(q@q / len(q) + eps)
    return np.eye(len(q))/s - np.outer(q,q)/(len(q)*s**3)
def softmax(z):
    e=np.exp(z-np.max(z)); return e/e.sum()
def kl_logits(z1,z2):
    p=softmax(z1); delta=z2-z1; x=delta-p@delta
    # E[x]=0 analytically; this stable formula avoids first-order cancellation.
    return float(np.log1p(p@(np.expm1(x)-x)))
def slope(x,y): return float(np.polyfit(np.log(x),np.log(y),1)[0])
eta=np.geomspace(0.002,0.02,9)
q=rng.normal(size=D); q=q/np.linalg.norm(q)*3
g=rng.normal(size=D)
s=np.sqrt(q@q/D); a=g/s; c=float(q@g/(D*s**3)); t=a-c*q
keys=rng.normal(size=(11,D))
isolated=[]; kl=[]; exact_errors=[]
for h in eta:
    qs=q-h*t; qd=q-h*a
    exact_errors.append(float(np.linalg.norm(norm(qd)-norm(q-h/(1-h*c)*t))))
    isolated.append(float(np.linalg.norm(norm(qd)-norm(qs))))
    kl.append(kl_logits(keys@norm(qs)/np.sqrt(D),keys@norm(qd)/np.sqrt(D)))

# Shared W: exact pullback difference and actual normalized forward difference.
X=rng.normal(size=(5,4)); W=rng.normal(size=(D,4)); Q=X@W.T
G=rng.normal(size=Q.shape)
ss=np.sqrt(np.mean(Q*Q,axis=1)); A=G/ss[:,None]
C=np.sum(Q*G,axis=1)/(D*ss**3)
R=C[:,None]*Q; T=A-R
GWd=A.T@X; GWs=T.T@X
pullback_error=float(np.max(np.abs((GWd-GWs)-R.T@X)))
pred=np.array([-jac(Q[j])@(R.T@X)@X[j] for j in range(len(X))])
pred_cross=np.array([-jac(Q[j])@sum((C[i]*(X[i]@X[j])*Q[i] for i in range(len(X)) if i!=j),np.zeros(D)) for j in range(len(X))])
shared=[]; shared_kl=[]; pred_errors=[]
for h in eta:
    Qs=X@(W-h*GWs).T; Qd=X@(W-h*GWd).T
    dz=norm(Qd)-norm(Qs)
    shared.append(float(np.linalg.norm(dz)))
    pred_errors.append(float(np.linalg.norm(dz-h*pred)))
    shared_kl.append(sum(kl_logits(keys@norm(Qs[j])/np.sqrt(D),keys@norm(Qd[j])/np.sqrt(D)) for j in range(len(X))))

# Independent q with a fixed diagonal preconditioner: first-order leakage.
B=np.diag(np.geomspace(.3,2.0,D)); precond=[]; precond_kl=[]
for h in eta:
    qs=q-h*(B@t); qd=q-h*(B@a)
    precond.append(float(np.linalg.norm(norm(qd)-norm(qs))))
    precond_kl.append(kl_logits(keys@norm(qs)/np.sqrt(D),keys@norm(qd)/np.sqrt(D)))

# Multiple random checks of finite-difference derivative and nonlinearity-aware KL bound.
jac_err=[]; kl_bound_violation=[]
for _ in range(500):
    q0=rng.normal(size=D); d=rng.normal(size=D); h=1e-5
    fd=(norm(q0+h*d)-norm(q0-h*d))/(2*h)
    jac_err.append(float(np.max(np.abs(fd-jac(q0)@d))))
    z=rng.normal(size=12); u=rng.normal(size=12)*.5; r=rng.normal(size=12)*.01
    p=softmax(z); v=np.dot(p,(u-p@u)**2); span=np.ptp(u)
    hm=2*(np.exp(-span)-1+span)/span**2
    hp=2*(np.exp(span)-1-span)/span**2
    K=kl_logits(z,z+u+r); rho=np.ptp(r)
    lo=max(0,.5*v*hm-rho); hi=.5*v*hp+rho
    kl_bound_violation.append(max(lo-K,K-hi,0))

out={
 'date':'2026-09-24', 'seed':20260924, 'purpose':'algebra and finite-step asymptotic checks only; not old experiment reruns',
 'isolated_sgd':{'output_order':slope(eta,isolated),'attention_kl_order':slope(eta,kl),'max_exact_mapping_error':max(exact_errors)},
 'shared_W_sgd':{'output_order':slope(eta,shared),'attention_kl_order':slope(eta,shared_kl),'first_order_residual_order':slope(eta,pred_errors),'pullback_max_error':pullback_error,'diagonal_cancellation_error':float(np.max(np.abs(pred-pred_cross)))},
 'fixed_diagonal_preconditioner':{'output_order':slope(eta,precond),'attention_kl_order':slope(eta,precond_kl)},
 'random_checks':{'n':500,'max_central_difference_error':max(jac_err),'max_range_KL_bound_violation':max(kl_bound_violation)},
 'eta':eta.tolist(), 'raw':{'isolated_output':isolated,'isolated_kl':kl,'shared_output':shared,'shared_kl':shared_kl,'shared_prediction_residual':pred_errors,'preconditioned_output':precond,'preconditioned_kl':precond_kl}}
assert out['isolated_sgd']['max_exact_mapping_error']<1e-12
assert pullback_error<1e-12
assert out['shared_W_sgd']['diagonal_cancellation_error']<1e-12
assert max(kl_bound_violation)<1e-12
Path(__file__).with_name('geometry_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['raw','eta']},indent=2))
