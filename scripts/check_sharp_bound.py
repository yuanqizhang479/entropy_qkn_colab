"""Independent numerical stress check of proved bounded-log-MGF inequalities.
This is a numerical audit, not a proof. numpy and scipy are required.
"""
from decimal import Decimal, localcontext
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp


def h(z):
    z = float(z)
    if abs(z) < 1e-3:
        return 1+z/3+z*z/12+z**3/60+z**4/360+z**5/2520
    return 2*(np.expm1(z)-z)/z**2


def numerical_audit(n_trials=50000, seed=20260924):
    rng=np.random.default_rng(seed)
    margins=[]
    violations=[]
    raw_float_flags=[]
    exact_margins=[]
    for i in range(n_trials):
        n=int(rng.integers(2,65))
        p=rng.dirichlet(np.full(n, rng.choice([.05,.2,1.,10.])))
        raw=rng.normal(size=n)
        x=raw-p@raw
        x*=10**rng.uniform(-3,2)/np.max(np.abs(x))
        v=float(p@(x*x)); b=float(np.max(np.abs(x)))
        if v < 1e-18: continue
        # Compute the centered cumulant without cancellation when b is small.
        if b < 0.1:
            log_m=float(np.log1p(p@(np.expm1(x)-x)))
        else:
            with np.errstate(divide='ignore'):
                log_m=float(logsumexp(np.log(p)+x))
        raw_ratio=2*log_m/v
        lo,hi=h(-b),h(b)
        # e^x - 1 - x removes the first-order cancellation at every scale here.
        log_m=float(np.log1p(p@(np.expm1(x)-x)))
        ratio=2*log_m/v
        if raw_ratio<lo-1e-7*max(1,lo) or raw_ratio>hi+1e-7*max(1,hi):
            with localcontext() as ctx:
                ctx.prec=150
                pd=[Decimal(float(pi)) for pi in p]
                pd=[pi/sum(pd) for pi in pd]
                xd=[Decimal(float(xi)) for xi in x]
                mean=sum(pi*xi for pi,xi in zip(pd,xd))
                xd=[xi-mean for xi in xd]
                bd=max(abs(xi) for xi in xd)
                vd=sum(pi*xi*xi for pi,xi in zip(pd,xd))
                md=sum(pi*(xi.exp()-1-xi) for pi,xi in zip(pd,xd))
                rd=2*(1+md).ln()/vd
                ld=2*((-bd).exp()-1+bd)/(bd*bd)
                ud=2*(bd.exp()-1-bd)/(bd*bd)
                raw_float_flags.append(dict(index=i,raw_ratio=raw_ratio,stable_ratio=ratio,
                    decimal150_ratio=str(rd),decimal150_lower=str(ld),decimal150_upper=str(ud),
                    decimal150_lower_margin=str(rd-ld),decimal150_upper_margin=str(ud-rd),
                    decimal150_violation=bool(rd<ld-Decimal('1e-100') or rd>ud+Decimal('1e-100'))))
        margins.append((ratio-lo,hi-ratio))
        if ratio<lo-1e-7*max(1,lo) or ratio>hi+1e-7*max(1,hi):
            violations.append(dict(index=i,ratio=ratio,lower=lo,upper=hi,b=b,v=v))
        s=v/b**2
        rem=lambda z: np.expm1(z)-z
        exact_lo=float(np.log1p((rem(s*b)+s*rem(-b))/(1+s)))
        exact_hi=float(np.log1p((rem(-s*b)+s*rem(b))/(1+s)))
        exact_margins.append((log_m-exact_lo,exact_hi-log_m))
    # High-precision rare-event sequences verify the sharpness limit numerically.
    sharpness=[]
    with localcontext() as ctx:
        ctx.prec=200
        for bs in ['0.01','0.1','1','10','100']:
            b=Decimal(bs)
            target_lo=2*((-b).exp()-1+b)/(b*b)
            target_hi=2*(b.exp()-1-b)/(b*b)
            for k in [2,6,12,30,60]:
                s=Decimal(10)**(-k)
                v=s*b*b
                lower_ratio=2*(((s*b).exp()+s*(-b).exp())/(1+s)).ln()/v
                upper_ratio=2*(((-s*b).exp()+s*b.exp())/(1+s)).ln()/v
                sharpness.append(dict(b=bs,s=str(s),lower_relative_gap=str((lower_ratio-target_lo)/target_lo),upper_relative_gap=str((target_hi-upper_ratio)/target_hi)))
    return dict(sharpness_decimal_precision=200,flag_recheck_decimal_precision=150,seed=seed,n_trials=n_trials,n_evaluated=len(margins),stable_float_violations=violations,
                raw_float_flag_count=len(raw_float_flags),raw_float_flags_decimal150_recheck=raw_float_flags,
                minimum_sandwich_margin=np.min(margins,axis=0).tolist(),
                minimum_exact_extremal_margin=np.min(exact_margins,axis=0).tolist(),
                sharpness=sharpness,
                caveat='Floating-point checks are supplementary evidence; the accompanying analytic proof establishes the theorem.')

if __name__=='__main__':
    out=numerical_audit()
    path=Path(__file__).with_name('numerical_audit.json')
    path.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['sharpness','raw_float_flags_decimal150_recheck']},indent=2))
