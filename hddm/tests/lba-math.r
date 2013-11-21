fptcdf=function(z,x0max,chi,driftrate,sddrift) {
  zs=z*sddrift;
  zu=z*driftrate;
  chiminuszu=chi-zu;
  xx=chiminuszu-x0max
  chizu=chiminuszu/zs;
  chizumax=xx/zs
  tmp1=zs*(dnorm(chizumax)-dnorm(chizu))
  tmp2=xx*pnorm(chizumax)-chiminuszu*pnorm(chizu)
  1+(tmp1+tmp2)/x0max
}

fptpdf=function(z,x0max,chi,driftrate,sddrift) {
  zs=z*sddrift
  zu=z*driftrate
  chiminuszu=chi-zu
  chizu=chiminuszu/zs
  chizumax=(chiminuszu-x0max)/zs
  (driftrate*(pnorm(chizu)-pnorm(chizumax)) +
    sddrift*(dnorm(chizumax)-dnorm(chizu)))/x0max
}

allrtCDF=function(t,x0max,chi,drift,sdI) {
  # Generates CDF for all RTs irrespective of response.
  N=length(drift) # Number of responses.
  tmp=array(dim=c(length(t),N))
  for (i in 1:N) tmp[,i]=fptcdf(z=t,x0max=x0max,chi=chi,driftrate=drift[i],sddrift=sdI)
  1-apply(1-tmp,1,prod)
}

n1PDF=function(t,x0max,chi,drift,sdI) {
  # Generates defective PDF for responses on node #1.
  N=length(drift) # Number of responses.
  if (N>2) {
    tmp=array(dim=c(length(t),N-1))
    for (i in 2:N) tmp[,i-1]=fptcdf(z=t,x0max=x0max,chi=chi,driftrate=drift[i],sddrift=sdI)
    G=apply(1-tmp,1,prod)
  } else {
    G=1-fptcdf(z=t,x0max=x0max,chi=chi,driftrate=drift[2],sddrift=sdI)
  }
  G*fptpdf(z=t,x0max=x0max,chi=chi,driftrate=drift[1],sddrift=sdI)
}

n1CDF=function(t,x0max,chi,drift,sdI) {
  # Generates defective CDF for responses on node #1.
  outs=numeric(length(t)) ; bounds=c(0,t)
  for (i in 1:length(t)) {
    tmp="error"
    repeat {
      if (bounds[i]>=bounds[i+1]) {outs[i]=0;break}
      tmp=try(integrate(f=n1PDF,lower=bounds[i],upper=bounds[i+1],
        x0max=x0max,chi=chi,drift=drift,sdI=sdI)$value,silent=T)
      if (is.numeric(tmp)) {outs[i]=tmp;break}
      # Try smart lower bound.
      if (bounds[i]<=0) {
	bounds[i]=max(c((chi-0.98*x0max)/(max(mean(drift),drift[1])+2*sdI),0))
	next
      }
      # Try smart upper bound.
      if (bounds[i+1]==Inf) {
	bounds[i+1]=0.02*chi/(mean(drift)-2*sdI)
	next
      }
      stop("Error in n1CDF that I could not catch.")
    }
  }
  cumsum(outs)
}

n1mean=function(x0max,chi,drift,sdI) {
  # Generates mean RT for responses on node #1.
   pc=n1CDF(Inf,x0max,chi,drift,sdI)
   fn=function(t,x0max,chi,drift,sdI,pc) t*n1PDF(t,x0max,chi,drift,sdI)/pc
   tmp=integrate(f=fn,lower=0,upper=100*chi,x0max=x0max,chi=chi,pc=pc,
     drift=drift,sdI=sdI)$value
   list(mean=tmp,p=pc)
}

actCDF=function(z,t,x0max,chi,drift,sdI) {
  # CDF for the distribution (over trials) of activation values at time t.
  zat=(z-x0max)/t ; zt=z/t ; sdi2=2*(sdI^2)
  exp1=exp(-((zt-drift)^2)/sdi2)
  exp2=exp(-((zat-drift)^2)/sdi2)
  tmp1=t*sdI*(exp1-exp2)/sqrt(2*pi)
  tmp2=pnorm(zat,mean=drift,sd=sdI)
  tmp3=pnorm(zt,mean=drift,sd=sdI)
  (tmp1+(x0max-z+drift*t)*tmp2+(z-drift*t)*tmp3)/x0max
}

actPDF=function(z,t,x0max,chi,drift,sdI) {
  # CDF for the distribution (over trials) of activation values at time t.
  tmp1=pnorm((z-x0max)/t,mean=drift,sd=sdI)
  tmp2=pnorm(z/t,mean=drift,sd=sdI)
  (-tmp1+tmp2)/x0max
}

lbameans=function(Is,sdI,x0max,Ter,chi) {
  # Ter should be a vector of length ncond, the others atomic,
  # except Is which is ncond x 2.
  ncond=length(Is)/2
  outm<-outp<-array(dim=c(ncond,2))
  for (i in 1:ncond) {
    for (j in 1:2) {
      tmp=n1mean(x0max,chi,drift=Is[i,switch(j,1:2,2:1)],sdI)
      outm[i,j]=tmp$mean+Ter[i]
      outp[i,]=tmp$p
  }}
  list(mns=c(outm[1:ncond,2],outm[ncond:1,1]),ps=c(outp[1:ncond,2],outp[ncond:1,1]))
}

deadlineaccuracy=function(t,x0max,chi,drift,sdI,guess=.5,meth="noboundary") {
  # Works out deadline accuracy, using one of three
  # methods:
  #   - noboundary = no implicity boundaries.
  #   - partial =  uses implicit boundaries, and partial information.
  #   - nopartial = uses implicit boundaries and guesses otherwise.
  meth=match.arg(meth,c("noboundary","partial","nopartial"))
  noboundaries=function(t,x0max,chi,drift,sdI,ulimit=Inf) {
    # Probability of a correct response in a deadline experiment
    # at times t, with no implicit boundaries.
    N=length(drift)
    tmpf=function(x,t,x0max,chi,drift,sdI) {
      if (N>2) {
	tmp=array(dim=c(length(x),N-1))
	for (i in 2:N) tmp[,i-1]=actCDF(x,t,x0max,chi,drift[i],sdI)
	G=apply(tmp,1,prod)*actPDF(x,t,x0max,chi,drift[1],sdI)
      } else {
	G=actCDF(x,t,x0max,chi,drift[2],sdI)*actPDF(x,t,x0max,chi,drift[1],sdI)
    }}
    outs=numeric(length(t))
    for (i in 1:length(t)) {
      if (t[i]<=0) {
	outs[i]=.5
      } else {
	outs[i]=integrate(f=tmpf,lower=-Inf,upper=ulimit,t=t[i],
          x0max=x0max,drift=drift,sdI=sdI)$value
      }
    }
    outs
  }
  if (meth=="noboundary") {
    noboundaries(t,x0max,chi,drift,sdI,ulimit=Inf)
  } else {
    pt=n1CDF(t=t,x0max=x0max,chi=chi,drift=drift,sdI=sdI)
    pa=allrtCDF(t=t,x0max=x0max,chi=chi,drift=drift,sdI=sdI)
    pguess=switch(meth,"nopartial"=guess*(1-pa),"partial"=
      noboundaries(t=t,x0max=x0max,chi=chi,drift=drift,sdI=sdI,ulimit=chi))
    pt+pguess
  }
}
