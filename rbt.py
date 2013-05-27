
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Sam Burden, UC Berkeley, 2013 

import os
from time import time
from glob import glob

import numpy as np
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from util import files,geom,num,util

m2mm = 1000.
deg2rad = np.pi / 180.
rad2deg = 180. / np.pi

fmts = ['png']
cdir = 'cal'
ddir = 'dat'
pdir = 'plt'

devs = ['opti','vicon','phase']
dsfx = {'opti':'.csv','vicon':'.dcr','phase':'.txt'}
#dsfx = {'opti':'.csv','vicon':'.dcr','phase':'.rob'}
#dsfx = {'opti':'.csv','vicon':'.dcr','phase':'.c3d'}
#dsfx = {'opti':'.csv','vicon':'.dcr','phase':'.nik'}
units = {'opti':'mm','vicon':'mm','phase':'mm'}

air = ['dat','plt']
cal = ['dat','plane','cal','plt']
ukf = ['dat','cal','geom','ukf','plt']
sync = ['dat','plt']
load = ['load']
skel = ['dat','geom','plt']

def do(di,dev=None,trk='rbt',procs=ukf,exclude='_ukf.npz',**kwds):
  """
  Process all unprocessed rigid body data

  Inputs:
    di - str - directory containing data
    (optional)
    dev - str - motion capture hardware
    trk - str - name of rigid body trackable
    procs - [str,...] - processes to do

  Outputs:
    rbs - list of rigid body structs

  See also:
    do_() is called on each file
  """
  sfx = dsfx[dev]
  dfis = glob( os.path.join(di, '*'+sfx) )
  efis = glob( os.path.join(di, ddir, '*'+exclude) )
  rbs = []
  for dfi in dfis:
    _,fi = os.path.split(dfi)
    fi = fi.split(sfx)[0]
    if dev == 'phase' or '_' not in fi:
      if os.path.join(di, ddir, fi+exclude) not in efis:
        rbs.append( do_(os.path.join(di,fi),dev=dev,trk=trk,procs=procs,**kwds) )
  return rbs

def do_(fi='',dev=None,trk='rbt',procs=ukf,**kwds):
  """
  Process mocap data, run ukf, generate plots from rigid body data 

  Inputs:
    fi - str - rigid body data file name
    (optional)
    trk - str - name of rigid body trackable
    dev - str - motion capture hardware
    procs - [str,...] - processes to do

  Outputs:
    rb - rigid body struct

  Workflow:
    >> # process mocap data, run ukf, & plot results
    >> rb = Rbt('test/20120612-0910',dev='opti',trk='rbt')
    >> rb.dat()
    >> rb.cal()
    >> rb.geom()
    >> rb.ukf()
    >> rb.plt()
  """
  rb = Rbt(fi,trk=trk,dev=dev)
  for proc in procs:
    cmd = 'rb.'+proc+'(**kwds)'
    #print cmd
    eval( cmd )
  return rb

class Rbt():
  """
  Rigid body data class

  Workflow:
    >> # process mocap data, run ukf, & plot results
    >> rb = rbt.Rbt('test/20120612-0910')
    >> rb.dat()
    >> rb.ukf()
    >> rb.plt()

    >> # load processed data & plot results
    >> rb = rbt.Rbt('test/20120612-0910')
    >> rb.load()
    >> rb.plt()
  """

  def __init__(self,fi='',trk='rbt',dev=None):
    """
    Rigid body data

    Inputs
      fi - str - mocap data file name
      (optional)
      trk - str - name of rigid body trackable
      dev - str - motion capture hardware
    """
    if '.' in fi:
      fi,_ = fi.split('.')
    self.fi = fi
    self.t = None
    self.d = None
    self.g = None
    self.X = None
    self.hz = None
    self.trk = trk
    self.dev = dev

  def load(self,fi=None,dbg=True):
    """
    Load processed data from file

    Inputs:
      fi - str - processed data file, *_dat.npz
    """
    if (not fi) and (self.fi):
      fi = self.fi
    if '_' in fi:
      fi,_ = fi.split('_')
    di,fi = os.path.split(fi)
    self.fi = os.path.join(di,fi)

    npf = os.path.join(di,ddir,fi+'_dat.npz')
    if os.path.exists(npf):
      if dbg:
        print 'loading '+npf
      npz = np.load( npf ) 
      t=npz['t']; d=npz['d']; hz=npz['hz'];
      self.t=t; self.d=d; self.hz=hz;
      s = util.Struct()
      s.read( os.path.join(di,ddir,fi+'_dat.py'), locals={'array':np.array})
      self.trk = s.trk; self.dev = s.dev

    npf = os.path.join(di,ddir,fi+'_geom.npz')
    if os.path.exists(npf):
      if dbg:
        print 'loading '+npf
      npz = np.load( npf ) 
      g=npz['g']; pd0=npz['pd0']; d0=npz['d0']
      self.g=g; self.pd0=pd0; self.d0=d0

    npf = os.path.join(di,ddir,fi+'_ukf.npz')
    if os.path.exists(npf):
      if dbg:
        print 'loading '+npf
      npz = np.load( npf ) 
      X = npz['X']; hz = npz['hz']
      self.X = X; self.hz = hz;
      s = util.Struct()
      s.read( os.path.join(di,ddir,fi+'_ukf.py'), locals={'array':np.array})
      self.j = s.j; self.u = s.u; 

  def dat(self,N=np.inf,hz0=None,save=True,dbg=True,**kwds):
    """
    Load raw mocap rigid body data

    Inputs
      (optional)
      N - int - max number of data samples to read
      hz - int - sample rate

    Effects:
      - assigns self.t,.d,.g
      - saves t,d to fi+'_dat.py' and fi+'_dat.npz'
    """
    # unpack data
    fi = self.fi; trk = self.trk
    # read data
    di,fi = os.path.split(self.fi)
    dev = self.dev; sfx = dsfx[dev]
    if dbg:
      print 'reading '+os.path.join(di,fi+sfx); ti = time()

    if dev == 'phase':
      _,s,a,r = fi.strip(sfx).split('_')
      self.trk = s+a+r
      if sfx == '.nik':
        d0 = np.loadtxt(os.path.join(di,fi+sfx))
        t = np.arange(d0.shape[0]) / 480. # fake time samples
        #t = d_[:,-1]; d0 = d_[:,:-2]; d0[d0 == 0.] = np.nan
        #if np.allclose(d_[:,-2],np.arange(d_.shape[0])):
        #  t = d_[:,-1]
        #else:
        #  t = d_[:,-2] + d_[:,-1]*1e-6; 
        #d0 = d_[:,:-2]; d0[d0 == 0.] = np.nan
        if dbg:
          print '%0.1f sec' % (time() - ti)
        if N < np.inf:
          t = t[:N]; d0 = d0[:N]
        N0,M0 = d0.shape; D = 3; M = M0 / D
        #d0.shape = (N0,M,D)
        d0.shape = (N0,D,M)
        #d0 = d0.transpose(0,2,1)
        d0 = d0 / 10. # convert from mm to cm
        #d[:,0] = -d[:,0] # flip x axis
        # insert nan's for missing samples
        dt = np.diff(t)#; dt = dt[(1-np.isnan(dt)).nonzero()]
        hz = int(np.round(1./np.median(dt)))
        if hz0:
          assert hz0 == hz
        else:
          if dbg:
            print 'assuming hz = %d' % hz
        N = int( np.ceil( (t[-1] - t[0]) * hz) ) + 1
        d = np.nan*np.zeros((N,M,D))
        j = np.array( np.round( (t - t[0]) * hz ), dtype=np.int )
        t = np.arange(N) / float(hz)
        # enforce uniform time increments
        d[j,:,:] = d0
        ## remove unobserved features
        #nn = np.logical_not( np.all(np.isnan(d[:,:,0]),axis=0) ).nonzero()[0]
        #if dbg:
        #  print 'keeping observed markers %s' % nn
        #d = d[:,nn,:]
      elif sfx == '.rob':
        d_ = np.loadtxt(os.path.join(di,fi+sfx))
        if np.allclose(d_[:,-2],np.arange(d_.shape[0])):
          t = d_[:,-1]
        else:
          t = d_[:,-2] + d_[:,-1]*1e-6; 
        d0 = d_[:,:-2]; d0[d0 == 0.] = np.nan
        if dbg:
          print '%0.1f sec' % (time() - ti)
        if N < np.inf:
          t = t[:N]; d0 = d0[:N]
        N0,M0 = d0.shape; D = 3; M = M0 / D
        d0.shape = (N0,M,D)
        #d0.shape = (N0,D,M); d0 = d0.transpose(0,2,1)
        d0 = d0 / 1. # convert from mm to cm
        dt = np.diff(t)#; dt = dt[(1-np.isnan(dt)).nonzero()]
        hz = int(np.round(1./np.median(dt)))
        if hz0:
          assert hz0 == hz
        else:
          if dbg:
            print 'measured hz = %d; setting hz = 480' % hz
        hz = 480
        N = int( np.ceil( (t[-1] - t[0]) * hz) ) + 1
        d = np.nan*np.zeros((N,M,D))
        j = np.array( np.round( (t - t[0]) * hz ), dtype=np.int )
        t = np.arange(N) / float(hz)
        # enforce uniform time increments
        #d[j,:,:] = d0
        d = d0
      elif sfx == '.txt':
        d_ = np.loadtxt(os.path.join(di,fi+sfx))
        if np.allclose(d_[:,-2],np.arange(d_.shape[0])):
          t = d_[:,-1]
        else:
          t = d_[:,-2] + d_[:,-1]*1e-6; 
        d0 = d_[:,:-2]; d0[d0 == 0.] = np.nan
        if dbg:
          print '%0.1f sec' % (time() - ti)
        if N < np.inf:
          t = t[:N]; d0 = d0[:N]
        N0,M0 = d0.shape; D = 3; M = M0 / D
        d0.shape = (N0,M,D)
        #d0.shape = (N0,D,M); d0 = d0.transpose(0,2,1)
        d0 = d0 / 10. # convert from mm to cm
        d0 = d0[...,[0,2,1]] # exchange y and z
        dt = np.diff(t)#; dt = dt[(1-np.isnan(dt)).nonzero()]
        hz = int(np.round(1./np.median(dt)))
        if hz0:
          assert hz0 == hz
        else:
          if dbg:
            print 'measured hz = %d; setting hz = 480' % hz
        hz = 480
        N = int( np.ceil( (t[-1] - t[0]) * hz) ) + 1
        d = np.nan*np.zeros((N,M,D))
        j = np.array( np.round( (t - t[0]) * hz ), dtype=np.int )
        t = np.arange(N) / float(hz)
        # enforce uniform time increments
        d[j,:,:] = d0
        ## remove unobserved features
        #nn = np.logical_not( np.all(np.isnan(d[:,:,0]),axis=0) ).nonzero()[0]
        #if dbg:
        #  print 'keeping observed markers %s' % nn
        #d = d[:,nn,:]
      else:
        import c3d
        with open(os.path.join(di,fi+sfx), 'rb') as h:
            r = c3d.Reader(h)
            d = np.dstack([p for p,_ in r.read_frames()])[:,:3,:].T
        
        t = np.arange(d.shape[0]) / 480. # fake time samples
        d[d == 0.0] = np.nan # missing observations
        d = d / 10. # convert from mm to cm
        #d[:,0] = -d[:,0] # flip x axis

    elif dev == 'opti':
      from mocap.python import optitrack as opti
      run = opti.Run()
      run.ReadFile(di,fi+sfx,N=N)
      if dbg:
        print '%0.1f sec' % (time() - ti)
      # extract data from trackable
      if trk:
        t,d0 = run.trk(trk)
      else:
        t,d0,_,_ = run.data()
      if N < np.inf:
        t = t[:N]; d0 = d0[:N]
      d0 *= m2mm
      N0,M,D = d0.shape
      # insert nan's for missing samples
      dt = np.diff(t)#; dt = dt[(1-np.isnan(dt)).nonzero()]
      hz = int(np.round(1./np.median(dt)))
      if hz0:
        assert hz0 == hz
      else:
        if dbg:
          print 'assuming hz = %d' % hz
      N = int( np.ceil( (t[-1] - t[0]) * hz) ) + 1
      d = np.nan*np.zeros((N,M,D))
      j = np.array( np.round( (t - t[0]) * hz ), dtype=np.int )
      t = np.arange(N) / float(hz)
      # enforce uniform time increments
      d[j,...] = d0
      if not( trk == 'l' ) and not( trk == 'r' ):
        try:
          # align time samples with sync electronics
          args = dict(fi=self.fi,dev='opti',procs=sync,dbg=False,save=False)
          l = do_(trk='l',**args).d
          l = np.mean( np.isnan( np.reshape( l, (l.shape[0],-1) ) ), axis=1 )
          r = do_(trk='r',**args).d
          r = np.mean( np.isnan( np.reshape( r, (r.shape[0],-1) ) ), axis=1 )
          j = (.5*(l + r) < .9).nonzero()[0]
          t = t[j] - t[j[0]]; d = d[j,...]
        except AssertionError:
          pass # l or r trackable not found


    elif dev == 'vicon':
      from shrevz import viconparser as vp
      p = vp.ViconParser()
      p.load(os.path.join(di,fi))
      self.p = p
      if dbg:
        print '%0.1f sec' % (time() - ti)
      # extract data from trackable
      t0,d0 = p.t.flatten(),p.xyz
      hz = p.fps
      N0,M,D = d0.shape
      # insert nan's for missing samples
      N = int( np.ceil( t0[-1] - t0[0] ) ) + 1
      d = np.nan*np.zeros((N,M,D))
      j = np.array( np.round( t0 - t0[0] ), dtype=np.int )
      t = np.arange(N) / float(hz)
      d[j,:,:] = d0

    if dbg:
      print 'd0.shape = %s, d.shape = %s' % (d0.shape,d.shape)
    self.t = t; self.d = d;
    self.hz = hz; self.trk = trk;
    if save:
      # save data
      s = util.Struct(hz=hz,trk=trk,dev=dev)
      dir = os.path.join(di,ddir)
      if not os.path.exists( dir ):
        os.mkdir( dir )
      s.write( os.path.join(dir,fi+'_dat.py') )
      np.savez(os.path.join(dir,fi+'_dat.npz'),t=t,d=d,hz=hz)

  def geom(self,**kwds):
    """
    Fit rigid body geometry

    Effects:
      - assigns self.g
      - saves g to fi+'_geom.npz'
    """
    # unpack data
    di,fi = os.path.split(self.fi)
    d = self.d
    N,M,D = d.shape
    # samples where all features appear
    nn = np.logical_not( np.any(np.isnan(d[:,:,0]),axis=1) ).nonzero()[0]
    #assert nn.size > 0
    # fit geometry to pairwise distance data
    pd0 = []; ij0 = []
    for i,j in zip(*[list(a) for a in np.triu(np.ones((M,M)),1).nonzero()]):
      ij0.append([i,j])
      pd0.append(np.sqrt(np.sum((d[:,i,:] - d[:,j,:])**2,axis=1)))
    pd0 = np.array(pd0).T; 
    d0 = num.nanmean(pd0,axis=0); ij0 = np.array(ij0)
    self.pd0 = pd0; self.d0 = d0
    g0 = d[nn[0],:,:]

    # TODO: fix geometry fitting
    if 1:
      g = g0.copy()
    else:
      print 'fitting geom'; ti = time()
      g,info,flag = geom.fit( g0, ij0, d0 )
      print '%0.1f sec' % (time() - ti)
      pd = []; pd0 = []
      for i,j in zip(*[list(a) for a in np.triu(np.ones((M,M)),1).nonzero()]):
        pd.append( np.sqrt( np.sum((g[i,:] - g[j,:])**2) ) )
        pd0.append( np.sqrt( np.sum((g0[i,:] - g0[j,:])**2) ) )
      pd = np.array(pd).T; 
      pd0 = np.array(pd0).T; 
      
    # center and rotate geom flat 
    m = np.mean(g,axis=0)
    g = g - m
    n = geom.plane(g)
    R = geom.orient(n)
    g = np.dot(g,R.T)
    self.g = g
    # save data
    dir = os.path.join(di,ddir)
    if not os.path.exists( dir ):
      os.mkdir( dir )
    np.savez(os.path.join(dir,fi+'_geom.npz'),g=g,pd0=pd0,d0=d0)

  def plane(self,fmts=['png'],out1=50.,out2=15.,**kwds):
    """
    Fit ground plane to data

    Inputs
      fmts - [str,...] - list of figure formats to export
      out1,out2 - float - magic numbers for outlier rejection

    Usage 
      >> d.shape # raw data, N samples of M markers with 3 coordinates (x,y,z)
      (N, M, 3)
      >> c = np.dot(d, R.T) - t # rectified data

    Effects
      - generates fi+'_cal.py' file containing dict of R,t,n
    """
    # unpack data
    t = self.t; d = self.d; g = self.g; hz=self.hz
    di,fi = os.path.split(self.fi)
    nn = np.logical_not( np.any( np.isnan(d[:,:,0]), axis=1) ).nonzero()[0]
    assert nn.size > 0
    N,M,_ = d.shape

    # swap axes in mocap hardware-dependent way
    if self.dev == 'opti':
      R0 = np.array([[0,0,1],[1,0,0],[0,1,0]])
    elif self.dev == 'vicon':
      R0 = np.identity(3)
    # R0 in SO(3)
    assert ( ( np.all(np.dot(R0,R0.T) == np.identity(3)) ) 
            and ( np.linalg.det(R0) == 1.0 ) ) 

    d = np.dot(d, R0.T)

    # collect non-nan data
    x = d[...,0]; y = d[...,1]; z = d[...,2]
    nn = np.logical_not(np.isnan(x.flatten())).nonzero()
    p = np.vstack((x.flatten()[nn],
                   y.flatten()[nn],
                   z.flatten()[nn])).T
    m = p.mean(axis=0)
    p -= m 
    # remove outliers
    p = p[np.abs(p[:,2]) < out1,:]
    # fit plane to data (n is normal vec)
    n = geom.plane(p)
    # rotate normal vertical
    R = geom.orient(n)
    p = np.dot(p,R.T)
    # save plane data
    s = util.Struct(R=np.dot(R,R0),t=np.dot(m,R.T),n=n)
    s.write( os.path.join(di,ddir,fi+'_cal.py') )

  def cal(self,cdir=cdir,**kwds):
    """
    Apply calibration to data

    Inputs:
      (optional)
      cdir - str - directory containing calibration data

    Effects:
      - applies calibration to self.d
    """
    # unpack data
    d0 = self.d; fi = self.fi;
    di,fi = os.path.split(self.fi)
    # load calibration
    c = util.Struct()
    c.read(files.file(fi,di=os.path.join(cdir,ddir),sfx='_cal.py'),
           locals={'array':np.array})
    R,t = c.R,c.t
    # apply calibration to data
    # NOTE: broadcasts over matrix multiplication AND vector addition . . .
    d = np.dot(d0, R.T) - t
    # pack data
    self.d = d

  def ukf(self,ord=2,N=np.inf,Ninit=20,viz=0,**kwds):
    """
    Use UKF to track previously-loaded trajectory

    Inputs:
      (optional)
      ord - int - order of state derivative to track
      N - int - max number of samples to track
      Ninit - int - # of init iterations for ukf
      viz - int - # of samps to skip between vizualization; 0 to disable
      plts - [str,...] - list of plots to generate

    Outputs:
      X - N x 6 - rigid body state estimate at each sample

    Effects:
      - assigns self.X
      - saves X to fi+'_ukf.py' and fi+'_ukf.npz'
    """
    # unpack data
    t = self.t; d = self.d; g = self.g; fi = self.fi; hz=self.hz
    nn = np.logical_not( np.any( np.isnan(d[:,:,0]), axis=1) ).nonzero()[0]
    assert nn.size > 0
    n = 0
    if nn[0] > 0:
      n = nn[0]
      print self.trk+' not visible until sample #%d; trimming data' % n
      t = t[n:]; d = d[n:,:,:]
    di,fi = os.path.split(fi)
    N0,_,_ = d.shape; N = min(N,N0)
    # init ukf
    from uk import uk, body, pts
    X0 = np.hstack( ( np.zeros(2), 2*np.random.rand(), # pitch,roll,yaw
                     num.nanmean(d[:100,:,:],axis=0).mean(axis=0) ) ) # xyz
    Qd = ( np.hstack( (np.array([1,1,1])*2e-3, np.array([1,1,1])*5e+0) ) )
    for o in range(ord-1):
      X0 = np.hstack( ( X0, np.zeros(6) ) )
      Qd = np.hstack( ( Qd, Qd[-6:]*1e-1) )
    b = body.Mocap( X0, g.T, viz=viz, Qd=Qd ); 
    b.Ninit = Ninit;
    self.b = b
    print 'running ukf on %d samps' % N; ti = time()
    t = t[:N]
    j = dict(pitch=0,roll=1,yaw=2,x=3,y=4,z=5)
    X = uk.mocap( b, np.swapaxes(d[:N,:,:],1,2) ).T
    N,M = X.shape
    X = np.vstack(( np.nan*np.zeros((n,M)), X ))
    X[:,0:3] *= rad2deg
    u = dict(pitch='deg',roll='deg',yaw='deg',x='mm',y='mm',z='mm')
    self.j = j; self.u = u; self.X = X
    print '%0.1f sec' % (time() - ti)
    s = util.Struct(X0=X0,Qd=Qd,ord=ord,b=b,j=j,u=u)
    dir = os.path.join(di,ddir)
    if not os.path.exists( dir ):
      os.mkdir( dir )
    s.write( os.path.join(dir,fi+'_ukf.py') )
    np.savez(os.path.join(dir,fi+'_ukf.npz'),X=X,hz=hz)

    return X

  def plt(self,fmts=fmts,plts=['3d','pd','xyz0','xyz','dxyz','pry','exp'],
          save=True,**kwds):
    """
    Plot trajectory data 
    Inputs:
      (optional)
      fmts - [str,...] - list of formats to export figures
      plts - [str,...] - list of plots to generate

    Effects:
      - generates & saves plots
    """
    di,fi = os.path.split(self.fi)
    dir = os.path.join(di,pdir)
    if save:
      if not os.path.exists( dir ):
        os.mkdir( dir )
    F = 1 # figure counter
    # processed marker data
    if self.d is not None:
      t = self.t; d = self.d
      # 3d
      if '3d' in plts:
        fig = plt.figure(F); fig.clf(); F += 1
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(d[::10,:,0],
                   d[::10,:,1],
                   d[::10,:,2])
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
        #ax.set_title('floor normal = %s'%np.array_str(n,precision=2))
        #ax.view_init(elev=0.,azim=-115.)
        ax.view_init(elev=90.,azim=90.)
        if save:
          for fmt in fmts:
            fig.savefig(os.path.join(di,pdir,fi+'_dat-3d.'+fmt))
      # xyz0
      if 'xyz0' in plts:
        fig = plt.figure(F); fig.clf(); F += 1
        ax = fig.add_subplot(311); ax.grid('on')
        ax.set_title('$x$, $y$, $z$ plot')
        ax.plot(t,d[...,0])
        ax.set_ylabel('$x$ (mm)')
        ax = fig.add_subplot(312); ax.grid('on')
        ax.plot(t,d[...,1])
        ax.set_ylabel('$y$ (mm)')
        ax = fig.add_subplot(313); ax.grid('on')
        ax.plot(t,d[...,2])
        ax.set_ylabel('$z$ (mm)')
        ax.set_xlabel('time (sec)')
        if save:
          for fmt in fmts:
            fig.savefig(os.path.join(di,pdir,fi+'_dat-xyz0.'+fmt))
    if hasattr(self,'pd0') and self.pd0 is not None and self.d0 is not None:
      pd0 = self.pd0; d0 = self.d0
      # pd
      if 'pd' in plts:
        fig = plt.figure(F); fig.clf(); F += 1
        ax = fig.add_subplot(111); ax.grid('on')
        ax.plot(t,pd0 - d0);
        ax.set_ylim(-5,5)
        ax.set_xlabel('time (sec)'); ax.set_ylabel('distance (mm)')
        ax.set_title('pairwise distances')
        if save:
          for fmt in fmts:
            fig.savefig(os.path.join(di,pdir,fi+'_dat-pd.'+fmt))
      if 'pdhist' in plts and self.X is not None:
        t = self.t; X = self.X; N,_ = X.shape; j = self.j; u = self.u
        x = X[...,j['x']]; y = X[...,j['y']];
        e = np.abs( pd0[:x.size,0] - d0[0] )
        nn = ((1 - np.isnan(e)) * (1 - np.isnan(x))).nonzero()
        x = x[nn]; y = y[nn]; e = e[nn]
        N = 10; de = 5; dd = 1000
        bins = [ 
                 #np.linspace(x.min(),x.max(),num=N),
                 #np.linspace(y.min(),y.max(),num=N),
                 np.linspace(-dd,dd,num=N),
                 np.linspace(-dd,dd,num=N),
                 np.linspace(0.,de,num=10*de) 
               ]
        samps = np.c_[x,y,e]
        H,_ = np.histogramdd( samps, bins )
        w = bins[2][1:]# + np.diff(bins[2]))
        im = np.sum( H * w, axis=2 )
        fig = plt.figure(F); fig.clf(); F += 1
        ax = fig.add_subplot(111)
        plt.imshow( im, interpolation='nearest' )
        ax.set_xticks( range(N-1)[::2] )
        ax.set_xticklabels(['%0.0f' % xe for xe in np.linspace(-dd,dd,num=N/2)])
        ax.set_yticks( range(N-1)[::2] )
        ax.set_yticklabels(['%0.0f' % xe for xe in np.linspace(-dd,dd,num=N/2)])
        ax.set_xlabel('$x$ (mm)'); ax.set_xlabel('$y$ (mm)')
        #1/0
    # ukf data
    if self.X is not None:
      t = self.t; X = self.X; N,_ = X.shape; j = self.j; u = self.u
      #s = util.Struct()
      #s.read(os.path.join(di,ddir,fi+'_ukf.py'),locals={'array':np.array})
      #j = s.j; u = s.u
      if self.d is not None and 'exp' in plts and hasattr(self,'pd0') and self.pd0 is not None and self.d0 is not None:
        pd0 = self.pd0; d0 = self.d0
        t = self.t; d = self.d
        hz = self.hz
        fig = plt.figure(F); fig.clf(); F += 1
        ax = fig.add_subplot(211); ax.grid('on')
        ax.set_title('%dhz' % hz)
        spd = np.sqrt(np.diff(X[...,j['x']])**2 + np.diff(X[...,j['y']])**2)*hz
        ax.plot(t[1:N],spd,'b')
        ax.set_ylabel('speed (%s / sec)'%u['x'])
        ax.set_ylim(-100.,2100.)
        ax = fig.add_subplot(212); ax.grid('on')
        ax.plot(t,pd0 - d0);
        ax.set_ylim(-5,5)
        ax.set_ylabel('distance (mm)')
        ax.set_xlabel('time (sec)')
        if save:
          for fmt in fmts:
            fig.savefig(os.path.join(di,pdir,fi+'_ukf-exp.'+fmt))
      # xyz
      if 'xyz' in plts:
        fig = plt.figure(F); fig.clf(); F += 1
        ax = fig.add_subplot(311); ax.grid('on')
        ax.set_title('$x$, $y$, $z$ plot')
        ax.plot(t[:N],X[...,j['x']],'b')
        ax.set_ylabel('$x$ (%s)'%u['x'])
        ax = fig.add_subplot(312); ax.grid('on')
        ax.plot(t[:N],X[...,j['y']],'g')
        ax.set_ylabel('$y$ (%s)'%u['y'])
        ax = fig.add_subplot(313); ax.grid('on')
        ax.plot(t[:N],X[...,j['z']],'r')
        ax.set_ylabel('$z$ (%s)'%u['z'])
        ax.set_xlabel('time (sec)')
        if save:
          for fmt in fmts:
            fig.savefig(os.path.join(di,pdir,fi+'_ukf-xyz.'+fmt))
      # pry
      if 'pry' in plts:
        fig = plt.figure(F); fig.clf(); F += 1
        ax = fig.add_subplot(311); ax.grid('on')
        ax.set_title('pitch, roll, yaw plot')
        ax.plot(t[:N],X[...,j['pitch']],'b')
        ax.set_ylabel('pitch (%s)'%u['pitch'])
        ax = fig.add_subplot(312); ax.grid('on')
        ax.plot(t[:N],X[...,j['roll']],'g')
        ax.set_ylabel('roll (%s)'%u['roll'])
        ax = fig.add_subplot(313); ax.grid('on')
        ax.plot(t[:N],X[...,j['yaw']],'r')
        ax.set_ylabel('yaw (%s)'%u['yaw'])
        ax.set_xlabel('time (sec)')
        if save:
          for fmt in fmts:
            fig.savefig(os.path.join(di,pdir,fi+'_ukf-rpy.'+fmt))
      if X.shape[1] >= 12:
        # xyz
        if 'dxyz' in plts:
          fig = plt.figure(F); fig.clf(); F += 1
          ax = fig.add_subplot(311); ax.grid('on')
          ax.set_title('$\dot{x}$, $\dot{y}$, $\dot{z}$ plot')
          ax.plot(t[:N],X[...,j['x']+6],'b')
          ax.set_ylabel('$\dot{x}$ (%s/sample)'%u['x'])
          ax = fig.add_subplot(312); ax.grid('on')
          ax.plot(t[:N],X[...,j['y']+6],'g')
          ax.set_ylabel('$\dot{y}$ (%s/sample)'%u['y'])
          ax = fig.add_subplot(313); ax.grid('on')
          ax.plot(t[:N],X[...,j['z']+6],'r')
          ax.set_ylabel('$\dot{z}$ (%s/sample)'%u['z'])
          ax.set_xlabel('time (sec)')
          if save:
            for fmt in fmts:
              fig.savefig(os.path.join(di,pdir,fi+'_ukf-dxyz.'+fmt))

    plt.show()


if __name__ == '__main__':

  rb = Rbt('square/20120619-1140')
  rb.dat()
  rb.ukf(viz=100,ord=0)
  rb.load()
  rb.plt()


