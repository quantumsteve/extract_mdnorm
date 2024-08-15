# modify and insert into Garnet or Mantid script before calling MDNorm

SaveMD(InputWorkspace='__md', Filename=f'/tmp/garnet_reduction/CORELLI_{n}_BEFORE_MDNorm.nxs')

import h5py
f = h5py.File(f'/tmp/garnet_reduction/CORELLI_{n}_extra_params.hdf5','w')
md_ws = mtd['__md']
numexpinfo = md_ws.getNumExperimentInfo()
for i in range(numexpinfo):
  grp = f.create_group(f'expinfo_{i}')
  numgoniometers = md_ws.getExperimentInfo(i).run().getNumGoniometers()
  for j in range(numgoniometers):
    print(numexpinfo, numgoniometers, md_ws.getExperimentInfo(i).run().getGoniometer(j).getR())
    dset = grp.create_dataset(f'goniometer_{j}', data=md_ws.getExperimentInfo(i).run().getGoniometer(j).getR(), dtype=np.float32)

f.create_dataset('ubmatrix', data=md_ws.getExperimentInfo(0).sample().getOrientedLattice().getUB()*2.*np.pi, dtype=np.float32)
print(md_ws.getExperimentInfo(0).sample().getOrientedLattice().getUB()*2.*np.pi)

from mantid.geometry import SpaceGroupFactory,PointGroupFactory
spaceGroup = SpaceGroupFactory.createSpaceGroup(self.getProperty("SymmetryOperations").value)
pointGroup = spaceGroup.getPointGroup()
#pointGroup = PointGroupFactory.createPointGroup(symmetry)
symmetryOps = pointGroup.getSymmetryOperations()
grp = f.create_group('symmetryOps')
for i,so in enumerate(symmetryOps):
  mat = np.zeros((3,3))
  mat[:,0] = so.transformHKL([1,0,0])
  mat[:,1] = so.transformHKL([0,1,0])
  mat[:,2] = so.transformHKL([0,0,1])
  mat = np.linalg.inv(mat)
  grp.create_dataset(f'op_{i}',data=mat, dtype=np.float32)
  print(mat)

info = md_ws.getExperimentInfo(0).spectrumInfo()
ndet = info.detectorCount()
skip_dets = np.zeros(ndet,dtype=bool)
for i in range(ndet):
  skip_dets[i] = not info.hasDetectors(i) or info.isMonitor(i) or info.isMasked(i)

f.create_dataset('skip_dets', data=skip_dets)
f.close()

