import h5py
import numpy as np

for run in range(0,36):
  run_str = str(run);
  print(run_str)
  f = h5py.File('CORELLI_'+run_str+'_BEFORE_MDNorm.nxs','r')
  event_data = f['MDEventWorkspace']['event_data']['event_data']
  print(event_data.shape, event_data.dtype)
  g = h5py.File('CORELLI_'+run_str+'_events.nxs','w')
  grp = g.create_group('MDEventWorkspace/event_data');
  event_weights = grp.create_dataset("weights", (event_data.shape[0],), dtype='float32', data=event_data[:,0])
  event_position = grp.create_dataset("position", (3, event_data.shape[0]), dtype='float32')
  event_position[0,:] = event_data[:,5]
  event_position[1,:] = event_data[:,6]
  event_position[2,:] = event_data[:,7]

  boxtype_data = f['MDEventWorkspace']['box_structure']['box_type']
  print(boxtype_data.shape, boxtype_data.dtype)
  grp = g.create_group('MDEventWorkspace/box_structure');
  box_type = grp.create_dataset("box_type", data=boxtype_data.astype(np.uint8))
  #box_type = boxtype_data

  extent_data = f['MDEventWorkspace']['box_structure']['extents']
  print(extent_data.shape, extent_data.dtype)
  #grp = g.create_group('MDEventWorkspace/box_structure');
  extents = grp.create_dataset("extents", (6, extent_data.shape[0]), dtype='float32', data=np.transpose(extent_data))

  signal_data = f['MDEventWorkspace']['box_structure']['box_signal_errorsquared']
  print(signal_data.shape, signal_data.dtype)
  #grp = g.create_group('MDEventWorkspace/event_data');
  signal = grp.create_dataset("box_signal", data=signal_data[:, 0])
  #errorsquared = grp.create_dataset("box_errorsquared", (signal_data.shape[0],), dtype='float32', data=signal_data[:,1])

  index_data = f['MDEventWorkspace']['box_structure']['box_event_index']
  print(index_data.shape, index_data.dtype)
  #grp = g.create_group('MDEventWorkspace/box_structure');
  index = grp.create_dataset("box_event_index", data=np.transpose(index_data))

  gd_prtn_chrg = f['MDEventWorkspace/experiment0/logs/gd_prtn_chrg/value']
  grp = g.create_group('MDEventWorkspace/experiment0/logs/gd_prtn_chrg')
  grp.create_dataset('value',data=gd_prtn_chrg)

