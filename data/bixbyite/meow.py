import h5py


for run in range(40704,40726):
  run_str = str(run);
  print(run_str)
  f = h5py.File('TOPAZ_'+run_str+'_BEFORE_MDNorm.nxs','r')
  event_data = f['MDEventWorkspace']['event_data']['event_data']
  print(event_data.shape, event_data.dtype)
  g = h5py.File('TOPAZ_'+run_str+'_events.nxs','w')
  grp = g.create_group('MDEventWorkspace/event_data');
  event_weights = grp.create_dataset("weights", (event_data.shape[0],), dtype='float32', data=event_data[:,0])
  event_position = grp.create_dataset("position", (3, event_data.shape[0]), dtype='float32')
  event_position[0,:] = event_data[:,5]
  event_position[1,:] = event_data[:,6]
  event_position[2,:] = event_data[:,7]

  gd_prtn_chrg = f['MDEventWorkspace/experiment0/logs/gd_prtn_chrg/value']
  grp = g.create_group('MDEventWorkspace/experiment0/logs/gd_prtn_chrg')
  grp.create_dataset('value',data=gd_prtn_chrg)

