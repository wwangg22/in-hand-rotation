import torch
filepath = 'demonstration-baoding-2/teacher_batch_0_2_0.pt'
filepath2 = 'demonstration-baoding-2/dagger_batch_1_3.pt'
obs, actions, sigmas, pointcloud  = torch.load(filepath, map_location="cpu")  # stays on CPU to be safe
obs2, actions2, sigmas2, pointcloud2 = torch.load(filepath2, map_location="cpu")  # stays on CPU to be safe
# print(obs.shape, actions.shape, sigmas.shape, pointcloud.shape)
# print(obs2.shape, actions2.shape, sigmas2.shape, pointcloud2.shape)
#print difference in values of obs, actions, sigmas, pointcloud
print("obs difference:", torch.abs(obs - obs2).mean())
print("obs shape difference:", obs.shape, obs2.shape)
print("actions difference:", torch.abs(actions - actions2).mean())
print("file path 1 avg actions:", actions.mean())
print("file path 2 avg actions:", actions2.mean())
#max actions
print("max actions file path 1:", actions.max())
print("max actions file path 2:", actions2.max())
#min actions
print("min actions file path 1:", actions.min())
print("min actions file path 2:", actions2.min())
print("sigmas difference:", torch.abs(sigmas - sigmas2).mean())
print("pointcloud difference:", torch.abs(pointcloud - pointcloud2).mean())
#print cloud size difference
print("pointcloud shape difference:", pointcloud.shape, pointcloud2.shape)
