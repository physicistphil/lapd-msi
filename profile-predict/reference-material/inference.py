model_path = "2024-04-05_15h-51m-10s"
model_version = "checkpoints/model-928-106"

# sys.path.append("/home/phil/Desktop/EBMs/lapd-ebm/experiments_modular/" + model_path + "/")
# os.chdir("/home/phil/Desktop/EBMs/lapd-ebm/experiments_modular/" + model_path "/")

os.chdir("/home/phil/Desktop/profile-predict/training_runs/" + model_path + "/")

print(sys.path)

spec = importlib.util.spec_from_file_location("train_cnn_copy", "train_cnn_copy.py")
loaded_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loaded_module)
with open("hyperparams.json") as json_f:
    hyperparams = json.loads(json_f.read())

# imp.reload(ebm)
model = loaded_module.ModelClass(hyperparams).to(device)
# ckpt = torch.load("experiments_modular/" + model_path + "/" + model_version + ".pt")
ckpt = torch.load(model_version + ".pt")

model_dict = OrderedDict()
pattern = re.compile('module.')
state_dict = ckpt['model_state_dict']
for k,v in state_dict.items():
    if re.search("module", k):
        model_dict[re.sub(pattern, '', k)] = v
    else:
        model_dict = state_dict
model.load_state_dict(model_dict, strict=True)

data_path = "/home/phil/Desktop/profile-predict/datasets/" + hyperparams['dataset']
data, x_len = loaded_module.load_data(data_path)

print("Number of parameters: {}".format(np.sum([p.numel() for p in model.parameters() if p.requires_grad])))

##################
pos = np.load(data_path)
pos = np.concatenate([pos[dr]['positions'] for dr in pos.keys()], axis=0)

##################
dr_idx = {}
n_shots = 0
dataset = np.load(data_path)
for dr in dataset.keys():
    dr_idx[dr] = (n_shots, n_shots + len(dataset[dr]))
    n_shots += len(dataset[dr])


##################
model = model.cpu()
prediction = model(data[:, 0:x_len])


##################
tp_start = 41
tp_end = 42
pred = prediction.cpu().detach().numpy()[:, tp_start:tp_end]
data_show = data[:, x_len + tp_start:x_len + tp_end].numpy()