import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsummary import summary  # Print model / parameter details

import numpy as np
from tqdm import tqdm
import wandb
import os
import datetime
import time
import shutil
import importlib
import argparse
import json
# import matplotlib.pyplot as plt
# import matplotlib.animation as ani

# Pretty tracebacks
# import rich.traceback
# rich.traceback.install()

import sys
import signal
from multiprocessing import shared_memory as sm


def load_data(path):
    datafile = np.load(path)
    data_keep = {}

    for dr in datafile.keys():
        if dr not in ['36', '37', '38', '39']:
            data_keep[dr] = datafile[dr]

    data = np.concatenate([data_keep[dr] for dr in data_keep.keys()], axis=0)

    # data = datafile['signals']
    # msi = datafile['msi']
    # pro = datafile['pro']
    # pos = datafile['pos']
    # ffc_5x60 = datafile['ffc_5x60']
    # ffc_3x15 = datafile['ffc_3x15']
    # ffc_flattened = ffc_5x60.reshape(-1, 101, 5 * 60)
    # ffc_flattened = ffc_3x15.reshape(-1, 101, 3 * 15)
    # ffc_flattened = np.zeros_like(ffc_flattened)

    num_examples = len(data)

    # Dimensions are exmaple, time, info
    time_signals = np.concatenate([data['diode_0'][:, :, np.newaxis] / np.max(np.abs(data['diode_0'])),
                                   data['diode_1'][:, :, np.newaxis] / np.max(np.abs(data['diode_1'])),
                                   data['diode_2'][:, :, np.newaxis] / np.max(np.abs(data['diode_2'])),
                                   data['diode_3'][:, :, np.newaxis] / np.max(np.abs(data['diode_3'])),
                                   data['diode_4'][:, :, np.newaxis] / np.max(np.abs(data['diode_4'])),
                                   data['discharge_current'][:, :, np.newaxis] / np.max(np.abs(data['discharge_current'])),
                                   data['discharge_voltage'][:, :, np.newaxis] / np.max(np.abs(data['discharge_voltage'])),
                                   data['interferometer'][:, :, np.newaxis] / np.max(np.abs(data['interferometer'])),
                                   # data['isat'][:, :, np.newaxis] / np.max(np.abs(data['isat'])),
                                   # ffc_flattened / np.max(np.abs(ffc_flattened)),
                                   # ffc_flattened / 1.0,
                                   ], axis=2)

    # pressures = np.nan_to_num(msi['pressures'], nan=-9)
    pressures = data['pressures']
    positions = data['positions']
    isat = data['isat']
    non_time_signals = np.concatenate([data['magnet_profile'] / np.max(np.abs(data['magnet_profile'])),
                                       pressures / np.max(np.abs(pressures)),
                                       positions[:] / np.max(np.abs(positions[:]))], axis=1)[:, :]
    # non_time_signals = np.tile(non_time_signals, (1, 1, 1))

    # Total size of time signals: 8*101 + 5*60*101 = 31108
    # if using ffc_3x15: 8*101 + 3*15*101 = 5353
    # Flatten so we can concat non_time_signals on the back
    time_signals = time_signals.reshape(num_examples, -1)

    all_signals = np.concatenate([time_signals,
                                  non_time_signals], axis=1)
    # all_signals = all_signals.reshape(num_examples, -1)
    # get the offset of the probe prediction, basically
    x_len = all_signals.shape[1]

    all_signals = np.concatenate([all_signals, isat / np.max(np.abs(isat))], axis=1)

    data = torch.tensor(all_signals, dtype=torch.float)

    return data, x_len


class ModelClass(torch.nn.Module):
    def __init__(self, hyperparams):
        super(ModelClass, self).__init__()
        # CNN setup
        # self.seq_length = seq_length
        # self.embed_dim = out_channels = embed_dim
        # self.kernel_size = kernel_size

        self.act = torch.nn.SiLU()

        # self.conv1 = torch.nn.LazyConv2d(16, kernel_size=(5, 430), padding='valid')
        self.conv1 = torch.nn.LazyConv2d(16, kernel_size=(5, 130), padding='valid')
        self.conv2 = torch.nn.LazyConv2d(16, kernel_size=(21, 1), padding='same')
        self.conv3 = torch.nn.LazyConv2d(16, kernel_size=(5, 1), padding='same')
        self.conv4 = torch.nn.LazyConv2d(16, kernel_size=(21, 1), padding='same')
        self.conv5 = torch.nn.LazyConv2d(16, kernel_size=(5, 1), padding='same')
        self.conv6 = torch.nn.LazyConv2d(16, kernel_size=(21, 1), padding='same')
        self.conv7 = torch.nn.LazyConv2d(16, kernel_size=(5, 1), padding='same')
        self.conv8 = torch.nn.LazyConv2d(16, kernel_size=(21, 1), padding='same')
        self.conv9 = torch.nn.LazyConv2d(16, kernel_size=(5, 1), padding='same')
        self.conv10 = torch.nn.LazyConv2d(16, kernel_size=(21, 1), padding='same')

        self.dense1 = torch.nn.LazyLinear(128)
        # self.dense2 = torch.nn.LazyLinear(128)
        self.dense6 = torch.nn.LazyLinear(76)

        # Sine and cosine for positional encoding
        # self.x_sin = torch.sin(torch.arange(101) / 101 * 2 * np.pi)
        # self.x_cos = torch.cos(torch.arange(101) / 101 * 2 * np.pi)
        self.register_buffer('x_sin', torch.sin(torch.arange(101) / 101 * 2 * np.pi))
        self.register_buffer('x_cos', torch.cos(torch.arange(101) / 101 * 2 * np.pi))

    def forward(self, x):
        # x_time = x[:, 0:5353].reshape(-1, 101, 308)
        # x_notime = torch.tile(x[:, 5353:].reshape(-1, 1, 64 + 51 + 5), dims=(1, 101, 1))
        # x_time = x[:, 0:5353].reshape(-1, 101, 53)
        # x_notime = torch.tile(x[:, 5353:].reshape(-1, 1, 64 + 51 + 5), dims=(1, 101, 1))
        x_time = x[:, 0:808].reshape(-1, 101, 8)
        x_notime = torch.tile(x[:, 808:].reshape(-1, 1, 64 + 51 + 5), dims=(1, 101, 1))

        x = torch.cat([x_time, x_notime,
                       torch.tile(self.x_sin.reshape(1, 101, 1), dims=(x.shape[0], 1, 1)),
                       torch.tile(self.x_cos.reshape(1, 101, 1), dims=(x.shape[0], 1, 1))],
                      dim=2)
        # Add channels dimension
        x = torch.unsqueeze(x, dim=1)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        x = self.conv5(x)
        x = self.act(x)
        x = self.conv6(x)
        x = self.act(x)
        x = self.conv7(x)
        x = self.act(x)
        x = self.conv8(x)
        x = self.act(x)
        x = self.conv9(x)
        x = self.act(x)
        x = self.conv10(x)
        x = self.act(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dense1(x)
        x = self.act(x)
        # x = self.dense2(x)
        # x = self.act(x)
        x = self.dense6(x)
        return x


# A comprehensive guide to distributed data parallel:
# https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def main(rank, world_size, hyperparams, port):
    # Distributed setup
    setup(rank, world_size, port)
    sh_mem = sm.SharedMemory(name="exit_mem_{}".format(os.getppid()),)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Handle directories, file copying, etc...
    identifier = datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss')
    hyperparams['identifier'] = identifier
    project_name = "train_cnn"
    exp_path = "training_runs/"
    path = exp_path + identifier
    if rank == 0:
        os.mkdir(path)
        os.mkdir(path + "/checkpoints")
        os.mkdir(path + "/plots")
        shutil.copy(project_name + ".py", path + "/" + project_name + "_copy.py")

        with open(path + "/" + "hyperparams.json", 'w') as json_f:
            json.dump(hyperparams, json_f)

    # Set local hyperparameter variables
    num_epochs = hyperparams["num_epochs"]
    batch_size_max = hyperparams["batch_size_max"]
    lr = hyperparams["lr"]
    momentum = hyperparams["momentum"]
    weight_decay = hyperparams["weight_decay"]
    resume = hyperparams["resume"]
    if resume:
        resume_path = hyperparams["resume_path"]
        resume_version = hyperparams["resume_version"]

    # For writing to Tensorboard
    writer = SummaryWriter(log_dir=path)

    # Creating/loading model
    model = ModelClass(hyperparams)
    if resume:
        with open(exp_path + resume_path + "/" + "hyperparams.json") as json_f:
            hyperparams_temp = json.loads(json_f.read())
        spec = importlib.util.spec_from_file_location(project_name + "_copy", exp_path +
                                                      resume_path + "/" + project_name + "_copy.py")
        loaded_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded_module)
        model = loaded_module.ModelClass(hyperparams_temp)

    # Load data
    if rank == 0:
        print("Loading data: " + path, flush=True)
    data_path = "datasets/" + hyperparams["dataset"]
    # Include both x and y; need to split in the training loop
    data, x_len = load_data(data_path)

    print(x_len)

    num_examples = data.shape[0]
    data_size = data.shape[1]
    if rank == 0:
        print("Data shape: ", end="")
        print(data.shape, flush=True)

    # initialze for lazy layers so that the num_parameters works properly
    model = model.to(rank)
    model(torch.zeros((2, data_size))[:, 0:x_len].to(rank))
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Start sampler, dataloader, and split the data
    generator1 = torch.Generator().manual_seed(42)
    train_data, test_data = torch.utils.data.random_split(data,
                                              [int(np.floor(0.8 * num_examples)), int(np.ceil(0.2 * num_examples))],
                                               generator=generator1)
    # train_data = data_list[0]
    test_data = torch.stack([t for t in test_data]).to(rank)
    sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size,
                                                              rank=rank, shuffle=True,
                                                              drop_last=False)
    dataloader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size_max, shuffle=False,
                                             num_workers=4, pin_memory=True, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.0, 0.999))
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                    momentum=momentum, nesterov=False)
    del data

    # Resume and laod the weights and optimizer state
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if resume:
        ckpt = torch.load(exp_path + resume_path + "/" + resume_version + ".pt", map_location=map_location)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    num_batches = len(dataloader)

    # Define learnign rate / warmup schedule for the model and if resuming, load the state
    def lr_func(x):
        return 1.0
        # if x <= 500:
        #     return 1.0
        # else:
        #     return 1 / torch.sqrt(torch.tensor(x - 500)).to(rank)
    lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    if resume:
        lrScheduler.load_state_dict(ckpt['lrScheduler_state_dict'])

    # Print number of parameters
    if rank == 0:
        # summary(model, (data_size,), batch_size=batch_size_max)
        num_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
        print("Parameters: {}".format(num_parameters))
        for name, module in model.named_modules():
            print(name, sum(param.numel() for param in module.parameters()))
        hyperparams['num_parameters'] = num_parameters
    # Initialize weights and biases
        wandb.init(project="profile-predict", entity='phil',
                   group="", job_type="",
                   config=hyperparams)

    # print(test_data.shape)

    t_start0 = t_start1 = t_start2 = t_start_autoclose = time.time()
    pbar = tqdm(total=num_epochs)
    batch_iteration = 0
    # grad_mag_list = []
    if resume:
        batch_iteration = ckpt['batch_iteration']
    model.train(True)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # If all the data fits in GPU memory lets just keep it there
    # examples_gpu = [i.to(rank) for i in dataloader]
    for epoch in range(num_epochs):
        # with torch.cuda.amp.autocast():s

        batch_pbar = tqdm(total=num_batches)
        # If all data does not fit on the GPU
        for examples, i in zip(dataloader, range(num_batches)):
        # for i in range(num_batches):
            # x has shape num_examples, 101, length of data
            examples = examples.to(rank)
            # examples = examples_gpu[i]

            x = examples[:, 0:x_len]
            y = examples[:, x_len:]

            # with torch.autograd.detect_anomaly():
            with torch.cuda.amp.autocast(enabled=False):

                # Backwards pass...
                optimizer.zero_grad()
                output = model(x)

                # loss = torch.nn.MSELoss(output, y)  # reduction = mean by default
                loss = torch.mean((output - y) ** 2)

            # loss.backward()
            scaler.scale(loss).backward()

            # For debugging missing gradients error
            # for name, p in model.named_parameters():
            #     if p.grad is None:
            #         print("found unused param: ")
            #         print(name)
            #         print(p)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            lrScheduler.step()

            # print(linearScheduler.get_last_lr())
            # print(sqrtScheduler.get_last_lr())

            if rank == 0:
                # print("\n")
                # print(reg_avg)
                loss = loss.detach()
                rmse = torch.sqrt(loss)
                loss_test = torch.mean((model(test_data[:, 0:x_len]) - test_data[:, x_len:]) ** 2)
                if i % 20 == 0:
                    tqdm.write("#: {} // L: {:.2e} // L_test: {:.2e} // RMSE: {:.2e}".format(i, loss, loss_test, rmse))
                wandb.log({"loss/train": loss,
                           "loss/test": loss_test,
                           "loss/rmse": rmse,
                           "batch_num": batch_iteration,
                           "epoch": epoch})

            # Longer-term metrics
            if rank == 0:
                # End training after fixed amount of time
                # if time.time() - t_start_autoclose > 3600 * hyperparams['time_limit']:
                #     sh_mem.buf[0] = 1

                # Log to tensorboard every 5 min
                if (epoch == 0 and i == 3) or time.time() - t_start0 > 300 or sh_mem.buf[0] == 1:
                    t_start0 = time.time()
                    # write scalars and histograms
                    writer.add_scalar("loss/train", loss, batch_iteration)  # used to be loss_avg
                    writer.add_scalar("loss/test", loss_test, batch_iteration)

                # Add histogram every 20 min
                if (epoch == 0 and i == 3) or time.time() - t_start1 > 1200 or sh_mem.buf[0] == 1:
                    t_start1 = time.time()
                    try:
                        for name, weight in model.named_parameters():
                            writer.add_histogram("w/" + name, weight, batch_iteration)
                            writer.add_histogram(f'g/{name}.grad', weight.grad, batch_iteration)
                    except Exception as e:
                        print(e)
                    writer.flush()
                    tqdm.write("E: {} // L: {:.2e} //".format(epoch, loss))  # used to be loss_avg

                # EVERY FIVE MIN
                # Save checkpoint every hour
                if ((epoch == 0 and i == 3) or (epoch == num_epochs - 1 and i == num_batches - 1)
                    or time.time() - t_start2 > 300 or sh_mem.buf[0] == 1):
                    t_start2 = time.time()
                    torch.save({'epoch': epoch,
                                'batch_iteration': batch_iteration,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lrScheduler_state_dict': lrScheduler.state_dict(),
                                },
                               path + "/checkpoints/model-{}-{}.pt".format(epoch, i))

            batch_iteration += 1
            batch_pbar.update(1)

            if sh_mem.buf[0] == 1:
                print("\n ---- Exiting process ----\n")
                break
        batch_pbar.close()
        pbar.update(1)
        if sh_mem.buf[0] == 1:
            break
    pbar.close()
    sh_mem.close()
    if rank == 0:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the modular EBM")
    parser.add_argument('--num_epochs', type=int)

    parser.add_argument('--batch_size_max', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float)

    parser.add_argument('--weight_decay', type=float)
    # parser.add_argument('--identifier')

    parser.add_argument('--resume', type=bool)
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--resume_version', type=str)

    parser.add_argument('--dataset', type=str)

    parser.add_argument('--time_limit', type=float, default=-1,
                        help='Time limit (in hours). -1 for unlimited')
    parser.add_argument('--port', type=int, default=26000)
    args = parser.parse_args()

    hyperparams = {
        "num_epochs": 1000,

        "batch_size_max": 128,
        "lr": 3e-4,
        "momentum": 0.99,

        "weight_decay": 0e-5,
        # "identifier": identifier,
        "resume": False,
        "resume_path": None,
        "resume_version": None,
        'time_limit': -1,

        'dataset': "msi_isat_02.npz",
    }

    for key in vars(args).keys():
        if vars(args)[key] is not None and key != 'port':
            hyperparams[key] = vars(args)[key]

    world_size = 1
    sh_mem = sm.SharedMemory(name="exit_mem_{}".format(os.getpid()), create=True, size=1)
    sh_mem.buf[0] = 0
    try:
        proc_context = torch.multiprocessing.spawn(main, args=(world_size, hyperparams, args.port),
                                                   nprocs=world_size, join=False)
        proc_context.join()
    except KeyboardInterrupt:
        sh_mem.buf[0] = 1
        proc_context.join(timeout=30)
        sh_mem.unlink()
    else:
        sh_mem.unlink()
