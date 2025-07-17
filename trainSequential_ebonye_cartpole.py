import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from random import randint
import uuid
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import wandb
import pickle
import random
import numpy as np
import torch
import gc
import json
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_scheduler

# Relative imports
import model.tasks as tasks
from model.curriculum import Curriculum
from model.schema import schema
from model.models import build_model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True


def log_training_info(file_path, i, args, xs, ys, output, loss):
    """
    Logs training information to txt file

    Args:
        file_path (str): This will be path of where model is saved.
        i (int): The current training iteration.
        args (Namespace): schema arguments
        xs (torch.Tensor): These are theta and thetadot values
        ys (torch.Tensor): The are ground truth control u values
        output (torch.Tensor): These are predicted control u values.
        output_states (torch.Tensor): These are predicted theta and thetadot values.
        loss (torch.Tensor): loss for model update.

    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(f"Iteration {i} - {args.model_logger_textfile}\n")
        f.write(f"xs\n{xs[0].detach().cpu().numpy()}\n\n")
        f.write(f"ys\n{ys[0].detach().cpu().numpy()}\n\n")
        f.write(f"output\n{output[0].detach().cpu().numpy()}\n\n")
        # f.write(f"output_states\n{output_states[0]}\n\n")
        f.write(f"Loss ---- {loss.item()}\n\n\n\n")
        # f.write(f"lyapunov_loss_value ---- {lyapunov_loss_value.item()}\n\n\n\n")





def train_step(model, xs, ys, optimizer, loss_func, i, args, numtrainingsteps, b=0.5, g=9.81):
    """
    Performs a single training step for the model, including forward pass, loss calculation, 
    backpropagation, and optimizer update. Also logs every 500 iteration.

    Args:
        model (torch.nn.Module): GPT2 decoder.
        xs (torch.Tensor): These are theta and thetadot values
        ys (torch.Tensor): These are ground truth control u values
        optimizer (torch.optim.Optimizer): Adam optimizer
        loss_func (callable): loss function that will be used for training loss.
        i (int): current training iteration.
        args (Namespace): schema arguments
        numtrainingsteps (int): number of training steps

    Returns:
        tuple: A tuple containing:
            - total_loss (float): total loss value for the current iteration.
            - output (torch.Tensor): The model's predicted outputs for the input data (don't really use this for anything).
    """
    optimizer.zero_grad()
    


    xs_scaled = xs
    ys_scaled = ys

    


    output_controls, output_states = model(xs_scaled, ys_scaled)
    


    output = [output_controls.detach(), output_states.detach()]

    # import pdb; pdb.set_trace()

    

    loss_controls = loss_func(output_controls.squeeze()[:,:-1], ys_scaled)
    # loss_controls = loss_func(output_controls.squeeze()[:,:-1], ys_scaled[..., 0])
    

    loss_states = loss_func(output_states[:,:-1], xs_scaled[:,1:])

    

    loss = loss_controls + loss_states
    


    total_loss = loss 
    total_loss = total_loss.to(xs.device).requires_grad_(True)

    

    file_path = os.path.join(args.out_dir, args.model_logger_textfile)
   
    total_loss.backward()
    old_grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
   

    optimizer.step()
    return total_loss.detach().item(), output, grad_norm, old_grad_norm

def count_files_in_folder(folder, prefix, suffix):
    """
    Counts the number of files in dataset folder that matches prefix and suffix.

    Args:
        folder (str): The path to the folder where the files are located.
        prefix (str): The prefix that the file names must start with.
        suffix (str): The file extention that the file must end with.

    Returns:
        int: the number of files in the folder.
    """
    return len([f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(suffix)])


def load_dataset_chunk(pickle_folder, start_idx, end_idx):
    """
    Loads chunk of dataset files from specified folder within a given index range, 
    and returns the data as a list of tuples.

    Args:
        pickle_folder (str): The path to folder containing the pickle files.
        start_idx (int): The starting index of the pickle files to load.
        end_idx (int): The ending index of the pickle files to load.

    Returns:
        list: A list of tuples, where each tuple contains:
            - xs (any): theta and thetadot values.
            - ys (any): control u values.
    """
    dataset = []
    # masses = []
    # lengths = []
    cartmasses = []
    polemasses = []
    polelengths = []
    dataset_full = []
    with tqdm(total=end_idx - start_idx + 1, desc=f"Loading files {start_idx}-{end_idx}", leave=False) as load_pbar:
        for i in range(start_idx, end_idx + 1):
            # pickle_path = os.path.join(pickle_folder, f"multipendulum_{i}.pkl")
            pickle_path = os.path.join(pickle_folder, f"batch_{i}.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    # xs, ys = pickle.load(f)
                    # xs, ys, mass, length = pickle.load(f)
                    xs, ys, cartmass, polemass, polelength = pickle.load(f)
                    # xs.to("cuda:3")
                    # ys.to("cuda:3")
                    dataset.append((xs, ys))
                    # masses.append(mass)
                    # lengths.append(length)
                    cartmasses.append(cartmass)
                    polemasses.append(polemass)
                    polelengths.append(polelength)
                    # dataset_full.append((xs, ys, mass, length))
                    dataset_full.append((xs, ys, cartmass, polemass, polelength))
            else:
                print(f"Pickle not found: {pickle_path}. Skipping...")
            load_pbar.update(1)
    # return dataset, dataset_full, masses, lengths
    return dataset, dataset_full, cartmasses, polemasses, polelengths


def load_dataset_full(pickle_folder):
    """
    Loads entire dataset from a specified folder

    Args:
        pickle_folder (str): The path to the folder containing the pickle files.

    Returns:
        list: A list of tuples, where each tuple contains:
            - xs (any): theta and thetadot values.
            - ys (any): control u values.
        list: A list of mass values for each trajectory.
        list: A list of length values for each trajectory.
    """    
    dataset = []
    cartmasses = []
    polemasses = []
    polelengths = []
    dataset_full = []
    # total_files = count_files_in_folder(pickle_folder, "multipendulum_", ".pkl")
    total_files = count_files_in_folder(pickle_folder, "batch_", ".pkl")
    with tqdm(total=total_files, desc="Loading all files", leave=False) as load_pbar:
        for i in range(total_files):
            # pickle_path = os.path.join(pickle_folder, f"multipendulum_{i}.pkl")
            pickle_path = os.path.join(pickle_folder, f"batch_{i}.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    # xs, ys, mass, length = pickle.load(f)
                    xs, ys, cartmass, polemass, polelength = pickle.load(f)
                    dataset.append((xs, ys))
                    # masses.append(mass)
                    # lengths.append(length)
                    cartmasses.append(cartmass)
                    polemasses.append(polemass)
                    polelengths.append(polelength)
                    # dataset_full.append((xs, ys, mass, length))
                    dataset_full.append((xs, ys, cartmass, polemass, polelength))
            else:
                print(f"Pickle not found: {pickle_path}. Skipping...")
            load_pbar.update(1)
    # return dataset, dataset_full, masses, lengths
    return dataset, dataset_full, cartmasses, polemasses, polelengths



def get_files_from_folder(folderpath):
    """ Get all .pkl files from a folder """
    return[os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.endswith(".pkl")]




# def evaluate_model(model, id_data, ood_data, loss_func):
def evaluate_model(model, id_data, loss_func):
    """
    Evaluates the model on the in-distribution and out-of-distribution data.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        id_data (list): The in-distribution data to evaluate the model on.\
        loss_func (callable): The loss function to use for evaluation.

    Returns:
        tuple: A tuple containing:
            - id_loss (float): The loss of the model on the in-distribution data.
    """
    model.eval()


    id_loss = 0.0
    total_samples_id = 0
    ood_loss = 0.0
    total_samples_ood = 0
    lambda_coeff2 = 1e-4

    # scaled_ys_id = log_scale_torch(id_data[1])
    # scaled_ys_id = (id_data[1] + 650)/(70 + 650)


    # id_data = TensorDataset(id_data[0], scaled_ys_id, torch.tensor(id_data[2]), torch.tensor(id_data[3]))
    # id_data = TensorDataset(id_data[0], id_data[1], torch.tensor(id_data[2]), torch.tensor(id_data[3]))
    id_data = TensorDataset(id_data[0], id_data[1], torch.tensor(id_data[2]), torch.tensor(id_data[3]), torch.tensor(id_data[4]))

    id_loader = DataLoader(id_data, batch_size=64, shuffle=False) #, collate_fn=collate_fn)
    # for xs, ys in id_loader:
    # for xs, ys, masses, lengths in id_loader:
    for xs, ys, cartmass, polemass, polelength in id_loader:
        with torch.no_grad():
            # print(f"xs: {xs.size()}")
            # print(f"ys: {ys.size()}")
            xs = torch.squeeze(xs)
            ys = torch.squeeze(ys)

            # import pdb; pdb.set_trace()
            # xs_scaled = batch_scaling(xs)
            # ys_scaled = batch_scaling(ys)

            # xs_scaled = batch_scaling_states(xs)
            # ys_scaled = batch_scaling_controls(ys)

            xs_scaled = xs
            ys_scaled = ys

            # print(f"xs: {xs.size()}")
            # print(f"ys: {ys.size()}")
            # xs = xs.cuda(3)
            # ys = ys.cuda(3)
            # context = torch.randint(low = 2, high = (xs.size(1)//4), size=(1,)).item()

            # output = model(xs, ys)
            # output_controls, output_states = model(xs, ys)
            output_controls, output_states = model(xs_scaled, ys_scaled)

            # loss_controls = loss_func(output_controls.squeeze(), ys)
            # loss_controls = loss_func(output_controls.squeeze(), ys_scaled)
            # loss_states = loss_func(output_states, xs)
            # loss_states = loss_func(output_states[:-1], xs_scaled[1:])

            loss_controls = loss_func(output_controls.squeeze()[:,:-1], ys_scaled)
    

            loss_states = loss_func(output_states[:,:-1], xs_scaled[:,1:])

            loss = loss_controls + loss_states
            # loss = loss_controls 


            # loss = loss_func(output[:, context:], ys[:, context:])
            # loss = loss_func(output, ys)
            batch_size = xs.size(0)
           
            id_loss += (loss.item() * batch_size)
            total_samples_id += batch_size
    # id_loss /= len(id_loader)
    id_loss /= total_samples_id

    model.train()
    return id_loss 


def train(model, args):
    """
    Trains a given model on a dataset using the specified arguments and configurations.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        args (Namespace): A configuration object containing training parameters and settings, including:
            - args.training.learning_rate (float): The learning rate for the optimizer.
            - args.loss (str): The name of the loss function to use (must be defined in the tasks module).
            - args.out_dir (str): Directory where training states and checkpoints will be saved.
            - args.dataset_filesfolder (str): Path to the folder containing dataset-related files.
            - args.pickle_folder (str): dataset_filesfolder subfolder name containing the pickled dataset files.
            - args.use_chunk (int): Number of chunks to divide the dataset for memory-efficient loading. default 1
            - args.wandb.log_every_steps (int): How often to log metrics
            - args.training.save_every_steps (int): How often to save model checkpoints
            - args.test_run (bool): If True, skips logging and checkpoint saving for debugging

    Raises:
        ValueError: If the specified loss function is not found in the `tasks` module.

    Notes:
        - The function supports chunk-based dataset loading if memory restraints, or loading full dataset into memory
        - Checkpoints and training states are saved here.
        - Wandb logs are done here.
    """
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=1e-4) #1e-2
    

    curriculum = Curriculum(args.training.curriculum)
    loss_function_name = args.loss
    loss_function = getattr(tasks, loss_function_name, None)
    if loss_function is None:
        raise ValueError(f"Unknown loss function: {loss_function_name}")

    state_path = os.path.join(args.out_dir, "state.pt")

    dataset_folder = args.dataset_filesfolder
    picklefolder = args.pickle_folder
    fullpicklepath = os.path.join(dataset_folder, picklefolder)

    



    current_step = 0


    batched_trajs_size = args.training.batch_size
    total_files = count_files_in_folder(fullpicklepath, "batch_", ".pkl")
    num_chunks = args.use_chunk  
    files_per_chunk = total_files // num_chunks
    remainder = total_files % num_chunks



    num_epochs = args.training.epochs
    start_epoch = 0

    
        
    ### ebonye 3/15/2025 Cosine Scheduler
    batch_size = 64 
    num_training_steps = num_epochs * total_files * batched_trajs_size // batch_size

    print(f"num_training_steps: {num_training_steps}")
    warmup_steps =int(0.01 * num_training_steps)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


    # for epoch in range(num_epochs):
    for epoch in range(start_epoch, num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
 
        
        chunk_to_resume = 0
        with tqdm(total=num_chunks-chunk_to_resume, desc="Chunk Progress") as chunk_pbar: ###ebonye 150
            for chunk_idx in range(chunk_to_resume, num_chunks):
                if args.use_chunk == 1:
                    
                    dataset, dataset_full, cartmasses, polemasses, polelengths = load_dataset_full(fullpicklepath)
                else:
                    start_idx = chunk_idx * files_per_chunk
                    end_idx = start_idx + files_per_chunk - 1
                    if chunk_idx == num_chunks - 1:  
                        end_idx += remainder
                    
                    dataset, dataset_full, cartmasses, polemasses, polelengths = load_dataset_chunk(fullpicklepath, start_idx, end_idx)

                print(f"loaded chunk {chunk_idx + 1}/{num_chunks}")

                

                ### reorganize dataset_full for dataloader
                all_xs = []
                all_ys = []
                # all_masses = []
                # all_lengths = []
                all_cartmasses = []
                all_polemasses = []
                all_polelengths = []

                # for xs, ys, mass, length in dataset_full:
                for xs, ys, cartmass, polemass, polelength in dataset_full:
                    all_xs.append(xs)
                    all_ys.append(ys)
                    # all_masses.append(torch.tensor(mass))
                    # all_lengths.append(torch.tensor(length))
                    all_cartmasses.append(torch.tensor(cartmass))
                    all_polemasses.append(torch.tensor(polemass))
                    all_polelengths.append(torch.tensor(polelength))


                
                xs_tensor = torch.cat(all_xs, dim=0)
                
                ys_tensor = torch.cat(all_ys, dim=0)
                
                cartmasses_tensor = torch.cat(all_cartmasses, dim=0)
                polemasses_tensor = torch.cat(all_polemasses, dim=0)
                polelengths_tensor = torch.cat(all_polelengths, dim=0)

                dataset_full = TensorDataset(xs_tensor, ys_tensor, cartmasses_tensor, polemasses_tensor, polelengths_tensor)


                dataloader = DataLoader(dataset_full, batch_size=batch_size, shuffle=True)   #, collate_fn=collate_fn)
                

                # with tqdm(total=len(dataset), desc=f"Training Chunk {chunk_idx + 1}/{num_chunks}") as pbar:
                with tqdm(total=len(dataloader), desc=f"Training Chunk {chunk_idx + 1}/{num_chunks}") as pbar:
                    # for xs, ys in dataset:
                    # for xs, ys, masses, lengths in dataloader:
                    for xs, ys, cartmass, polemass, polelength in dataloader:
                        


                        # loss, output, gradnorm = train_step(model, xs, ys, optimizer, loss_function, current_step, args)
                        # print(f"initial lr: {optimizer.param_groups[0]['lr']}")
                        loss, output, gradnorm, oldgradnorm = train_step(model, xs, ys, optimizer, loss_function, current_step, args, num_training_steps)
                        lr_scheduler.step()
                        # loss, output_actions, output_states, gradnorm = train_step_with_rk4(model, xs, ys, optimizer, loss_function, current_step, args, masses, lengths)
                        # import pdb; pdb.set_trace()
                        # print(f"loss: {loss}")
                        # print(f"Epoch {epoch + 1}/{num_epochs}, Step {current_step}, Loss: {loss}, Current LR: {lr_scheduler.get_lr()[0]}")
                        print(f"Epoch {epoch + 1}/{num_epochs}, Step {current_step}, Loss: {loss}, Current LR: {optimizer.param_groups[0]['lr']}, Old Grad Norm: {oldgradnorm}")
                        # import pdb; pdb.set_trace()

                        # print(f"Phase {phase}, Epoch {epoch + 1}/{epochs_per_phase[phase]}, Step {current_step}, Loss: {loss}")
                        current_step += 1
                        pbar.update(1)
                        # torch.cuda.empty_cache()
                        # gc.collect()



                        if current_step % args.wandb.log_every_steps == 0 and not args.test_run:
                            # avg_id_loss = evaluate_model(model, id_data, loss_function)
                            # avg_id_loss, avg_ood_loss = evaluate_model_with_rk4(model, id_data, ood_data, loss_function)
                            wandb.log(
                                {
                                    # "epoch": epoch + 1,
                                    # "epoch": overall_epochs,
                                    # "phase": phase,
                                    "step": current_step,
                                    "loss": loss,
                                    "grad_norm": gradnorm,
                                    # "id_loss": avg_id_loss,
                                    # "ood_loss": avg_ood_loss
                                }
                            )

                        curriculum.update()

                        if current_step % args.training.save_every_steps == 0 and not args.test_run:
                            training_state = {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "train_step": current_step,
                                "epoch": epoch+1,
                                # "epoch": overall_epochs,
                                "loss": loss,
                            }
                            torch.save(training_state, state_path)

                            # checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{current_step}.pt")
                            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch+1}_step{current_step}.pt")
                            # checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{overall_epochs}_step{current_step}.pt")
                            # checkpoint_path = os.path.join(args.out_dir, f"checkpoint_phase{phase}_epoch{epoch+1}_step{current_step}.pt")
                            torch.save(model.state_dict(), checkpoint_path)
                            # print(f"Checkpoint saved at step {current_step}: {checkpoint_path}")
                            print(f"Checkpoint saved at epoch {epoch+1}, step {current_step}: {checkpoint_path}")
                            # print(f"Checkpoint saved at epoch {overall_epochs}, step {current_step}: {checkpoint_path}")

                print(f"Chunk {chunk_idx + 1}/{num_chunks} finished. unloading dataset from memory...")
                del dataset_full
                del dataset
                # del dataset
                torch.cuda.empty_cache()
                gc.collect()
                chunk_pbar.update(1)
        print(f"============== Finished Epoch {epoch + 1}/{num_epochs} ==============\n")
        # print(f"============== Finished Epoch {epoch + 1}/{epochs_per_phase[phase]} of Phase {phase} ==============\n")
        # overall_epochs += 1
        # print(f"Overall Epochs: {overall_epochs}/{total_epochs}")

    ##### Final Checkpoint
    training_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_step": current_step,
        "epoch": epoch+1,
        # "epoch": overall_epochs,
        "loss": loss,
    }
    torch.save(training_state, state_path)
    checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch+1}_step{current_step}.pt")
    # checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{overall_epochs}_step{current_step}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    # print(f"Final Checkpoint saved at epoch {overall_epochs}, step {current_step}: {checkpoint_path}")
    print(f"Final Checkpoint saved at epoch {epoch+1}, step {current_step}: {checkpoint_path}")
                        

def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 10
    else:
        # ##### ebonye resume training run
        # if os.path.exists(os.path.join(args.out_dir, "wandb", "wandb-resume.json")):
        #     with open(os.path.join(args.out_dir, "wandb", "wandb-resume.json"), "r") as f:
        #         resume_info = json.load(f)
        #         run_id = resume_info.get("run_id", None)
        #         # args.training.resume_id = run_id #### ebonye
        # else:
        #     run_id = None

        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
            id=run_id if run_id is not None else None #### ebonye
        )

    model = build_model(args.model)
    device_ids = [0,1]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to('cuda:0')
    # model.cuda()



    # ### ebonye resume training run
    # if args.training.resume_id is not None:
    #     # checkpoint_path = os.path.join(args.out_dir, "checkpoint_epoch1_step202000.pt")
    #     checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch1_step80000.pt")
    #     print(f"checkpoint_path: {checkpoint_path}")

    #     state_path = os.path.join(args.out_dir, "state.pt")
    #     # if os.path.exists(checkpoint_path):
    #     if os.path.exists(state_path):
    #         # checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    #         state = torch.load(state_path, map_location='cuda:0')

           
    #         # model.load_state_dict(checkpoint['model_state_dict'])
    #         model.load_state_dict(state['model_state_dict'])
    #         # print(f"checkpoint keys: {checkpoint.keys()}")
    #         # import pdb; pdb.set_trace()
    #         # model.load_state_dict(checkpoint['model_state_dict'])
    #         # optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    #         optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=1e-2)



    #         # if 'optimizer_state_dict' in checkpoint:
    #         #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         if 'optimizer_state_dict' in state:
    #             optimizer.load_state_dict(state['optimizer_state_dict'])

    #         start_step = state.get('train_step', 0) + 1
    #         loss = state.get('loss', 0.0)
    #         # start_step = checkpoint.get('train_step', 0) + 1
    #         # loss = checkpoint.get('loss', 0.0)

    #         print(f"Resuming training from step {start_step} with loss {loss}")
    #     else:
    #         start_step = 0
    #         loss = 0.0
    #         print("Starting training from scratch 1")

    # else:
    #     start_step = 0
    #     loss = 0.0
    #     print("Starting training from scratch 2")

    
    model.train()

    train(model, args)

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")
    
    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(args.out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)


    main(args)
