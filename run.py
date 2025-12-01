import sys
import subprocess
import os
import argparse
import torch
import os
os.environ['WANDB_API_KEY'] = '28acffdccf6e95ce7e4f346d3fec96439080b1dd'
# These are different commands to launch training for diffusion models.
def diffusion_training(gpus, model):
    print("Current model", model)
    pretrain = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port=1235 \
               main.py --mode train \
               --config-yml /root/autodl-tmp/AdsorbDiff/configs/flow/painn_conditional_flow.yml \
               --distributed --amp --identifier pretrainis2rstrain200_fewshot_std0.1-10_so30.01-1.55_lr1e-4 \
               --optim.lr_initial=1.e-4"
    equiformerv2 = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port=1235 \
               main.py --mode train \
               --config-yml configs/denoising/eqv2_conditional.yml \
               --distributed --amp --identifier FTis2rstrain200_cond_std0.1-10_so30.01-1.55_wf4 \
               --optim.lr_initial=1.e-4"

    painn = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port=1234 \
               main.py --mode train \
               --config-yml configs/flow/painn_conditional_flow.yml \
               --distributed  --identifier pretrainis2rs_sde_std0.1-10_so30.01-1.55_painn_new"
    gemnet_oc = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port 1234 \
               main.py --mode train \
               --config-yml configs/denoising/gemnet_so3.yml \
               --distributed  --identifier pretrainis2rs_sde_std0.1-10_so30.01-1.55_gemnet"
    return eval(model)


def sampling_and_relaxation(ngpus=1, nsite=1):
    out_path = f"/root/autodl-tmp/AdsorbDiff"
    #ckpt_path = "/home/jovyan/repos/ocp-modeling/checkpoints/2024-01-08-13-05-04-pretrainis2rs_sde_std0.1-10_so30.01-1.55_painn/checkpoint.pt"
    ckpt_path = "/root/autodl-tmp/AdsorbDiff/checkpoints/2025-11-18-19-20-32-debug-head/epoch0004_valloss4.1006.pt"
    relax_ckpt_path = (
        "/root/autodl-tmp/AdsorbDiff/gemnet_oc_base_s2ef_2M.pt"
    )
    val_id = "/root/autodl-tmp/AdsorbDiff/val_nonrelaxed_update"
    val_ood = "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/lmdbs/valood50_R1I0.1"
    final_cmd = ""
    for step in range(nsite):
        if step != 0:
            final_cmd += " && \n"
        step_path = f"{out_path}/{step}"
        com_sde = f"python -u -m torch.distributed.launch \
                --nproc_per_node={ngpus} --master_port 1234 \
                main.py --mode run-relaxations \
                --config-yml configs/flow/painn_conditional_flow.yml \
                --task.relax_dataset.src={val_id} \
                --task.relax_opt.traj_dir={step_path} \
                --checkpoint {ckpt_path} \
                --distributed   --model.sampling=True --seed {step} --debug"
        lmdb = f"python scripts/create_lmdbs/pred_traj_to_lmdb.py \
                --data-path {step_path} \
                --out-path {step_path}/final_struct_lmdb \
                --num-workers 4"
        com = f"python -u -m torch.distributed.launch \
                --nproc_per_node={ngpus} --master_port 1235 \
                main.py --mode run-relaxations \
                --config-yml configs/relaxation/gemnet_oc/gemnet_relax.yml \
                --checkpoint /root/autodl-tmp/AdsorbDiff/gemnet_oc_base_s2ef_2M.pt \
                --task.relax_dataset.src={step_path}/final_struct_lmdb \
                --task.relax_opt.traj_dir={step_path}/relaxations \
                --debug"
        cmd = com_sde + " && " + lmdb + " && " + com
        final_cmd += cmd
        print("final_cmd",final_cmd)
    return final_cmd


if __name__ == "__main__":

    # For training

    # command = diffusion_training(1, "painn")

    # To perform sampling and relaxation
    command = sampling_and_relaxation()

    with open("submit.sh", "w") as f:
        f.write(command)
    f.close()
    p = subprocess.Popen(["bash", "submit.sh"])
    p.wait()
