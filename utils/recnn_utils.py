import wandb

def write_losses(writer, loss_dict, kind="train"):
    step = loss_dict["step"]
    
    for k, v in loss_dict.items():
        if k == "step":
            continue
        # Логируем каждый параметр через wandb
        writer.log({f"{kind}/{k}": v}, step=step)

    # Не нужно закрывать writer, так как в wandb это делается автоматически
