import os
import pickle
import torch
import torch.distributed as dist

def gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return data

    tensor_list = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(tensor_list, data)
    return torch.stack(tensor_list, dim=0)

def verify_ddp_weights_equal(model: torch.nn.Module, atol: float = 1e-5) -> None:
    if hasattr(model, "module"):
        model = model.module

    world_size = get_world_size()
    correct = []
    if world_size == 1:
        return print("Skipping DDP verification as world_size=1")
    for name, param in model.named_parameters():
        gathered_param = gather(param.data).reshape((world_size, -1))
        absolute_diffs = (gathered_param[None, 0, :] - gathered_param).abs()
        rank_params_eq = (absolute_diffs < atol).all()
        correct.append(rank_params_eq.item())

    if not all(correct):
        print("DDP parameter verification failed ❌")
    else:
        print("Verified DDP parameter correctness ✅")


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    local_rank = get_local_rank()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(local_rank)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=torch.device("cuda", local_rank))
    size_list = [torch.tensor([0], device=torch.device("cuda", local_rank)) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8,
                            device=torch.device("cuda", local_rank)))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8,
                            device=torch.device("cuda", local_rank))
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def barrier():
    if dist.is_initialized():
        dist.barrier()
    else:
        pass

broadcast = dist.broadcast