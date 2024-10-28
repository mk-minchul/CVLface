import os
import torch
import omegaconf
import shutil


class BasePipeline():

    def __init__(self):
        self.start_epoch = 0
        self.last_save_name = None
        self.color_space = None

    @property
    def module_names_list(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def integrity_check(self, dataset_color_space):
        raise NotImplementedError

    def resume_from_dir(self, resume_dir):
        pipepline_path = os.path.join(resume_dir, 'pipeline.pt')
        pipeline_st = torch.load(pipepline_path, map_location='cpu')
        self.start_epoch = pipeline_st['epoch']
        self.last_save_name = pipeline_st['last_save_name']
        self.color_space = pipeline_st['color_space']
        assert self.module_names_list == pipeline_st['module_names_list']

        for name in self.module_names_list:
            mod = getattr(self, name)
            if mod is None:
                continue

            if hasattr(mod, 'load_state_dict_from_path'):
                mod.load_state_dict_from_path(pipepline_path.replace('pipeline.pt', f'{name}.pt'))
            else:
                mod_st = torch.load(pipepline_path.replace('pipeline.pt', f'{name}.pt'), map_location='cpu')
                mod.load_state_dict(mod_st)

        epoch = pipeline_st['epoch']
        step = pipeline_st['step']
        n_images_seen = pipeline_st['n_images_seen']
        return epoch, step, n_images_seen

    def save(self, fabric, pipeline, cfg, epoch, step, n_images_seen, is_best=False):

        # save model (it could happen in more than rank 0)
        save_dir = os.path.join(cfg.trainers.output_dir, 'checkpoints', f'epoch:{epoch}_step:{step}')
        self.save_pipelines_and_configs(save_dir, fabric, pipeline, cfg, epoch, step, n_images_seen)

        fabric.barrier()

        if fabric.local_rank == 0:
            if is_best:
                best_save_dir = os.path.join(cfg.trainers.output_dir, 'checkpoints', f'best')
                if os.path.exists(best_save_dir):
                    shutil.rmtree(best_save_dir)
                shutil.copytree(save_dir, best_save_dir, dirs_exist_ok=True)

            # remove old checkpoints
            if self.last_save_name is not None:
                if os.path.exists(self.last_save_name):
                    os.system(f'rm -rf {self.last_save_name}')
            self.last_save_name = save_dir

        fabric.barrier()


    @staticmethod
    def save_pipelines_and_configs(save_dir, fabric, pipeline, cfg, epoch, step, n_images_seen):
        os.makedirs(save_dir, exist_ok=True)
        for name in pipeline.module_names_list:
            mod = getattr(pipeline, name)
            if hasattr(mod, 'save_pretrained'):
                # model, classifier, aligner, etc
                mod.save_pretrained(save_dir=save_dir, name=f'{name}.pt', rank=fabric.local_rank)
            elif mod is None:
                pass
            else:
                # optimizer and lr_scheduler
                if fabric.local_rank == 0:
                    torch.save(mod.state_dict(), os.path.join(save_dir, f'{name}.pt'))

        if fabric.local_rank == 0:
            # save omega config to yaml
            omegaconf.OmegaConf.save(cfg, os.path.join(save_dir, 'config.yaml'))
            torch.save({'epoch': epoch, 'step': step, 'n_images_seen': n_images_seen,
                        'cfg': cfg, 'module_names_list': pipeline.module_names_list,
                        'last_save_name': pipeline.last_save_name,
                        'color_space': pipeline.color_space, },
                       os.path.join(save_dir, f'pipeline.pt'))
