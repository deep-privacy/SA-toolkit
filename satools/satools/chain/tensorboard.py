import os


from .. import script_utils
from .. import utils

class ChainTensorBoard():

    def __init__(otherself, self):
        from torch.utils.tensorboard import SummaryWriter

        if "valid" in self.chain_opts.egs:
            otherself.tensorboard = SummaryWriter(log_dir=self.chain_opts.dir + "/runs/" + "valid_"
                                                  + str(utils.creation_date_file(os.path.join(self.chain_opts.dir, "0.pt"))))
        else:
            otherself.tensorboard = SummaryWriter(log_dir=self.chain_opts.dir + "/runs/" + "worker_"
                                                  + self.chain_opts.new_model.split(".")[-2] + "--" + str(utils.creation_date_file(os.path.join(self.chain_opts.dir, "0.pt"))))
        otherself.global_step_save_path = otherself.tensorboard.log_dir + "/" + "global_step"
        otherself.global_step = 0
        if os.path.exists(otherself.global_step_save_path):
            otherself.global_step = script_utils.read_single_param_file(otherself.global_step_save_path)
        
        first_worker_step_save_path = self.chain_opts.dir + "/runs/" + "worker_1" + "--" + str(utils.creation_date_file(os.path.join(self.chain_opts.dir, "0.pt"))) + "/" + "global_step"
        if os.path.exists(first_worker_step_save_path):
            otherself.global_step = script_utils.read_single_param_file(first_worker_step_save_path)

    def add_scalar(otherself, main_tag, tag_scalar_dict, global_step=1):
        otherself.tensorboard.add_scalar(main_tag, tag_scalar_dict,  otherself.global_step + global_step)
        otherself.global_step_old = otherself.global_step + global_step

    def close(otherself):
        if otherself.global_step_old:
            script_utils.write_single_param_file(otherself.global_step_old, otherself.global_step_save_path)
        otherself.tensorboard.close()


