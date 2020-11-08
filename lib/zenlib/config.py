import yaml
import os


class Config:

    def __init__(self, cfg_id):
        self.id = cfg_id
        cfg_name = 'kinematic_synthesis/cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        base_dir = '~/results/diverse_forecast'
        self.base_dir = os.path.expanduser(base_dir)
        if not os.path.exists(os.path.join(self.base_dir, 'remote.txt')):
            print("Didn't mount correct directory!")
            exit(0)

        self.cfg_dir = '%s/human_new/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.meta_id = cfg['meta_id']
        self.horizon = cfg['horizon']
        self.history = cfg['history']
        self.fr_num = self.horizon + self.history
        self.nz = cfg['nz']
        self.beta = cfg['beta']
        self.bi_rnn = cfg.get('bi_rnn', False)
        self.q_scale = cfg.get('q_scale', 100)
        self.D_scale = cfg.get('D_scale', 1e-2)
        self.z_percent = cfg.get('z_percent', 0.9)
        self.map_inf = cfg.get('map_inf', True)
        self.num_sample = cfg.get('num_sample', 10)
        self.model_lr = cfg['model_lr']
        self.sampler_lr = cfg.get('sampler_lr', 1e-4)
        self.num_vae_data_sample = cfg.get('num_vae_data_sample', 1e4)
        self.num_dpp_data_sample = cfg.get('num_dpp_data_sample', 1e4)
        self.num_r2p2_data_sample = cfg.get('num_r2p2_data_sample', 1e4)
        self.num_vib_data_sample = cfg.get('num_vib_data_sample', 1e4)
        self.num_epoch = cfg['num_epoch']
        self.num_dpp_epoch = cfg.get('num_dpp_epoch', 20)
        self.save_model_interval = cfg['save_model_interval']
        self.r2p2_beta = cfg.get('r2p2_beta', 1.0)
        self.r2p2_lr = cfg.get('r2p2_lr', 1e-3)
        self.r2p2_noise = cfg.get('r2p2_noise', 1e-3)
        self.generator_lr = cfg.get('generator_lr', 1e-4)
        self.discrim_lr = cfg.get('discrim_lr', 1e-4)
        self.gan_l2 = cfg.get('gan_l2', 0.0)

        self.vae_model_path = self.model_dir + '/iter_%04d.p'
        self.r2p2_model_path = self.model_dir + '/r2p2_iter_%04d.p'
        self.gan_model_path = self.model_dir + '/gan_iter_%04d.p'
        self.dpp_model_path = '%s/sp.p' % self.model_dir
        self.mcl_model_path = '%s/mcl.p' % self.model_dir
        self.vib_model_path = self.model_dir + '/vib_iter_%04d.p'

        self.vae_cfg = cfg.get('vae_cfg', None)
        self.dpp_cfg = cfg.get('dpp_cfg', None)
        if self.vae_cfg is not None:
            self.vae_model_path = '%s/human_new/%s/models/' % (self.base_dir, self.vae_cfg) + 'iter_%04d.p'

        self.model_cfg = cfg.get('model_cfg', None)
        if self.model_cfg is not None:
            self.vae_model_path = '%s/human_new/%s/models/' % (self.base_dir, self.model_cfg) + 'iter_%04d.p'
            self.gan_model_path = '%s/human_new/%s/models/' % (self.base_dir, self.model_cfg) + 'gan_iter_%04d.p'
            self.r2p2_model_path = '%s/human_new/%s/models/' % (self.base_dir, self.model_cfg) + 'r2p2_iter_%04d.p'


if __name__ == '__main__':
    cfg = Config('0503-1')
    pass
