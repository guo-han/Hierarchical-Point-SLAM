import os

import torch


class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, cfg, args, mapper
                 ):
        self.verbose = mapper.verbose
        self.ckptsdir = mapper.ckptsdir
        self.gt_c2w_list = mapper.gt_c2w_list
        self.estimate_c2w_list = mapper.estimate_c2w_list
        self.decoders = mapper.decoders
        self.radius_hierarchy = cfg['pointcloud']['radius_hierarchy']

    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes, npc):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        save_dict = {}
        for level in self.radius_hierarchy.keys():
            save_dict['geo_feats' + level] = npc.get_geo_feats(level = level)
            save_dict['col_feats' + level] = npc.get_col_feats(level = level)
            save_dict['cloud_pos' + level] = npc.cloud_pos(level)
            save_dict['cloud_normal' + level] = npc.cloud_normal(level = level)
            
        save_dict['pts_num'] = npc.pts_num() # no level input
        save_dict['input_pos'] = npc.input_pos()
        save_dict['input_rgb'] = npc.input_rgb()
        save_dict['input_normal'] = npc.input_normal()
        save_dict['input_normal_cartesian'] = npc.input_normal_cartesian()
        
        save_dict['decoder_state_dict'] = self.decoders.state_dict()
        save_dict['gt_c2w_list'] = self.gt_c2w_list
        save_dict['estimate_c2w_list'] = self.estimate_c2w_list
        save_dict['keyframe_list'] = keyframe_list
        save_dict['keyframe_dict'] = keyframe_dict
        save_dict['selected_keyframes'] = selected_keyframes
        save_dict['idx'] = idx

        # torch.save({
        #     'geo_feats':npc.get_geo_feats(),  
        #     'col_feats':npc.get_col_feats(),  
        #     'cloud_pos':npc.cloud_pos(),      
        #     'pts_num': npc.pts_num(),         
        #     'input_pos': npc.input_pos(),     
        #     'input_rgb': npc.input_rgb(),     
        #     'input_normal': npc.input_normal(),  
        #     'input_normal_cartesian': npc.input_normal_cartesian(),
        #     'cloud_normal': npc.cloud_normal(),

        #     'decoder_state_dict': self.decoders.state_dict(),
        #     'gt_c2w_list': self.gt_c2w_list,             
        #     'estimate_c2w_list': self.estimate_c2w_list, 
        #     'keyframe_list': keyframe_list,
        #     'keyframe_dict': keyframe_dict,
        #     'selected_keyframes': selected_keyframes,
        #     'idx': idx,
        # }, path, _use_new_zipfile_serialization=False)

        torch.save(save_dict, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)