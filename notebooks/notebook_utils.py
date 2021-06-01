# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import torch
import numpy as np
from os import makedirs
from PIL import Image
import time
import sys
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))
from ganspace_utils import prettify_name, pad_frames

# Apply edit to given latents, return strip of images
def create_strip(inst, mode, layer, latents, x_comp, z_comp, act_stdev, lat_stdev, sigma, layer_start, layer_end, num_frames=5):
    return _create_strip_impl(inst, mode, layer, latents, x_comp, z_comp, act_stdev,
                lat_stdev, None, None, sigma, layer_start, layer_end, num_frames, center=False)

# Strip where the sample is centered along the component before manipulation
def create_strip_centered(inst, mode, layer, latents, x_comp, z_comp, act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames=5, only_pos=False, only_neg=False):
    return _create_strip_impl(inst, mode, layer, latents, x_comp, z_comp, act_stdev,
                lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center=True, only_pos=only_pos, only_neg=only_neg)

def _create_strip_impl(inst, mode, layer, latents, x_comp, z_comp, act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center, only_pos=False, only_neg=False):
    if not isinstance(latents, list):
        latents = list(latents)

    max_lat = inst.model.get_max_latents()
    if layer_end < 0 or layer_end > max_lat:
        layer_end = max_lat
    layer_start = np.clip(layer_start, 0, layer_end)

    if len(latents) > num_frames:
        # Batch over latents
        return _create_strip_batch_lat(inst, mode, layer, latents, x_comp, z_comp,
            act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center, only_pos, only_neg)
    else:
        # Batch over strip frames
        return _create_strip_batch_sigma(inst, mode, layer, latents, x_comp, z_comp,
            act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center, only_pos, only_neg)

# Batch over frames if there are more frames in strip than latents
def _create_strip_batch_sigma(inst, mode, layer, latents, x_comp, z_comp, act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center, only_pos, only_neg): 
    inst.close()
    batch_frames = [[] for _ in range(len(latents))]
    
    B = min(num_frames, 5)
    lep_padded = ((num_frames - 1) // B + 1) * B
    if only_pos:
        sigma_range = np.linspace(0, sigma, num_frames)
    elif only_neg:
        sigma_range = np.linspace(0, -sigma, num_frames)
    else:
        sigma_range = np.linspace(-sigma, sigma, num_frames)
    
    sigma_range = np.concatenate([sigma_range, np.zeros((lep_padded - num_frames))])
    sigma_range = torch.from_numpy(sigma_range).float().to(inst.model.device)
    normalize = lambda v : v / torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True) + 1e-8)

    for i_batch in range(lep_padded // B):
        sigmas = sigma_range[i_batch*B:(i_batch+1)*B]
        
        for i_lat in range(len(latents)):
            z_single = latents[i_lat]
            z_batch = z_single.repeat_interleave(B, axis=0)
            
            zeroing_offset_act = 0
            zeroing_offset_lat = 0
            if center:
                if mode == 'activation':
                    # Center along activation before applying offset
                    inst.retain_layer(layer)
                    _ = inst.model.sample_np(z_single)
                    value = inst.retained_features()[layer].clone()
                    dotp = torch.sum((value - act_mean)*normalize(x_comp), dim=-1, keepdim=True)
                    zeroing_offset_act = normalize(x_comp)*dotp # offset that sets coordinate to zero
                else:
                    # Shift latent to lie on mean along given component
                    dotp = torch.sum((z_single - lat_mean)*normalize(z_comp), dim=-1, keepdim=True)
                    zeroing_offset_lat = dotp*normalize(z_comp)

            with torch.no_grad():
                z = z_batch

                if mode in ['latent', 'both']:
                    z = [z]*inst.model.get_max_latents()
                    delta = z_comp * sigmas.reshape([-1] + [1]*(z_comp.ndim - 1)) * lat_stdev
                    for i in range(layer_start, layer_end):
                        z[i] = z[i] - zeroing_offset_lat + delta

                if mode in ['activation', 'both']:
                    comp_batch = x_comp.repeat_interleave(B, axis=0)
                    delta = comp_batch * sigmas.reshape([-1] + [1]*(comp_batch.ndim - 1))
                    inst.edit_layer(layer, offset=delta * act_stdev - zeroing_offset_act)

                img_batch = inst.model.sample_np(z)
                if img_batch.ndim == 3:
                    img_batch = np.expand_dims(img_batch, axis=0)
                    
                for j, img in enumerate(img_batch):
                    idx = i_batch*B + j
                    if idx < num_frames:
                        batch_frames[i_lat].append(img)

    return batch_frames

# Batch over latents if there are more latents than frames in strip
def _create_strip_batch_lat(inst, mode, layer, latents, x_comp, z_comp, act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center, only_pos, only_neg):    
    n_lat = len(latents)
    B = min(n_lat, 5)

    max_lat = inst.model.get_max_latents()
    if layer_end < 0 or layer_end > max_lat:
        layer_end = max_lat
    layer_start = np.clip(layer_start, 0, layer_end)
    
    len_padded = ((n_lat - 1) // B + 1) * B
    batch_frames = [[] for _ in range(n_lat)]

    for i_batch in range(len_padded // B):
        zs = latents[i_batch*B:(i_batch+1)*B]
        if len(zs) == 0:
            continue
        
        z_batch_single = torch.cat(zs, 0)

        inst.close() # don't retain, remove edits
        sigma_range = np.linspace(-sigma, sigma, num_frames, dtype=np.float32)

        normalize = lambda v : v / torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True) + 1e-8)
        
        zeroing_offset_act = 0
        zeroing_offset_lat = 0
        if center:
            if mode == 'activation':
                # Center along activation before applying offset
                inst.retain_layer(layer)
                _ = inst.model.sample_np(z_batch_single)
                value = inst.retained_features()[layer].clone()
                dotp = torch.sum((value - act_mean)*normalize(x_comp), dim=-1, keepdim=True)
                zeroing_offset_act = normalize(x_comp)*dotp # offset that sets coordinate to zero
            else:
                # Shift latent to lie on mean along given component
                dotp = torch.sum((z_batch_single - lat_mean)*normalize(z_comp), dim=-1, keepdim=True)
                zeroing_offset_lat = dotp*normalize(z_comp)

        for i in range(len(sigma_range)):
            s = sigma_range[i]

            with torch.no_grad():
                z = [z_batch_single]*inst.model.get_max_latents() # one per layer

                if mode in ['latent', 'both']:
                    delta = z_comp*s*lat_stdev
                    for i in range(layer_start, layer_end):
                        z[i] = z[i] - zeroing_offset_lat + delta

                if mode in ['activation', 'both']:
                    act_delta = x_comp*s*act_stdev
                    inst.edit_layer(layer, offset=act_delta - zeroing_offset_act)

                img_batch = inst.model.sample_np(z)
                if img_batch.ndim == 3:
                    img_batch = np.expand_dims(img_batch, axis=0)
                    
                for j, img in enumerate(img_batch):
                    img_idx = i_batch*B + j
                    if img_idx < n_lat:
                        batch_frames[img_idx].append(img)

    return batch_frames


def save_frames(title, model_name, rootdir, frames, strip_width=10):
    test_name = prettify_name(title)
    outdir = f'{rootdir}/{model_name}/{test_name}'
    makedirs(outdir, exist_ok=True)
    
    # Limit maximum resolution
    max_H = 512
    real_H = frames[0][0].shape[0]
    ratio = min(1.0, max_H / real_H)
    
    # Combined with first 10
    strips = [np.hstack(frames) for frames in frames[:strip_width]]
    if len(strips) >= strip_width:
        left_col = np.vstack(strips[0:strip_width//2])
        right_col = np.vstack(strips[5:10])
        grid = np.hstack([left_col, np.ones_like(left_col[:, :30]), right_col])
        im = Image.fromarray((255*grid).astype(np.uint8))
        im = im.resize((int(ratio*im.size[0]), int(ratio*im.size[1])), Image.ANTIALIAS)
        im.save(f'{outdir}/{test_name}_all.png')
    else:
        print('Too few strips to create grid, creating just strips!')
    
    for ex_num, strip in enumerate(frames[:strip_width]):
        im = Image.fromarray(np.uint8(255*np.hstack(pad_frames(strip))))
        im = im.resize((int(ratio*im.size[0]), int(ratio*im.size[1])), Image.ANTIALIAS)
        im.save(f'{outdir}/{test_name}_{ex_num}.png')
        
####################################################################################################################################
#added
####################################################################################################################################
def LayerMode(layer_mode):
    if layer_mode == 'coarse':
        layer_start = 0
        layer_end = 3
    elif layer_mode == 'middle':
        layer_start = 3
        layer_end = 7
    elif layer_mode == 'fine':
        layer_start = 7
        layer_end = 18
    else:
        layer_start = 0
        layer_end = 18
    return layer_start, layer_end
def get_random_local_basis(model, random_state, noise = None):
    '''
    random_state = np.random.RandomState(seed)
    noise, z, z_local_basis, z_sv = get_random_local_basis(model, random_state)
    '''
    
    '''
    z = model.sample_latent(n_samples = 1, seed=seed)   in GanSpace
    : StyleGAN2.sample_latent(self, n_samples=1, seed=None, truncation=None)
    '''
    n_samples = 1
    if noise is not None:
        assert(list(noise.shape) == [n_samples, 512])
        noise = noise.detach().float().to(model.device)
    else:
        noise = torch.from_numpy(
                random_state.standard_normal(512 * n_samples)
                .reshape(n_samples, 512)).float().to(model.device) #[N, 512]
    noise.requires_grad = True
    if model.w_primary:
        z = model.model.style(noise)
    else:
        z = noise

    ''' Compute Jacobian by batch '''
    noise_dim, z_dim = noise.shape[1], z.shape[1]
    noise_pad = noise.repeat(z_dim, 1).requires_grad_(True)
    if model.w_primary:
        z_pad = model.model.style(noise_pad)
    else:
        z_pad = noise_pad

    grad_output = torch.eye(z_dim).cuda()
    jacobian = torch.autograd.grad(z_pad, noise_pad, grad_outputs=grad_output, retain_graph=True)[0].cpu()
    
    ''' Get local basis'''
    # jacobian \approx torch.mm(torch.mm(z_basis, torch.diag(s)), noise_basis.t())
    z_basis, s, noise_basis = torch.svd(jacobian)
    return noise, z.detach(), z_basis.detach(), s.detach(), noise_basis.detach()


def compare_basis_componentwise(reference_basis, compared_basis):
    '''
    Each basis should be normalized.
    reference_basis = (N, 1, basis_dim) or (N, basis_dim)
    compared_basis = (M, 1, basis_dim) or (M, basis_dim)
    sim_matrix = (N, M)
    '''
    if len(reference_basis) == 3:
        reference_basis_sq = reference_basis.squeeze()
    else:
        reference_basis_sq = reference_basis
        
    if len(compared_basis) == 3:
        compared_basis_sq = compared_basis.squeeze()
    else:
        compared_basis_sq = compared_basis
    
    sim_matrix, basis_orient = _similarity_by_inner_product(reference_basis_sq, compared_basis_sq)
    return sim_matrix, basis_orient

def _similarity_by_inner_product(reference_basis, compared_basis):
    signed_sim_matrix = torch.matmul(reference_basis, compared_basis.t()).detach().cpu()
    return torch.abs(signed_sim_matrix), torch.sign(signed_sim_matrix)

def get_topN_idx_sorted(ary, top_n = 1):
    if ary is not np.array:
        ary = np.array(ary)
    idx_not_sorted = np.argpartition(ary, -top_n)[-top_n:]
    idx_sorted = idx_not_sorted[np.argsort(ary[idx_not_sorted])[::-1]]
    return idx_sorted

def create_grid(inst, mode, layer, latents, x_comp, z_comps, act_stdev, lat_stdevs, sigma, layer_start, layer_end, num_frames=5):
    '''
    Create Image grid of 2-dimensional traversal
    '''
    if not isinstance(latents, list):
        latents = list(latents)
    assert(isinstance(z_comps, list) and isinstance(lat_stdevs, list))
    assert(len(latents) == 1 and len(z_comps) == 2 and len(lat_stdevs) == 2)
    
    max_lat = inst.model.get_max_latents()
    if layer_end < 0 or layer_end > max_lat:
        layer_end = max_lat
    layer_start = np.clip(layer_start, 0, layer_end)
    
    ver_z_comp, hor_z_comp = z_comps
    ver_lat_std, hor_lat_std = lat_stdevs
    
    sigma_range = np.linspace(-sigma, sigma, num_frames).reshape(-1, 1)
    sigma_range = torch.from_numpy(sigma_range).float().to(inst.model.device)
    z_batch = latents[0].repeat_interleave(num_frames, axis=0)
    verti_traversal = z_batch + ver_lat_std * sigma_range * ver_z_comp
    
    strips = []
    for i in range(num_frames):
        batch_frames = create_strip(inst, 'latent', 'style', [verti_traversal[i:i+1]], 0, hor_z_comp, 0, hor_lat_std, sigma, 0, 18, num_frames=5)[0]
        strips.append(np.hstack(pad_frames(batch_frames)))
#             for j, frame in enumerate(batch_frames):
#                 Image.fromarray(np.uint8(frame*255)).save(out_root / 'global' / f'{seed}_pc{i}_{j}.png')

    #col_left = np.vstack(pad_frames(strips[:n_pcs//2], 0, 64))
    #col_right = np.vstack(pad_frames(strips[n_pcs//2:], 0, 64))
    grid = np.vstack(strips)
    return grid
def create_strip_iter(inst, mode, layer, trvs_idx, sigma, layer_start, layer_end, random_state = None, noise = None, compare_basis = False, num_frames=5, num_steps = 100, only_pos=False, scale=False, verbose=False):
    if (random_state is None) and (noise is None):
        print('Either the random_state or noise must be given.')
        raise
    return _create_strip_iter_impl(inst, mode, layer, random_state, noise, trvs_idx, sigma,
                                   layer_start, layer_end, compare_basis, num_frames, num_steps, only_pos, scale, verbose)
 
def _create_strip_iter_impl(inst, mode, layer, random_state, noise, trvs_idx, sigma, layer_start, layer_end, compare_basis, num_frames, num_steps, only_pos, scale, verbose):
    max_lat = inst.model.get_max_latents()
    if layer_end < 0 or layer_end > max_lat:
        layer_end = max_lat
    layer_start = np.clip(layer_start, 0, layer_end)

    return _create_strip_batch_sigma_iter(inst, mode, layer, random_state, noise, trvs_idx, sigma,
                                          layer_start, layer_end, compare_basis, num_frames, num_steps, only_pos, scale, verbose)
            
# Batch over frames if there are more frames in strip than latents
def _create_strip_batch_sigma_iter(inst, mode, layer, random_state, noise, trvs_idx, sigma, layer_start, layer_end, compare_basis, num_frames, num_steps, only_pos, scale, verbose):    
    assert(num_frames % 2 == 1)
    inst.close()
    model = inst.model
    batch_frames = [[]]
    
    normalize = lambda v : v / torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True) + 1e-8)
    
    timer = time.time()
    z_init, z_trvs_batch = get_iter_trvs_latent(model, random_state, noise, trvs_idx, sigma, compare_basis, num_frames, num_steps, only_pos, scale, verbose)
    print(f'Getting iter trvs took {time.time() - timer}')
    if only_pos:
        z_init_batch = z_init.repeat_interleave(num_frames - num_frames // 2, axis=0)
    else:
        z_init_batch = z_init.repeat_interleave(num_frames, axis=0)
    if verbose:
        print(f'z difference from origin : {torch.norm(z_init_batch - z_trvs_batch, dim = 1)}')
    
    with torch.no_grad():
        if only_pos:
            z = z_init.repeat(num_frames - num_frames // 2, 1)
        else:
            z = z_init.repeat(num_frames, 1)

        if mode in ['latent', 'both']:
            z = [z]*inst.model.get_max_latents()
            for i in range(layer_start, layer_end):
                z[i] = z_trvs_batch

        if mode in ['activation', 'both']:
            print('Not implemented')
            raise
            
        img_batch = inst.model.sample_np(z)
        #assert(1==0)
    
        if img_batch.ndim == 3:
            img_batch = np.expand_dims(img_batch, axis=0)
        
        for img in (img_batch):
            batch_frames[0].append(img)

    return batch_frames, z_trvs_batch

def get_iter_trvs_latent(model, random_state, noise, trvs_idx, sigma, compare_basis, num_frames, num_steps, only_pos=False, scale=False, verbose=False):
    torch.autograd.set_grad_enabled(True)
    device = model.device
    
    each_dir_frames = (num_frames // 2)
    frame_steps = []
    for i in range(1,each_dir_frames+1):
        frame_steps.append(int(float(i) / each_dir_frames * (num_steps // 2))-1)
        
    ''' Initial latent '''
    noise, z, z_local_basis, z_sv, noise_basis = get_random_local_basis(model, random_state, noise)
    
    noise_iter_pos, noise_iter_neg = noise.clone().detach(), noise.clone().detach()
    
    z_iter_pos, z_iter_neg = z.clone().detach(), z.clone().detach()

    pos_trvs_z, neg_trvs_z = [], []
    # pos_trvs_z_orig, neg_trvs_z_orig = [], []
    trvs_idx_pos, trvs_idx_neg = trvs_idx, trvs_idx #k번째 factor
    z_sv_pos, z_sv_neg = z_sv, z_sv
    basis_orient_pos, basis_orient_neg = 1, 1
    pos_norm_sum, neg_norm_sum = 0, 0

    step_perturb_intensity = 2*sigma / num_steps
    for i in range(num_steps // 2):
        if i == 0:
            noise_basis_pos, noise_basis_neg = noise_basis, noise_basis
            z_local_basis_pos, z_local_basis_neg = z_local_basis, z_local_basis
            
        if compare_basis:
            z_ptb_pos = z_local_basis_pos[:, trvs_idx_pos].unsqueeze(0)
            z_ptb_neg = z_local_basis_neg[:, trvs_idx_neg].unsqueeze(0)
        trvs_idx_pos_pre = trvs_idx_pos
        trvs_idx_neg_pre = trvs_idx_neg
            
        # Approximate below by local basis
        # z_iter_pos \approx z_iter_pos + z_local_basis_prev[:, trvs_idx].unsqueeze(0) * z_sv_prev[trvs_idx] * sigma / num_steps
        # noise_iter_pos_prev = noise_iter_pos.clone().detach().cuda()
        noise_basis_pos_pre = noise_basis_pos.clone().detach().cuda()
        z_local_basis_pos_pre = z_local_basis_pos.clone().detach().cuda()
        
        if scale:
            noise_iter_pos = (noise_iter_pos.cpu() + noise_basis_pos[:, trvs_idx_pos].unsqueeze(0) * 2*(sigma/z_sv_pos[trvs_idx_pos]) / num_steps).clone().detach()
#             print(f"positive sv = {z_sv_pos[trvs_idx_pos]}")
        else:
            noise_iter_pos = (noise_iter_pos.cpu() + noise_basis_pos[:, trvs_idx_pos].unsqueeze(0) * 2*sigma / num_steps).clone().detach()
        
        z_iter_pos_original = (z_iter_pos.cpu() + z_local_basis_pos[:, trvs_idx_pos].unsqueeze(0) * 2*sigma / num_steps).cuda().clone().detach()
        noise_iter_pos, z_iter_pos, z_local_basis_pos, z_sv_pos, noise_basis_pos = get_random_local_basis(model, random_state, noise=noise_iter_pos)
        
        #print(f"Positive direction Noise ptb size : {torch.norm(noise_iter_pos_prev - noise_iter_pos)}")
        #print(noise_iter_pos[0, :10])
        noise_basis_neg_pre = noise_basis_neg.clone().detach().cuda()
        z_local_basis_neg_pre = z_local_basis_neg.clone().detach().cuda()
        if scale:
            noise_iter_neg = (noise_iter_neg.cpu() - noise_basis_neg[:, trvs_idx_neg].unsqueeze(0) * 2*(sigma/z_sv_neg[trvs_idx_neg]) / num_steps).clone().detach()
#             print(f"negative sv = {z_sv_neg[trvs_idx_neg]}")
        else:
            noise_iter_neg = (noise_iter_neg.cpu() - noise_basis_neg[:, trvs_idx_neg].unsqueeze(0) * 2*sigma / num_steps).clone().detach()

        z_iter_neg_original = (z_iter_neg.cpu() - z_local_basis_neg[:, trvs_idx_neg].unsqueeze(0) * 2*sigma / num_steps).cuda().clone().detach()
        noise_iter_neg, z_iter_neg, z_local_basis_neg, z_sv_neg, noise_basis_neg = get_random_local_basis(model, random_state, noise=noise_iter_neg.clone().detach())

        if compare_basis:
            sim_matrix_pos, basis_orient_pos = compare_basis_componentwise(z_ptb_pos, z_local_basis_pos.t())
            trvs_idx_pos = get_topN_idx_sorted(sim_matrix_pos[0], top_n = 1)[0]
            
            sim_matrix_neg, basis_orient_neg = compare_basis_componentwise(z_ptb_neg, z_local_basis_neg.t())
            trvs_idx_neg = get_topN_idx_sorted(sim_matrix_neg[0], top_n = 1)[0]

            #print(trvs_idx_pos, trvs_idx_neg)
        ######
        noise_basis_pos_cos = torch.dot(noise_basis_pos_pre[:, trvs_idx_pos_pre], noise_basis_pos[:, trvs_idx_pos].detach().cuda())
        noise_basis_neg_cos = torch.dot(noise_basis_neg_pre[:, trvs_idx_neg_pre], noise_basis_neg[:, trvs_idx_neg].detach().cuda())
        
        #print(f"Positive direction Noise difference from origin : {torch.norm(noise - noise_iter_pos)} ")
        if noise_basis_pos_cos < 0:
            noise_basis_pos = -noise_basis_pos
        if noise_basis_neg_cos < 0:
            noise_basis_neg = -noise_basis_neg
        pos_norm_sum += torch.norm(z_iter_pos - z_iter_pos_original, dim=1)
        neg_norm_sum += torch.norm(z_iter_neg - z_iter_neg_original, dim=1)
        ########
        if i in frame_steps:
            pos_trvs_z.append(z_iter_pos.clone().detach())
            neg_trvs_z.append(z_iter_neg.clone().detach())
            # pos_trvs_z_orig.append(z_iter_pos_original.cuda().clone().detach())
            # neg_trvs_z_orig.append(z_iter_neg_original.cuda().clone().detach())
            
        if verbose:
            z_basis_pos_cos = torch.dot(z_local_basis_pos_pre[:, trvs_idx_pos_pre], z_local_basis_pos[:, trvs_idx_pos].detach().cuda())
            z_iter_pos_cos_diff = torch.dot(z_iter_pos_original.squeeze().cuda() / z_iter_pos_original.squeeze().cuda().norm(), z_iter_pos.squeeze() / z_iter_pos.squeeze().norm())            
            print(z_iter_pos_original.shape, z_iter_pos.shape)
            z_iter_pos_norm_diff = torch.norm(z_iter_pos_original.squeeze().cuda() - z_iter_pos.squeeze())
            z_basis_neg_cos = torch.dot(z_local_basis_neg_pre[:, trvs_idx_neg_pre], z_local_basis_neg[:, trvs_idx_neg].detach().cuda())
            z_iter_neg_cos_diff = torch.dot(z_iter_neg_original.squeeze().cuda() / z_iter_neg_original.squeeze().norm(), z_iter_neg.squeeze() / z_iter_neg.squeeze().norm())
            z_iter_neg_norm_diff = torch.norm(z_iter_neg_original.squeeze().cuda() - z_iter_neg.squeeze())
            print(f"Perturbation intensity per step = {step_perturb_intensity}")
            if compare_basis:
                print("Comparing basis mode")
                print(f"Traveling with positive pre: {trvs_idx_pos_pre+1}th, cur: {trvs_idx_pos+1}th vector result ( step {i+1} )")
                print(f"Traveling with negative pre: {trvs_idx_neg_pre+1}th, cur: {trvs_idx_neg+1}th vector result ( step {i+1} )")
                z_pos_cos_ranking = torch.sort(sim_matrix_pos,descending=True)[0]
                z_neg_cos_ranking = torch.sort(sim_matrix_neg,descending=True)[0]
            else:
                print(f"Traveling with {trvs_idx_pos+1}th vector ( step {i+1} )")
            print(f"Positive noise basis cosine similarity = {noise_basis_pos_cos:.3f}")
            print(f"Negative noise basis cosine similarity = {noise_basis_neg_cos:.3f}")
            print(f"Positive z basis cosine similarity = {z_basis_pos_cos:.3f}")
            print(f"Negative z basis cosine similarity = {z_basis_neg_cos:.3f}")
            print(f"Positive direction z approx cosine similiratiry = {z_iter_pos_cos_diff:.3f}")
            print(f"Negative direction z approx cosine similiratiry = {z_iter_neg_cos_diff:.3f}")
            print(f"Positive direction z approx norm = {z_iter_pos_norm_diff:.3f}")
            print(f"Negative direction z approx norm = {z_iter_neg_norm_diff:.3f}")
            if compare_basis:
                print(f"Positive z basis top5 consine similarity = {z_pos_cos_ranking[0,:5]}")
                print(f"Negative z basis top5 consine similarity = {z_neg_cos_ranking[0,:5]}")
            print()
    if verbose:
        print(f"Positive z norm sum : {pos_norm_sum}")
        print(f"Negative z norm sum : {neg_norm_sum}")        
        print("###############################################################################################################")
        
    if only_pos:
        z_batch = torch.cat([z] + pos_trvs_z, dim=0)
    else:
        z_batch = torch.cat(neg_trvs_z[::-1] + [z] + pos_trvs_z, dim=0)
    return z, z_batch