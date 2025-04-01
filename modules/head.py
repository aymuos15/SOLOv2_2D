import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.nms import multi_apply, matrix_nms
from .utils.losses import dice_loss, py_sigmoid_focal_loss
from .branches.category import CategoryBranch
from .branches.kernel import KernelBranch
from .branches.feature import FeatureBranch

from scipy import ndimage

INF = 1e8

class SOLOv2Head(nn.Module):
    def __init__(self, num_classes, in_channels, seg_feat_channels=256, stacked_convs=4,
                 strides=(4, 8, 16, 32, 64), base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2, num_grids=None, ins_out_channels=64, loss_ins=None,
                 loss_cate=None, conv_cfg=None, norm_cfg=None,
                 mask_feat_start_level=0, mask_feat_end_level=3):
        super().__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.ins_loss_weight = 3.0
        self.norm_cfg = norm_cfg
        
        self.category_branch = CategoryBranch(
            in_channels=self.in_channels,
            seg_feat_channels=self.seg_feat_channels,
            stacked_convs=self.stacked_convs,
            cate_out_channels=self.cate_out_channels,
            norm_cfg=self.norm_cfg
        )
        
        self.kernel_branch = KernelBranch(
            in_channels=self.in_channels,
            seg_feat_channels=self.seg_feat_channels,
            stacked_convs=self.stacked_convs,
            kernel_out_channels=self.kernel_out_channels,
            norm_cfg=self.norm_cfg
        )
        
        self.mask_feat_start_level = mask_feat_start_level
        self.mask_feat_end_level = mask_feat_end_level
        self.feature_branch = FeatureBranch(
            in_channels=in_channels,
            out_channels=ins_out_channels,
            start_level=mask_feat_start_level,
            end_level=mask_feat_end_level,
            num_classes=ins_out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )

    def init_weights(self):
        self.category_branch.init_weights()
        self.kernel_branch.init_weights()
        self.feature_branch.init_weights()

    def forward(self, feats, eval=False):
        mask_feat_pred = self.forward_mask_feat(
            feats[self.mask_feat_start_level:self.mask_feat_end_level + 1])
        
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        cate_pred, kernel_pred = multi_apply(self.forward_single, new_feats,
                                           list(range(len(self.seg_num_grids))),
                                           eval=eval, upsampled_size=upsampled_size)
        
        return cate_pred, kernel_pred, mask_feat_pred
    
    def forward_mask_feat(self, inputs):
        return self.feature_branch(inputs)
    
    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear', align_corners=False))

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        batch_size = x.shape[0]
        x_range = torch.linspace(-1, 1, x.shape[-1], device=x.device)
        y_range = torch.linspace(-1, 1, x.shape[-2], device=x.device)
        y, x_grid = torch.meshgrid(y_range, x_range)
        coord_feat = torch.cat([
            x_grid.expand([batch_size, 1, -1, -1]), 
            y.expand([batch_size, 1, -1, -1])
        ], 1)
        
        ins_kernel_feat = torch.cat([x, coord_feat], 1)
        
        kernel_pred = self.kernel_branch(ins_kernel_feat, grid_size=self.seg_num_grids[idx])
        
        cate_pred = self.category_branch(x, grid_size=self.seg_num_grids[idx], eval=eval)
        
        return cate_pred, kernel_pred

    def apply_kernels_to_features(self, kernel_preds, mask_features):
        if isinstance(kernel_preds, list) and isinstance(kernel_preds[0], list):
            mask_pred_list = []
            for b_kernel_pred in kernel_preds:
                b_mask_pred = []
                for idx, kernel_pred in enumerate(b_kernel_pred):
                    if kernel_pred.size()[-1] == 0:
                        continue
                    cur_mask_feat = mask_features[idx, ...]
                    H, W = cur_mask_feat.shape[-2:]
                    N, I = kernel_pred.shape
                    cur_mask_feat = cur_mask_feat.unsqueeze(0)
                    kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                    
                    cur_mask_pred = F.conv2d(cur_mask_feat, kernel_pred, stride=1).view(-1, H, W)
                    b_mask_pred.append(cur_mask_pred)
                if len(b_mask_pred) == 0:
                    b_mask_pred = None
                else:
                    b_mask_pred = torch.cat(b_mask_pred, 0)
                mask_pred_list.append(b_mask_pred)
            return mask_pred_list
        else:
            I, N = kernel_preds.shape
            kernel_preds = kernel_preds.view(I, N, 1, 1)
            
            mask_preds = F.conv2d(mask_features, kernel_preds, stride=1).squeeze(0).sigmoid()
            return mask_preds

    def loss(self,
             cate_preds,
             kernel_preds,
             mask_feat_pred,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             img_metas,
             cfg = None,
             gt_bboxes_ignore=None):
        mask_feat_size = mask_feat_pred.size()[-2:]
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.solov2_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list, 
            mask_feat_size=mask_feat_size)

        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]
        
        ins_pred_list = self.apply_kernels_to_features(kernel_preds, mask_feat_pred)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        loss_ins = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        loss_cate = py_sigmoid_focal_loss(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return dict(loss_ins=loss_ins, loss_cate=loss_cate)

    def solov2_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               mask_feat_size):

        device = gt_labels_raw[0].device

        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue

            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = 4
            for gt_label, half_h, half_w in zip(gt_labels, half_hs, half_ws):
                
                seg_mask = gt_masks_raw[0, :, :]

                if seg_mask.sum() == 0:
                   continue

                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                top = max(0, coord_h - 1, int((center_h - half_h) * num_grid / upsampled_size[0]))
                down = min(num_grid - 1, coord_h + 1, int((center_h + half_h) * num_grid / upsampled_size[0]))
                left = max(0, coord_w - 1, int((center_w - half_w) * num_grid / upsampled_size[1]))
                right = min(num_grid - 1, coord_w + 1, int((center_w + half_w) * num_grid / upsampled_size[1]))

                cate_label[top:(down+1), left:(right+1)] = gt_label
                
                if not isinstance(seg_mask, torch.Tensor):
                    seg_mask = torch.tensor(seg_mask, device=device)
                seg_mask = seg_mask.float().unsqueeze(0).unsqueeze(0)
                seg_mask = F.interpolate(seg_mask, 
                                         scale_factor=1./output_stride,
                                         mode='nearest')
                seg_mask = seg_mask.squeeze(0).squeeze(0)
                
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)

            ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def get_seg(self, cate_preds, kernel_preds, mask_feat_pred, img_metas, cfg, rescale=None):
        num_levels = len(cate_preds)
        featmap_size = mask_feat_pred.size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]
            seg_pred_list = mask_feat_pred[img_id, ...].unsqueeze(0)
            kernel_pred_list = [
                kernel_preds[i][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach()
                                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list, kernel_pred_list,
                                         featmap_size, img_shape, ori_shape, cfg, rescale)
            result_list.append(result)
        return result_list

    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       kernel_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       cfg,
                       rescale=False, debug=False):
        assert len(cate_preds) == len(kernel_preds)

        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        inds = (cate_preds > cfg['score_thr'])
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None

        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        seg_preds = self.apply_kernels_to_features(kernel_preds, seg_preds)
        
        seg_masks = seg_preds > cfg['mask_thr']
        sum_masks = seg_masks.sum((1, 2)).float()

        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg['nms_pre']:
            sort_inds = sort_inds[:cfg['nms_pre']]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                    kernel=cfg['kernel'],sigma=cfg['sigma'], sum_masks=sum_masks)

        keep = cate_scores >= cfg['update_thr']
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg['max_per_img']:
            sort_inds = sort_inds[:cfg['max_per_img']]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                    size=upsampled_size_out,
                                    mode='bilinear', align_corners=False)[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                               size=ori_shape[:2],
                               mode='bilinear',
                               align_corners=False).squeeze(0)
        seg_masks = seg_masks > cfg['mask_thr']
        return seg_masks, cate_labels, cate_scores
