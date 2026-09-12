import unittest
import torch
from src.voxel_reconstruction import VoxelGuidedOptimizer, VoxelGuidedConfig

class GrowthSupportTests(unittest.TestCase):
    def setup_growth(self):
        p={n:torch.zeros((2,d) if d else (2,),requires_grad=True) for n,d in zip(VoxelGuidedOptimizer.PARAM_NAMES,(3,3,3,4,0))}
        opt=torch.optim.Adam([{'params':[t],'name':n} for n,t in p.items()])
        v=VoxelGuidedOptimizer(p['means'].detach(),VoxelGuidedConfig(max_gaussians=10),'cpu')
        v.grad_accum.fill_(1);v.grad_count.fill_(1)
        return p,opt,v

    def test_repeated_single_view_cannot_trigger_growth(self):
        p,opt,v=self.setup_growth()
        for _ in range(10):
            v.record_visible_views(torch.tensor([0,1]),torch.tensor([0,0]),torch.tensor([42]))
        self.assertEqual((v.seen_views>=0).sum().item(),2)
        self.assertIs(v._densify(opt,p),p)
        self.assertTrue((v.seen_views == -1).all())

    def test_distinct_views_allow_one_clone_per_target(self):
        p,opt,v=self.setup_growth()
        for camera in (10,20,30):
            v.record_visible_views(torch.tensor([0,1]),torch.tensor([0,0]),torch.tensor([camera]))
        result=v._densify(opt,p)
        self.assertEqual(len(result['means']),3)
        self.assertEqual(v.seen_views.shape,(3,3))
        self.assertTrue((v.seen_views == -1).all())
        self.assertTrue(all(t.is_leaf for t in result.values()))

    def test_dataset_ids_not_batch_slots(self):
        p,opt,v=self.setup_growth()
        v.record_visible_views(torch.tensor([0,0,1]),torch.tensor([0,1,1]),torch.tensor([7,9]))
        v.record_visible_views(torch.tensor([0]),torch.tensor([0]),torch.tensor([9]))
        self.assertEqual((v.seen_views[0]>=0).sum().item(),2)
        self.assertEqual((v.seen_views[1]>=0).sum().item(),1)
