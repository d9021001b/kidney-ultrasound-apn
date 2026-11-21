import unittest
import numpy as np
import torch

from nckuh_dmsa_metric_optimization_20261102 import warm_start,new_network,optimize


class MetricObjectiveTests(unittest.TestCase):
    def test_preprocessing_fits_only_selected_rows(self):
        x=np.c_[np.arange(8),[0,0,0,0,0,0,999,999]].astype(np.float32)
        y=np.array([0,1,0,1,0,1,0,1])
        z,transform,lr=warm_start(x,y,np.arange(6),1)
        self.assertEqual(z.shape[1],1)
        np.testing.assert_allclose(transform.named_steps['simpleimputer'].statistics_,[2.5,0])
        with torch.inference_mode():
            score=torch.sigmoid(new_network(lr)(z)).numpy().ravel()
        np.testing.assert_allclose(score,lr.predict_proba(transform.transform(x))[:,1],atol=1e-6)

    def test_all_native_optimizers_update_finite_parameters(self):
        x=np.random.default_rng(1).normal(size=(12,4)).astype(np.float32)
        target=np.array([0,1]*6); train=np.arange(8)
        z,transform,lr=warm_start(x,target,train,1)
        y=torch.tensor(target,dtype=torch.float32)
        for mode in ['BCE_SGD','BCE_Adam','AUCM_PESG','AP_SOAP']:
            model,epoch,history,audit=optimize(z,y,train,None,lr,mode,epochs=2)
            self.assertEqual(epoch,2); self.assertEqual(len(history),2)
            self.assertGreater(audit['parameter_squared_displacement'],0)
            self.assertFalse(audit['AP_heldout_indices_used'])
            if mode=='AP_SOAP':
                self.assertEqual(audit['AP_memory_length'],8)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__=='__main__':
    unittest.main()
