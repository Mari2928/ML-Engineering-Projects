import unittest
import os

class TestsForBlocks(unittest.TestCase):
    """
    Unittest class for convnet blocks.
    Ensure that layers can fprop some data without throwing errors.
    Each test runs an experiment with epoch=1.
    """
    
    # Test 1: test BatchNorm layers can fprop
    def test_bn_layers_fprop(self):   
        raised = False
        try:
            os.system('sh mlpractical/pytorch_mlp_framework/run_vgg_38_bn_unittest.sh')
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised')    

    # Test 2: test BatchNorm + ResidualConnection layers can fprop
    def test_bn_rc_layers_fprop(self):      
        raised = False
        try:
            os.system('sh mlpractical/pytorch_mlp_framework/run_vgg_38_bn_rc_unittest.sh')
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised')   
        
unittest.main(argv=['ignored', '-v'], exit=False)