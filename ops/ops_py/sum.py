# ops/ops_py/sum.py

import torch
from torch.autograd import Function
import sum_double

class SumDouble(Function):
    @staticmethod
    def forward(ctx, array1, array2):
        '''sum_double function forward.
        Arg:
            array1 (torch.Tensor): [n,]
            array2 (torch.Tensor): [n,]

        
        Returns:
            ans (torch.Tensor): [n,]

        '''

        array1 = array1.float()
        array2 = array2.float()
        ans = array1.new_zeros(array1.shape)
        sum_double.forward(array1.contiguous(), array2.contiguous(), ans)

        return ans
    
    @staticmethod
    def backward(ctx, g_out):
        g_in1 = g_out.clone()
        g_in2 = g_out.clone()
        return g_in1, g_in2
    

sum_double_op = SumDouble.apply