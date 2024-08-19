# Copyright (c) 2023 Val Krigan, no rights reserved. 
# I.e. you can copy/paste SoftMoE in your project, modify, remove comments, including this one

import torch
import torch.nn as nn
from torch.nn import functional as F

# Notes:
# Soft Mode: In this mode, the outputs from all experts are combined in a weighted sum, where the weights are determined by a routing mechanism (a trainable linear layer).
# Sparse Mode: In this mode, only a subset of the experts (usually 2) is selected based on the highest weights.  This mode is more efficient as it reduces the number of experts that need to be 
# called during the forward pass, leading to faster inference and potentially lower costs
class SoftMoE(nn.Module):
    """ soft mixture of experts 
        the idea:  have a trainable routing table in front of N*FF experts.
            table multiplied by token produces weights for each of N experts.
            output is a weighted sum (softmaxed) of N experts output
        sparse mixture of experts is trained soft MoE with fixed routing table,
            then only K (=2 usually) outputs a selected, with highests weights.
            as weights are known before experts call we can call only K selected experts
    """
    
    # making state shared by all objects, it's hard to control individual within mingpt architecture
    # Global state management: use global global state ('SoftMoE.nums' and 'SoftMoE.sparse') to manage
    # the configuration across multiple instances of the 'SoftMoE' class. This approach simplifies the 
    # control of expert selection and routing behavior across the entire model.
    moe_nums = None  # (total, active) number of experts
    sparse = None    # None for soft moe mode, means router and all experts are trained
                     # active number of experts for sparse mode

    def set_sparse(on=True):
        print(f"SoftMoE setting sparse mode to '{on}', MoE params: {SoftMoE.moe_nums}") 
        if on:
            SoftMoE.sparse = SoftMoE.moe_nums[1]
        else:
            SoftMoE.sparse = None

# Expert class defines a simplified feed-forward network which represents a single expert in the MoE. Consists of two linear
# layers with a GELU activation function in between. This design allows each expert to process the input independently and contribute
# to the final output based on the routing weights

    class Expert(nn.Module):
        """ simplified FF network, note: with no dropout"""
        def __init__(self, expert_size, n_embd, forward_width):
            super().__init__()
            self.n_embd = n_embd
            self.forward_in = expert_size  # all experts work in parallel, have the same input
            self.forward_width = forward_width 
            self.net = nn.Sequential(
                nn.Linear(self.forward_in, forward_width),
                nn.GELU(), 
                nn.Linear(forward_width, n_embd),
            )

        def forward(self, x):
            return self.net(x)
        
    """ params:
            moe_nums  - (total, active)  numbers of experts
            expert_size - whatever comes out of attention layer, usually ==m_embd
            m_embd      - emedding width
            forward_width  - internal intermediate data width, controls expert's size, usually 4*m_embd
    #"""
    def __init__(self, moe_nums, expert_size, n_embd, forward_width, dropout, sparse =False):
        super().__init__()
        self.n_embd = n_embd
        self.dropout_ratio = dropout
        self.forward_in = expert_size  # all experts have the same input
        self.forward_width = forward_width if forward_width!=None else 4*n_embd
        #!!self.forward_width = forward_width if forward_width!=None else n_embd//2
        #self.moe_nums = moe_nums
        SoftMoE.moe_nums = moe_nums   # globalizing it
        
        self.num_experts = moe_nums[0]
        # The router is a trainable linear layer that maps the input (token) to a set of weights, one for each expert,
        # signifying how important that weight is for the input token.
        # These weights determine how much influence each expert should have on the final output.
        self.router = nn.Linear(self.forward_in, self.num_experts)   # from token produces weights for each expert
        self.experts = nn.ModuleList([SoftMoE.Expert(expert_size, n_embd, self.forward_width) for _ in range(self.num_experts)])
        self.dropout = nn.Dropout(dropout)  # dropout at the end

    def forward(self, x):
        import pdb
        pdb.set_trace()
        sparse = SoftMoE.sparse  # global state for all objects
        # there is no separate call for it, changing here
        # When sparse mode is enabled, the router is set to evaluation mode, and its gradients are frozen,
        # meaning it is no longer trainable. The forward pass in sparse mode involves selecting the top-k experts based
        # on the routing weights, computing the outputs for these experts, and summing them, weighted by the normalized 
        # top-k weights.
        if sparse:
            self.router.eval()
            self.router.requires_grad_(False)
        else:
            self.router.train()
            self.router.requires_grad_(True)
        
        weights = self.router(x)
        # In soft mode, the weights are softmaxed and used to compute a weighted sum of the outputs of all experts.
        # In sparse mode, the top-k experts (determined byt eh highest weights) are selected, and only these 
        # experts' outputs are considered.
        if sparse == None or sparse==self.num_experts:
            # can be done more efficiently if we represent all experts as one matrix
            weights = F.softmax(weights, dim=-1) 
            out = torch.stack([e(x) for e in self.experts], dim=-1)
            out = torch.sum(out * weights.unsqueeze(-2), dim=-1)
        else:
            # x: Tensor of shape [batch_size, n, d_model]
            # weights: Tensor of shape [batch_size, n, k],  k=self.num_experts
            # m: number of top experts to select (integer), m=self.sparse
            batch_size, n, _ = x.shape
            
            # the number of selected experts
            m = min(self.num_experts, sparse)

            # Select the top m weights and their indices
            top_m_values, top_m_indices = torch.topk(weights, m, dim=-1)
            
            # Apply softmax to the selected weights
            normalized_weights = F.softmax(top_m_values, dim=-1)
            

            # Initialize the result tensor
            out = torch.zeros_like(x)

            # Process each expert's inputs
            for j in range(m):
                for expert_idx in range(len(self.experts)):
                    # Create a mask for the current expert
                    mask = (top_m_indices[:, :, j] == expert_idx)

                    # Check if the current expert is used
                    if mask.any():
                        # Gather inputs and weights for the current expert
                        expert_inputs = x[mask]
                        expert_weights = normalized_weights[:, :, j][mask].unsqueeze(-1)

                        # Call the expert with gathered inputs
                        expert_output = self.experts[expert_idx](expert_inputs)

                        # Scale outputs by weights and add them to the result tensor
                        out[mask] += expert_output * expert_weights
        # Dropout is applied to the final output of the MoE layer to prevent overfitting. The dropout probability
        # is configurable and applied conditionally based on the 'dropout_ratio' parameter.
        if self.dropout_ratio:
            out = self.dropout(out)
        return out

# Summary notes: This implementation of SoftMixture of Experts (SoftMoE) is a powerful extension to transformer-based models,
# particularly for large-scale models where computational efficiency and flebility are critical.  By allowing dynamic selection of experts
# this approach can improve both the training and inference efficiency while maintaining the model's performance. The 
# ability to switch between soft and sparse modes provides a useful trade-off between model espressiveness and computational cost.
