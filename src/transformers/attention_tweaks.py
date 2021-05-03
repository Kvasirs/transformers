import torch
import numpy as np

def adjust_att_scores(attention_weights, w1_i, w2_i, value):

    # Covert attention weight tensor to numpy array for modification.
    np_weights = attention_weights.numpy()

    # For each attention head within block
    for h in range(len(np_weights[0])):
    
        # Get all weights for the given first word at index w1_i.
        w1_weights = np_weights[0][h][w1_i]

        # Select all weights (except for weight at index w2_i) and normalize to 1.
        w1_weights = [w1_weights[i] for i in range(len(w1_weights)) if i != w2_i]
        w1_weights = [float(i) / sum(w1_weights) for i in w1_weights]

        # Multiply by 1 - value, and add new value back in at w1_i.
        # This ensures all weights will sum to 1.
        w1_weights = [w * (1 - value) for w in w1_weights]
        w1_weights.insert(w2_i, value)

        # Set recaculated weights for first word.
        np_weights[0][h][w1_i] = w1_weights

    # Convert back to tensor.
    attention_weights = torch.tensor(np_weights) 
    
    return attention_weights

def adjust_att_vecs(q, k, w1_i, w2_i, sim, mag):

    # Convert Q and K tensors to numpy arrays.
    q_np = q.numpy()
    k_np = k.numpy()

    for i in range(len(k_np[0])):

        # Get Q and K vectors we are targetting.
        Q = q_np[0][i][w1_i] 
        K = k_np[0][i][w2_i]

        # Adjust vector using vector interpolate equation.
        K = sim * Q + (1 - sim) * K

        # Change K to its unit vector (magnitude = 1).
        K = K / np.linalg.norm(K)

        # Multiply by specified scalar value to increase magnitude.
        K *= mag

        k_np[0][i][w2_i] = K

        # Print new cosine similarity between Q and K vectors.
        cos_sim = np.dot(Q, K) / (np.linalg.norm(Q) * np.linalg.norm(K))
        #print(cos_sim)

    # Convert K vectors back to tensors.
    k_vecs = torch.tensor(k_np) 

    return k_vecs



